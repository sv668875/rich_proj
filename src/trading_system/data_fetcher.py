"""Historical data fetching utilities.

歷史資料擷取工具：負責從交易所或檔案讀取 K 線 | Fetch OHLCV data for backtests.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List

import numpy as np
import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)


@dataclass
class HistoricalDataFetcher:
    """簡化資料抓取器 | Lightweight OHLCV fetcher."""

    symbol: str
    interval: str
    source: str = "local_csv"

    BINANCE_ENDPOINT: str = "https://api.binance.com/api/v3/klines"
    BINANCE_LIMIT: int = 1000
    BINANCE_TESTNET_ENDPOINT: str = "https://testnet.binance.vision/api/v3/klines"

    def fetch_ohlcv(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Return placeholder OHLCV frame.

        此為示意版，實務上可改為調用實際 API 或讀取快取檔 | Replace with real exchange integration.
        """

        source_name = self.source.lower()
        if source_name in {"binance", "binance_rest", "binance_testnet"}:
            try:
                return self._fetch_from_binance(start=start, end=end)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.error("Binance 資料抓取失敗，改用示意資料 | Fallback to mock data: %s", exc)

        index = pd.date_range(start=start, end=end, freq="h")
        prices = pd.Series(range(len(index)), index=index, dtype="float64")
        prices += 5 * np.sin(np.linspace(0, 6.28, len(index)))
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices + 5,
                "low": prices - 5,
                "close": prices + 1,
                "volume": 1000,
            }
        )
        df.index.name = "timestamp"
        return df.reset_index()

    def resample(self, candles: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Resample OHLCV to new interval.

        重新取樣 K 線資料以符合策略需求 | Resample candles to match strategy timeframe.
        """

        candles = candles.copy()
        candles["timestamp"] = pd.to_datetime(candles["timestamp"])
        candles.set_index("timestamp", inplace=True)
        agg = candles.resample(interval).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        return agg.dropna().reset_index()

    def _resolve_endpoint(self) -> str:
        if self.source.lower() == "binance_testnet":
            return self.BINANCE_TESTNET_ENDPOINT
        return self.BINANCE_ENDPOINT

    def _fetch_from_binance(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLCV data from Binance public REST API."""

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        klines: List[List[str]] = []
        fetch_start = start_ms
        endpoint = self._resolve_endpoint()

        while fetch_start < end_ms:
            params = {
                "symbol": self.symbol.upper(),
                "interval": self.interval,
                "startTime": fetch_start,
                "endTime": end_ms,
                "limit": self.BINANCE_LIMIT,
            }
            headers = {}
            api_env_key = "BINANCE_TESTNET_API_KEY" if self.source.lower() == "binance_testnet" else "BINANCE_API_KEY"
            api_key = os.getenv(api_env_key)
            if api_key:
                headers["X-MBX-APIKEY"] = api_key
            response = requests.get(endpoint, params=params, headers=headers or None, timeout=10)
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break

            klines.extend(batch)
            last_close_time = batch[-1][6]
            fetch_start = int(last_close_time) + 1
            if len(batch) < self.BINANCE_LIMIT:
                break

        if not klines:
            raise ValueError("Binance API 未返回任何資料 | Binance API returned no data")

        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        df = pd.DataFrame(klines, columns=columns)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.sort_values("timestamp", inplace=True)
        df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
        return df.reset_index(drop=True)


def merge_data_sources(sources: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple data sources sequentially.

    整併多來源資料，優先保留時間序一致性 | Concatenate data sources while preserving chronological order.
    """

    merged = pd.concat(sources, ignore_index=True)
    merged.sort_values("timestamp", inplace=True)
    merged.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    return merged.reset_index(drop=True)
