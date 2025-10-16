"""Live execution blueprint.

實盤執行藍圖：處理交易所 API 下單與錯誤管理 | Outline for live execution module.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict


LOGGER = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    """下單請求結構 | Standardized order request."""

    symbol: str
    side: str
    quantity: float
    order_type: str = "MARKET"
    price: float | None = None


class LiveExecutor:
    """Exchange execution orchestrator.

    統籌交易所下單流程：負責調用 API、處理重試與風控檢查 | Manage order placement workflow.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def place_order(self, request: OrderRequest) -> Dict[str, Any]:
        """Send order to exchange.

        送出訂單，並處理常見錯誤，確保策略穩定執行 | Place order with basic retry handling.
        """

        LOGGER.info("準備下單 | Preparing order: %s", request)
        try:
            response = self.client.create_order(
                symbol=request.symbol,
                side=request.side.upper(),
                type=request.order_type.upper(),
                quantity=request.quantity,
                price=request.price,
            )
            LOGGER.info("下單成功 | Order placed successfully: %s", response)
            return response
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error("下單失敗 | Order failed: %s", exc)
            raise
