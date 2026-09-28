"""Testnet Trading Orchestrator for real sandbox execution on Binance Spot Testnet."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.backtesting.signals import SignalAction, SignalGenerator
from app.data.collectors.binance import BinanceClient
from app.data.storage.models import ModelVersion, PortfolioSnapshot, Prediction, Signal
from app.data.storage.repository import CandleRepository
from app.execution.testnet.order_manager import TestnetOrderManager
from app.features.pipeline import FeaturePipeline
from app.features.repository import FeatureRepository
from app.models.inference.predictor import ModelPredictor
from app.models.registry.registry import ModelRegistry
from app.risk.manager import RiskManager
from app.risk.state import OrderRequest

logger = logging.getLogger(__name__)


class TestnetTradingEngine:
    """Orchestrates live trading steps directly on Binance Spot Testnet."""

    __test__ = False

    def __init__(
        self,
        client: BinanceClient,
        risk_manager: Optional[RiskManager] = None,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float = 0.02,
    ):
        # Security Assertion: URL must be Binance Testnet
        if "testnet.binance.vision" not in client.base_url:
            raise ValueError(
                f"SECURITY VIOLATION: TestnetTradingEngine can only run on Binance Testnet (got {client.base_url})"
            )

        self.client = client
        self.risk_manager = risk_manager or RiskManager()
        self.order_manager = TestnetOrderManager(client=client)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.active_model_version: Optional[ModelVersion] = None
        self.predictor: Optional[ModelPredictor] = None

    async def initialize(self, session: AsyncSession) -> bool:
        """Load active model from Model Registry."""
        self.active_model_version = await ModelRegistry.get_active_model(session)
        if self.active_model_version:
            xgb_model = ModelRegistry.load_model_artifact(self.active_model_version)
            self.predictor = ModelPredictor(
                model=xgb_model,
                feature_names=self.active_model_version.feature_names,
                model_version=self.active_model_version.version,
                model_version_id=self.active_model_version.id,
            )
            logger.info(
                f"TestnetTradingEngine initialized with model: {self.active_model_version.version}"
            )
            return True
        else:
            logger.warning("No active model found in Model Registry.")
            return False

    async def process_step(
        self,
        symbol: str,
        timeframe: str,
        session: AsyncSession,
    ) -> Dict[str, Any]:
        """Execute one live testnet bar cycle: Fetch -> Features -> ML -> Risk -> Live Testnet Order."""
        # 1. Fetch latest candles from Binance Testnet
        candles = await self.client.get_klines(symbol=symbol, timeframe=timeframe, limit=100)
        if not candles:
            return {"status": "no_data"}

        await CandleRepository.save_candles(candles, session)
        latest_candle = candles[-1]
        ts = latest_candle.timestamp

        # 2. Query real account balance & build live PortfolioState
        portfolio_state = await self.order_manager.fetch_testnet_portfolio_state(
            symbol=symbol,
            current_price=latest_candle.close,
            session=session,
        )

        # 3. Extract features
        df_candles = await CandleRepository.get_as_dataframe(
            symbol=symbol,
            timeframe=timeframe,
            as_of_time=ts,
            session=session,
        )
        features_df = FeaturePipeline.compute_features(df_candles, drop_na=False)

        step_res: Dict[str, Any] = {
            "timestamp": ts.isoformat(),
            "symbol": symbol,
            "close_price": latest_candle.close,
            "testnet_cash": portfolio_state.cash,
            "testnet_equity": portfolio_state.equity,
            "signal": None,
            "order_placed": None,
        }

        if not features_df.empty and not features_df.iloc[-1].isnull().any():
            latest_feat_dict = features_df.iloc[-1].to_dict()
            await FeatureRepository.save_features(
                features_df.iloc[[-1]], symbol, timeframe, session
            )

            # 4. ML Inference
            if self.predictor is None:
                await self.initialize(session)

            if self.predictor is not None:
                model_ver_id = self.active_model_version.id if self.active_model_version else None
                pred_result = self.predictor.predict_single(
                    features=latest_feat_dict,
                    timestamp=ts,
                    symbol=symbol,
                    timeframe=timeframe,
                )

                pred_obj = Prediction(
                    timestamp=ts,
                    symbol=symbol,
                    timeframe=timeframe,
                    model_version_id=model_ver_id,
                    probability_up=pred_result.probability_up,
                    probability_down=pred_result.probability_down,
                    predicted_class=pred_result.predicted_class,
                    horizon=pred_result.horizon,
                    raw_features=latest_feat_dict,
                )
                session.add(pred_obj)
                await session.flush()

                # 5. Signal
                signal = SignalGenerator.generate_signal_from_prediction(
                    prediction=pred_result,
                    buy_threshold=self.buy_threshold,
                    sell_threshold=self.sell_threshold,
                )

                confidence_val = abs(pred_result.probability_up - 0.5) * 2.0
                sig_obj = Signal(
                    timestamp=ts,
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type=signal.action.value,
                    probability_up=pred_result.probability_up,
                    probability_down=pred_result.probability_down,
                    confidence=confidence_val,
                    threshold_used=self.buy_threshold,
                    prediction_id=pred_obj.id,
                    is_processed=False,
                )
                session.add(sig_obj)
                await session.flush()
                step_res["signal"] = signal.action.value

                # 6. Risk Check & Testnet Execution
                has_position = symbol in portfolio_state.open_positions and portfolio_state.open_positions[symbol] > 0

                if signal.action == SignalAction.BUY and not has_position:
                    proposed_qty = (portfolio_state.cash * 0.95) / latest_candle.close
                    sl_price = latest_candle.close * (1.0 - self.stop_loss_pct)
                    tp_price = latest_candle.close * (1.0 + self.take_profit_pct)

                    order_req = OrderRequest(
                        symbol=symbol,
                        side="BUY",
                        quantity=proposed_qty,
                        price=latest_candle.close,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                        model_version=self.active_model_version.version if self.active_model_version else None,
                        signal_id=sig_obj.id,
                        timestamp=ts,
                    )

                    risk_check = self.risk_manager.validate_order(order_req, portfolio_state)
                    if risk_check.allowed:
                        sig_obj.is_processed = True
                        order, fill, _ = await self.order_manager.execute_testnet_buy_order(
                            order_req=order_req,
                            approved_qty=risk_check.approved_quantity,
                            session=session,
                            model_version_id=model_ver_id,
                        )
                        step_res["order_placed"] = {
                            "order_id": order.id,
                            "exchange_id": order.exchange_order_id,
                            "side": "BUY",
                            "qty": fill.quantity if fill else order.quantity,
                            "price": fill.price if fill else order.price,
                        }
                    else:
                        sig_obj.is_processed = True
                        await RiskManager.record_risk_event(
                            event_type=risk_check.rule_triggered or "TESTNET_ORDER_REJECTED",
                            reason="; ".join(risk_check.reasons),
                            symbol=symbol,
                            severity=risk_check.severity.value,
                            details={"order_price": latest_candle.close, "proposed_qty": proposed_qty},
                            session=session,
                        )

                elif signal.action == SignalAction.SELL and has_position:
                    pos_qty = portfolio_state.open_positions[symbol]
                    order, fill, trade = await self.order_manager.execute_testnet_sell_order(
                        symbol=symbol,
                        quantity=pos_qty,
                        market_price=latest_candle.close,
                        exit_reason="signal_exit",
                        session=session,
                        model_version_id=model_ver_id,
                        signal_id=sig_obj.id,
                    )
                    sig_obj.is_processed = True
                    step_res["order_placed"] = {
                        "order_id": order.id,
                        "exchange_id": order.exchange_order_id,
                        "side": "SELL",
                        "qty": fill.quantity if fill else order.quantity,
                        "pnl": trade.pnl if trade else 0.0,
                    }

        # 7. Portfolio Snapshot in DB
        snapshot = PortfolioSnapshot(
            timestamp=ts,
            mode="testnet",
            total_equity=portfolio_state.equity,
            cash_balance=portfolio_state.cash,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            peak_equity=portfolio_state.peak_equity,
            current_drawdown_pct=portfolio_state.current_drawdown_pct,
            daily_pnl=0.0,
            daily_return_pct=0.0,
            open_positions_count=len(portfolio_state.open_positions),
        )
        session.add(snapshot)
        await session.commit()

        return step_res
