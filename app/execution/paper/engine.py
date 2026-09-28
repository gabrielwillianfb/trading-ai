"""Paper Trading Orchestration Engine integrating collectors, features, ML inference, risk, and execution."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.backtesting.signals import SignalAction, SignalGenerator, TradingSignal
from app.data.collectors.base import CandleData
from app.data.storage.models import (
    ModelVersion,
    PortfolioSnapshot,
    Prediction,
    Signal,
)
from app.data.storage.repository import CandleRepository
from app.execution.paper.account import VirtualAccount
from app.execution.paper.order_manager import PaperOrderManager
from app.features.pipeline import FeaturePipeline
from app.features.repository import FeatureRepository
from app.models.inference.predictor import ModelPredictor
from app.models.registry.registry import ModelRegistry
from app.risk.manager import RiskManager
from app.risk.state import OrderRequest, RiskAction

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """Orchestrates end-to-end simulated live trading steps with full database audit trails."""

    def __init__(
        self,
        account: Optional[VirtualAccount] = None,
        risk_manager: Optional[RiskManager] = None,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float = 0.02,
        max_holding_bars: int = 12,
    ):
        self.account = account or VirtualAccount()
        self.risk_manager = risk_manager or RiskManager()
        self.order_manager = PaperOrderManager()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_bars = max_holding_bars

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
                f"PaperTradingEngine initialized with active model: {self.active_model_version.version}"
            )
            return True
        else:
            logger.warning("No active model found in Model Registry.")
            return False

    async def process_candle(
        self,
        candle: CandleData,
        session: AsyncSession,
    ) -> Dict[str, Any]:
        """Execute one complete bar lifecycle step.

        Lifecycle:
            1. Save candle to DB.
            2. Check Intrabar SL/TP for open positions.
            3. Query historical bar window and compute technical features.
            4. Run ML model probability inference.
            5. Map probability to discrete TradingSignal.
            6. Validate proposed order with independent RiskManager.
            7. If authorized, simulate order fill in VirtualAccount.
            8. Record PortfolioSnapshot.
        """
        symbol = candle.symbol
        timeframe = candle.timeframe
        ts = candle.timestamp

        step_log: Dict[str, Any] = {
            "timestamp": ts.isoformat(),
            "symbol": symbol,
            "close_price": candle.close,
            "intrabar_exit": None,
            "prediction": None,
            "signal": None,
            "risk_decision": None,
            "order_filled": None,
            "equity": None,
        }

        # -------------------------------------------------------------
        # 1. Check Intrabar SL/TP on active positions
        # -------------------------------------------------------------
        model_ver_id = self.active_model_version.id if self.active_model_version else None
        intrabar_trade = await self.order_manager.check_intrabar_exits(
            account=self.account,
            symbol=symbol,
            high=candle.high,
            low=candle.low,
            open_price=candle.open,
            timestamp=ts,
            session=session,
            model_version_id=model_ver_id,
            max_holding_bars=self.max_holding_bars,
        )
        if intrabar_trade:
            step_log["intrabar_exit"] = {
                "exit_reason": intrabar_trade.exit_reason,
                "net_pnl": intrabar_trade.pnl,
            }

        # -------------------------------------------------------------
        # 2. Persist Candle to Database
        # -------------------------------------------------------------
        await CandleRepository.save_candles([candle], session)

        # -------------------------------------------------------------
        # 3. Fetch History & Compute Features
        # -------------------------------------------------------------
        df_candles = await CandleRepository.get_as_dataframe(
            symbol=symbol,
            timeframe=timeframe,
            as_of_time=ts,
            session=session,
        )

        features_df = FeaturePipeline.compute_features(df_candles, drop_na=False)

        # Check if latest row has computed features (no NaNs)
        if not features_df.empty and not features_df.iloc[-1].isnull().any():
            latest_features_row = features_df.iloc[-1]
            features_dict = latest_features_row.to_dict()

            # Save features to DB table
            await FeatureRepository.save_features(
                features_df.iloc[[-1]], symbol, timeframe, session
            )

            # ---------------------------------------------------------
            # 4. ML Inference
            # ---------------------------------------------------------
            if self.predictor is None:
                await self.initialize(session)

            if self.predictor is not None:
                pred_result = self.predictor.predict_single(
                    features=features_dict,
                    timestamp=ts,
                    symbol=symbol,
                    timeframe=timeframe,
                )

                # Persist Prediction record in database
                pred_obj = Prediction(
                    timestamp=ts,
                    symbol=symbol,
                    timeframe=timeframe,
                    model_version_id=model_ver_id,
                    probability_up=pred_result.probability_up,
                    probability_down=pred_result.probability_down,
                    predicted_class=pred_result.predicted_class,
                    horizon=pred_result.horizon,
                    raw_features=features_dict,
                )
                session.add(pred_obj)
                await session.flush()
                step_log["prediction"] = pred_result.to_dict()

                # -----------------------------------------------------
                # 5. Discrete Signal Generation
                # -----------------------------------------------------
                signal = SignalGenerator.generate_signal_from_prediction(
                    prediction=pred_result,
                    buy_threshold=self.buy_threshold,
                    sell_threshold=self.sell_threshold,
                )

                # Persist Signal record in database
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
                step_log["signal"] = signal.action.value

                # -----------------------------------------------------
                # 6. Risk Validation & Order Execution
                # -----------------------------------------------------
                pos = self.account.get_position(symbol)

                if signal.action == SignalAction.BUY and pos is None:
                    # Calculate proposed order parameters
                    proposed_qty = (self.account.cash * 0.95) / candle.close
                    sl_price = candle.close * (1.0 - self.stop_loss_pct)
                    tp_price = candle.close * (1.0 + self.take_profit_pct)

                    order_req = OrderRequest(
                        symbol=symbol,
                        side="BUY",
                        quantity=proposed_qty,
                        price=candle.close,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                        model_version=self.active_model_version.version,
                        signal_id=sig_obj.id,
                        timestamp=ts,
                    )

                    # Validate with RiskManager
                    portfolio_state = self.account.get_portfolio_state()
                    risk_check = self.risk_manager.validate_order(order_req, portfolio_state)
                    step_log["risk_decision"] = risk_check.action.value

                    if risk_check.allowed:
                        sig_obj.is_processed = True
                        order, fill, _ = await self.order_manager.create_and_fill_buy_order(
                            account=self.account,
                            order_req=order_req,
                            approved_qty=risk_check.approved_quantity,
                            session=session,
                            model_version_id=model_ver_id,
                        )
                        step_log["order_filled"] = {
                            "order_id": order.id,
                            "side": "BUY",
                            "qty": fill.quantity,
                            "price": fill.price,
                        }
                    else:
                        sig_obj.is_processed = True
                        await RiskManager.record_risk_event(
                            event_type=risk_check.rule_triggered or "ORDER_REJECTED",
                            reason="; ".join(risk_check.reasons),
                            symbol=symbol,
                            severity=risk_check.severity.value,
                            details={"order_price": candle.close, "proposed_qty": proposed_qty},
                            session=session,
                        )

                elif signal.action == SignalAction.SELL and pos is not None:
                    order, fill, trade = await self.order_manager.create_and_fill_sell_order(
                        account=self.account,
                        symbol=symbol,
                        quantity=pos.quantity,
                        market_price=candle.close,
                        exit_reason="signal_exit",
                        session=session,
                        model_version_id=model_ver_id,
                        signal_id=sig_obj.id,
                        timestamp=ts,
                    )
                    sig_obj.is_processed = True
                    step_log["order_filled"] = {
                        "order_id": order.id,
                        "side": "SELL",
                        "qty": fill.quantity,
                        "price": fill.price,
                        "net_pnl": trade.pnl,
                    }

        # -------------------------------------------------------------
        # 7. Update Portfolio Price & Snapshot
        # -------------------------------------------------------------
        self.account.update_price(symbol, candle.close)
        tot_equity = self.account.get_total_equity()
        unrealized = self.account.get_unrealized_pnl()
        peak_eq = self.account.peak_equity
        drawdown_pct = max(0.0, ((peak_eq - tot_equity) / peak_eq) * 100.0) if peak_eq > 0 else 0.0

        snapshot = PortfolioSnapshot(
            timestamp=ts,
            mode="paper",
            total_equity=tot_equity,
            cash_balance=self.account.cash,
            unrealized_pnl=unrealized,
            realized_pnl=self.account.realized_pnl,
            peak_equity=peak_eq,
            current_drawdown_pct=drawdown_pct,
            daily_pnl=self.account.daily_pnl,
            daily_return_pct=(self.account.daily_pnl / self.account.initial_cash) * 100.0,
            open_positions_count=len(self.account.positions),
        )
        session.add(snapshot)
        await session.commit()

        step_log["equity"] = tot_equity
        return step_log
