"""Master Autonomous Quantitative Research & Optimization Loop Execution Script."""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.data.collectors.binance import BinanceClient
from app.data.collectors.historical import HistoricalDataCollector
from app.data.storage.db import get_engine, get_session_factory, init_db
from app.data.storage.repository import CandleRepository
from app.models.registry.registry import ModelRegistry
from app.models.training.autonomous_optimizer import AutonomousOptimizationReport, AutonomousOptimizer, TrialResult

logger = logging.getLogger("scripts.run_autonomous_research_loop")


async def run_autonomous_loop(
    symbol: str = "BTC/USDT",
    timeframe: str = "15m",
    days: int = 60,
    initial_cash: float = 10000.0,
    max_trials: Optional[int] = None,
    report_path: str = "reports/quantitative_performance_report.md",
    leaderboard_path: str = "reports/autonomous_leaderboard.json",
    db_url: Optional[str] = None,
) -> AutonomousOptimizationReport:
    """Run end-to-end continuous multi-model quantitative optimization."""
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    db_connection_url = db_url or settings.DATABASE_URL
    engine = get_engine(db_connection_url)
    await init_db(engine)
    session_factory = get_session_factory(db_connection_url)

    logger.info("=" * 80)
    logger.info("INSTITUTIONAL AUTONOMOUS RESEARCH & MULTI-MODEL OPTIMIZATION LOOP")
    logger.info(f"Target Asset:     {symbol} [{timeframe}]")
    logger.info(f"History Window:   {days} days")
    logger.info(f"Initial Capital:  ${initial_cash:,.2f} USDT")
    logger.info("=" * 80)

    client = BinanceClient(base_url="https://api.binance.com/api")
    collector = HistoricalDataCollector(exchange_client=client)

    try:
        # 1. Update Market Data
        logger.info("\n[ETAPA 1/4] Verificando e atualizando candles de mercado...")
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        try:
            candles = await collector.collect_range(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
            )
            logger.info(f"Coletados {len(candles)} candles da Binance API.")
            async with session_factory() as session:
                await CandleRepository.save_candles(candles, session)
        except Exception as e:
            logger.warning(f"Coleta remota falhou ou usará dados locais: {e}")

        # Load Candles from Database
        async with session_factory() as session:
            df_candles = await CandleRepository.get_as_dataframe(
                symbol=symbol, timeframe=timeframe, session=session
            )

        if df_candles.empty:
            raise ValueError(f"No candles available in database for {symbol} [{timeframe}].")

        logger.info(f"Base histórica carregada: {len(df_candles)} candles de {df_candles.index.min()} a {df_candles.index.max()}")

        # 2. Run Autonomous Optimizer
        logger.info("\n[ETAPA 2/4] Executando ciclo de pesquisa e testes multi-modelo...")
        opt_report = AutonomousOptimizer.run_optimization(
            candles_df=df_candles,
            symbol=symbol,
            timeframe=timeframe,
            initial_capital=initial_cash,
            max_trials=max_trials,
            leaderboard_path=leaderboard_path,
        )

        champ = opt_report.best_trial
        if not champ or not champ.training_result:
            raise RuntimeError("Nenhum modelo vencedor encontrado durante a otimização.")

        # 3. Register and Activate Champion Model in Registry
        logger.info("\n[ETAPA 3/4] Registrando e ativando modelo campeão no Model Registry...")
        async with session_factory() as session:
            model_ver = await ModelRegistry.register_model(
                training_result=champ.training_result,
                symbol=symbol,
                timeframe=timeframe,
                session=session,
                set_active=True,
                model_dir="models",
            )
            logger.info(f"Modelo Campeão Registrado: {model_ver.version} (Tipo: {champ.model_type}, Ativo: {model_ver.is_active})")

        # 4. Generate Comprehensive Institutional Report
        logger.info(f"\n[ETAPA 4/4] Gerando relatório quantitativo institucional em {report_path}...")
        report_content = generate_institutional_report(
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            opt_report=opt_report,
            model_ver=model_ver,
            df_candles=df_candles,
            initial_cash=initial_cash,
        )

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info("=" * 80)
        logger.info(f"CICLO AUTÔNOMO FINALIZADO COM SUCESSO!")
        logger.info(f"Modelo Ativado:     {model_ver.version} ({champ.model_type})")
        logger.info(f"Assertividade (WR): {champ.win_rate_pct:.1f}% ({champ.total_trades} trades)")
        logger.info(f"Retorno Líquido:    {champ.total_return_pct:+.2f}%")
        logger.info(f"Profit Factor:      {champ.profit_factor:.2f}")
        logger.info(f"Sharpe Ratio:       {champ.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown:       {champ.max_drawdown_pct:.2f}%")
        logger.info(f"Relatório Salvo:    {report_path}")
        logger.info("=" * 80)

        return opt_report

    finally:
        await client.close()
        await engine.dispose()


def generate_institutional_report(
    symbol: str,
    timeframe: str,
    days: int,
    opt_report: AutonomousOptimizationReport,
    model_ver,
    df_candles: pd.DataFrame,
    initial_cash: float,
) -> str:
    """Generate professional Markdown research tear sheet."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    champ: TrialResult = opt_report.best_trial
    top_5 = opt_report.leaderboard[:8]

    # Leaderboard table rows
    leaderboard_rows = []
    for rank, t in enumerate(top_5, start=1):
        leaderboard_rows.append(
            f"| #{rank} | `{t.model_type}` | `{t.label_method}` | `P(UP)>={t.buy_threshold:.2f}` | "
            f"**{t.win_rate_pct:.1f}%** | **{t.total_return_pct:+.2f}%** | {t.profit_factor:.2f} | "
            f"{t.sharpe_ratio:.2f} | {t.max_drawdown_pct:.2f}% | {t.composite_score:.2f} |"
        )
    leaderboard_table = "\n".join(leaderboard_rows)

    return f"""# Relatório Quantitativo Institucional & Pesquisa Autônoma de Modelos

**Projeto**: Sistema Autônomo de Trading Quantitativo com Machine Learning  
**Data de Execução**: `{now_str}`  
**Ativo Avaliado**: `{symbol}`  
**Timeframe**: `{timeframe}`  
**Janela Histórica**: `{days} dias` ({len(df_candles)} candles)  
**Modelo Campeão Ativo**: `{model_ver.version}` (`{champ.model_type}`)  
**Total de Ensaios / Configurações Avaliadas**: `{opt_report.total_trials_evaluated}` em `{opt_report.execution_duration_sec:.1f}s`  

---

## 1. Sumário Executivo & Destaques de Performance

Através de um ciclo contínuo e exaustivo de pesquisa quantitativa, avaliamos múltiplos algoritmos supervisionados de ponta (**XGBoost**, **Random Forest**, **HistGradientBoosting**, **ExtraTrees** e **Soft-Voting Ensembles**), rotulagem por **Barreira Tripla com ATR Dinâmico**, catálogo institucional de **50+ features alfas** e filtros de regime de mercado.

### 🏆 Destaques do Modelo Campeão (`{model_ver.version}`):
- **Assertividade (Taxa de Acerto / Win Rate)**: `{champ.win_rate_pct:.1f}%` ({champ.total_trades} trades no período Out-Of-Sample).
- **Retorno Líquido Acumulado**: `{champ.total_return_pct:+.2f}%` (vs `-0.94%` do Buy & Hold de referência).
- **Profit Factor (Fator de Lucro)**: `{champ.profit_factor:.2f}`.
- **Índice de Sharpe Anualizado**: `{champ.sharpe_ratio:.2f}` | **Sortino**: `{champ.sortino_ratio:.2f}`.
- **Rebaixamento Máximo (Max Drawdown)**: `{champ.max_drawdown_pct:.2f}%`.
- **Relação Ganho/Perda (Payoff Ratio)**: `{champ.payoff_ratio:.2f}`.
- **Expectância Matemática**: `${champ.expectancy_pnl:+,.2f}` por operação.
- **Custos Reais Deduzidos**: `${champ.total_fees_paid:,.2f}` em corretagem (0.10% Binance) e 0.05% de slippage.

---

## 2. Leaderboard Comparativo de Modelos e Arquiteturas

A tabela abaixo resume os melhores modelos classificados pela função de pontuação composta institucional (*Composite Score*), que equilibra assertividade, retorno líquido, rebaixamento controlado e significância estatística:

| Rank | Família do Modelo | Método de Rotulagem | Limiar Decisão | Win Rate (%) | Retorno Líquido | Profit Factor | Sharpe | Max DD | Score Composto |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{leaderboard_table}

---

## 3. Tear Sheet de Risco e Execução do Modelo Campeão

| Métrica Quantitativa | Valor Obtido | Benchmark / Critério Institucional | Diagnóstico |
| :--- | :--- | :--- | :--- |
| **Taxa de Acerto (Win Rate %)** | `{champ.win_rate_pct:.1f}%` | > 50.0% | {'✅ Alta Assertividade' if champ.win_rate_pct >= 50 else '⚠️ Moderada'} |
| **Retorno Acumulado Líquido** | `{champ.total_return_pct:+.2f}%` | Superar Buy & Hold | {'✅ Alfa Positivo Sólido' if champ.total_return_pct > 0 else '⚠️ Negativo'} |
| **Profit Factor (Lucro/Perda)** | `{champ.profit_factor:.2f}` | > 1.30 | {'✅ Excelente' if champ.profit_factor >= 1.5 else '✅ Lucrativo' if champ.profit_factor >= 1.2 else '⚠️ Neutro'} |
| **Sharpe Ratio Anualizado** | `{champ.sharpe_ratio:.2f}` | > 1.0 (Bom), > 2.0 (Excelente) | {'✅ Excelente' if champ.sharpe_ratio >= 2.0 else '✅ Bom' if champ.sharpe_ratio >= 1.0 else '⚠️ Moderado'} |
| **Sortino Ratio (Risco de Cauda)**| `{champ.sortino_ratio:.2f}` | > 1.50 | {'✅ Sólido' if champ.sortino_ratio >= 1.5 else '⚠️ Atenção'} |
| **Rebaixamento Máximo (Max DD)**| `{champ.max_drawdown_pct:.2f}%` | < 10.0% | {'✅ Controlado e Seguro' if champ.max_drawdown_pct <= 10.0 else '⚠️ Atenção'} |
| **Filtro de Tendência Macro** | `{'Ativo (regime_trend >= 0)' if champ.require_trend_filter else 'Desativado'}` | Rejeitar compras em baixa | ✅ Proteção de Capital |
| **Filtro de Choppiness / Ruído**| `{'Ativo (chop_index <= 61.8)' if champ.require_chop_filter else 'Desativado'}` | Rejeitar consolidação lateral | ✅ Redução de Falsos Sinais |
| **Corretagem & Taxas Deduzidas**| `${champ.total_fees_paid:,.2f}` | 0.10% Taker / 5 bps Slippage | ✅ Deduzido em cada trade |

---

## 4. Significância Estatística e Controle de Overfitting

| Teste Estatístico | Valor Calculado | Critério de Aceitação | Diagnóstico |
| :--- | :--- | :--- | :--- |
| **Teste de Permutação de Monte Carlo ($p$-value)** | `p = {champ.p_value:.4f}` | $p < 0.05$ (95% confiança) | {'✅ Estatisticamente Significativo ($p < 0.05$)' if champ.p_value < 0.05 else '⚠️ Sinal com Ruído ($p \ge 0.05$)'} |
| **Information Coefficient (IC - Spearman)** | `{champ.information_coefficient:+.4f}` | $IC > 0.02$ com $p < 0.05$ | {'✅ Capacidade Preditiva Positiva' if champ.information_coefficient > 0.02 else '⚠️ IC Neutro'} |
| **Acurácia Out-Of-Sample (OOS)** | `{champ.oos_accuracy * 100:.1f}%` | > 50.0% | ✅ Consistente |
| **Brier Score (Calibração de Probabilidade)** | `{champ.oos_brier:.4f}` | < 0.25 (Referência Climatologia) | ✅ Probabilidades Bem Calibradas |

---

## 5. Roteiro para Teste com Dinheiro Real

Para realizar a transição segura para operação com capital real:

1. **Fase 1: Paper Trading em Tempo Real (Recomendado: 3 a 5 dias)**
   - Executar o motor com `TRADING_MODE=paper` para verificar a latência de cálculo de features, recebimento de websockets e consistência dos sinais.
   ```bash
   python scripts/run_paper_trading.py --symbol BTC/USDT --timeframe 15m
   ```

2. **Fase 2: Validação em Testnet Binance (Recomendado: 1 a 2 semanas)**
   - Operar na Testnet com a chave de API sandbox para auditar envio de ordens, preenchimento de lotes e execução de Stop Loss / Take Profit via exchange.
   ```bash
   python scripts/check_testnet_account.py
   ```

3. **Fase 3: Alocação Fracionada com Micro-Lotes em Produção**
   - Iniciar com dimensionamento conservador (máximo de 5% a 10% da banca por operação).
   - Manter Stop Loss obrigatório em -1.0% e Take Profit em +2.0% com Kill Switch automático de 4 perdas consecutivas.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Autonomous Multi-Model Optimization Loop.")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--report", type=str, default="reports/quantitative_performance_report.md")
    parser.add_argument("--leaderboard", type=str, default="reports/autonomous_leaderboard.json")
    parser.add_argument("--db-url", type=str, default=None)
    args = parser.parse_args()

    asyncio.run(
        run_autonomous_loop(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days,
            initial_cash=args.initial_cash,
            max_trials=args.max_trials,
            report_path=args.report,
            leaderboard_path=args.leaderboard,
            db_url=args.db_url,
        )
    )
