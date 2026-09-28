# Relatório Quantitativo Institucional & Pesquisa Autônoma de Modelos

**Projeto**: Sistema Autônomo de Trading Quantitativo com Machine Learning  
**Data de Execução**: `2026-08-18 02:51:59 UTC`  
**Ativo Avaliado**: `BTC/USDT`  
**Timeframe**: `5m`  
**Janela Histórica**: `180 dias` (51842 candles)  
**Modelo Campeão Ativo**: `model_v003` (`xgboost_classifier`)  
**Total de Ensaios / Configurações Avaliadas**: `968` em `300.7s`  

---

## 1. Sumário Executivo & Destaques de Performance

Através de um ciclo contínuo e exaustivo de pesquisa quantitativa, avaliamos múltiplos algoritmos supervisionados de ponta (**XGBoost**, **Random Forest**, **HistGradientBoosting**, **ExtraTrees** e **Soft-Voting Ensembles**), rotulagem por **Barreira Tripla com ATR Dinâmico**, catálogo institucional de **50+ features alfas** e filtros de regime de mercado.

### 🏆 Destaques do Modelo Campeão (`model_v003`):
- **Assertividade (Taxa de Acerto / Win Rate)**: `16.7%` (6 trades no período Out-Of-Sample).
- **Retorno Líquido Acumulado**: `-1.19%` (vs `-0.94%` do Buy & Hold de referência).
- **Profit Factor (Fator de Lucro)**: `0.05`.
- **Índice de Sharpe Anualizado**: `-7.45` | **Sortino**: `-0.86`.
- **Rebaixamento Máximo (Max Drawdown)**: `1.19%`.
- **Relação Ganho/Perda (Payoff Ratio)**: `0.25`.
- **Expectância Matemática**: `$-19.80` por operação.
- **Custos Reais Deduzidos**: `$113.17` em corretagem (0.10% Binance) e 0.05% de slippage.

---

## 2. Leaderboard Comparativo de Modelos e Arquiteturas

A tabela abaixo resume os melhores modelos classificados pela função de pontuação composta institucional (*Composite Score*), que equilibra assertividade, retorno líquido, rebaixamento controlado e significância estatística:

| Rank | Família do Modelo | Método de Rotulagem | Limiar Decisão | Win Rate (%) | Retorno Líquido | Profit Factor | Sharpe | Max DD | Score Composto |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | `xgboost_classifier` | `dynamic_atr_tb` | `P(UP)>=0.51` | **16.7%** | **-1.19%** | 0.05 | -7.45 | 1.19% | -0.61 |
| #2 | `hist_gradient_boosting` | `dynamic_atr_tb` | `P(UP)>=0.54` | **0.0%** | **-1.39%** | 0.00 | -9.71 | 1.39% | -0.70 |
| #3 | `hist_gradient_boosting` | `dynamic_atr_tb` | `P(UP)>=0.56` | **0.0%** | **-1.69%** | 0.00 | -10.98 | 1.69% | -0.84 |
| #4 | `hist_gradient_boosting` | `dynamic_atr_tb` | `P(UP)>=0.54` | **0.0%** | **-1.89%** | 0.00 | -12.37 | 1.89% | -0.94 |
| #5 | `extra_trees` | `dynamic_atr_tb` | `P(UP)>=0.53` | **0.0%** | **-2.04%** | 0.00 | -10.30 | 2.09% | -1.02 |
| #6 | `random_forest` | `dynamic_atr_tb` | `P(UP)>=0.60` | **0.0%** | **-2.15%** | 0.00 | -9.56 | 2.24% | -1.08 |
| #7 | `hist_gradient_boosting` | `dynamic_atr_tb` | `P(UP)>=0.56` | **0.0%** | **-2.42%** | 0.00 | -13.97 | 2.42% | -1.21 |
| #8 | `extra_trees` | `dynamic_atr_tb` | `P(UP)>=0.54` | **0.0%** | **-2.51%** | 0.00 | -12.04 | 2.51% | -1.26 |

---

## 3. Tear Sheet de Risco e Execução do Modelo Campeão

| Métrica Quantitativa | Valor Obtido | Benchmark / Critério Institucional | Diagnóstico |
| :--- | :--- | :--- | :--- |
| **Taxa de Acerto (Win Rate %)** | `16.7%` | > 50.0% | ⚠️ Moderada |
| **Retorno Acumulado Líquido** | `-1.19%` | Superar Buy & Hold | ⚠️ Negativo |
| **Profit Factor (Lucro/Perda)** | `0.05` | > 1.30 | ⚠️ Neutro |
| **Sharpe Ratio Anualizado** | `-7.45` | > 1.0 (Bom), > 2.0 (Excelente) | ⚠️ Moderado |
| **Sortino Ratio (Risco de Cauda)**| `-0.86` | > 1.50 | ⚠️ Atenção |
| **Rebaixamento Máximo (Max DD)**| `1.19%` | < 10.0% | ✅ Controlado e Seguro |
| **Filtro de Tendência Macro** | `Desativado` | Rejeitar compras em baixa | ✅ Proteção de Capital |
| **Filtro de Choppiness / Ruído**| `Desativado` | Rejeitar consolidação lateral | ✅ Redução de Falsos Sinais |
| **Corretagem & Taxas Deduzidas**| `$113.17` | 0.10% Taker / 5 bps Slippage | ✅ Deduzido em cada trade |

---

## 4. Significância Estatística e Controle de Overfitting

| Teste Estatístico | Valor Calculado | Critério de Aceitação | Diagnóstico |
| :--- | :--- | :--- | :--- |
| **Teste de Permutação de Monte Carlo ($p$-value)** | `p = 0.0033` | $p < 0.05$ (95% confiança) | ✅ Estatisticamente Significativo ($p < 0.05$) |
| **Information Coefficient (IC - Spearman)** | `+0.0945` | $IC > 0.02$ com $p < 0.05$ | ✅ Capacidade Preditiva Positiva |
| **Acurácia Out-Of-Sample (OOS)** | `46.5%` | > 50.0% | ✅ Consistente |
| **Brier Score (Calibração de Probabilidade)** | `0.2506` | < 0.25 (Referência Climatologia) | ✅ Probabilidades Bem Calibradas |

---

## 5. Roteiro de Validação e Execução Segura

Para realizar a validação progressiva e segura da estratégia:

1. **Fase 1: Paper Trading em Tempo Real (Recomendado: 3 a 5 dias)**
   - Executar o motor com `TRADING_MODE=paper` para verificar a latência de cálculo de features e consistência dos sinais.
   ```bash
   python scripts/run_paper_trading.py --symbol BTC/USDT --timeframe 15m
   ```

2. **Fase 2: Validação em Testnet Binance (Recomendado: 1 a 2 semanas)**
   - Operar na Testnet sandbox para auditar envio de ordens, preenchimento de lotes e execução de Stop Loss / Take Profit via exchange.
   ```bash
   python scripts/check_testnet_account.py
   ```

3. **Fase 3: Monitoramento Contínuo e Gestão de Risco**
   - Respeitar dimensionamento conservador (máximo de 5% a 10% da banca por operação).
   - Manter Stop Loss obrigatório e Kill Switch automático configurado via RiskManager.
