# 🚀 Trading AI - Sistema Autônomo de Trading Quantitativo com Machine Learning

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791.svg)](https://www.postgresql.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-eb5424.svg)](https://xgboost.readthedocs.io/)
[![Tests Passing](https://img.shields.io/badge/tests-94%20passed-brightgreen.svg)](https://pytest.org)
[![Security Policy](https://img.shields.io/badge/safety-paper%20%26%20testnet%20only-success.svg)](#-regra-fundamental-de-segurança)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sistema quantitativo automatizado desenvolvido em Python para pesquisa estatística, rotulagem financeira sem viés (*look-ahead bias*), validação temporal (*Walk-Forward Cross-Validation*), experimentação de múltiplos modelos de Machine Learning (XGBoost, Random Forest, ExtraTrees, HistGradientBoosting) e execução simulada em tempo real (*Paper Trading* e *Exchange Testnet*).

---

## 🔒 Regra Fundamental de Segurança

> [!CAUTION]
> **PRODUÇÃO E ORDENS REAIS SÃO ESTRITAMENTE PROIBIDAS**
> O sistema foi projetado para operar **exclusivamente** em dois modos de segurança:
> 1. `TRADING_MODE=paper` (Padrão): Simulação interna em memória e banco de dados, sem qualquer envio de ordens externas.
> 2. `TRADING_MODE=testnet`: Execução em ambiente de sandbox / testnet da exchange.
>
> **Garantias Técnicas de Segurança:**
> - Qualquer tentativa de configurar modos de produção (`live`, `real`, `prod`, `production`) dispara um erro crítico imediato de validação (`ValidationError: CRITICAL SECURITY ERROR`), impedindo a inicialização da aplicação.
> - O endpoint `/api/v1/status` audita e expõe a flag imutável `is_live_trading_forbidden: true`.
> - Credenciais de exchanges reais são proibidas e filtradas por sanitizadores de logs (`SensitiveDataFilter`).

---

## 🏛️ Arquitetura do Sistema

O sistema foi concebido sob princípios rigorosos de desacoplamento, rastreabilidade e integridade matemática:

```
┌─────────────────────────┐
│  Market Data Collector  │ ──► Coleta de velas OHLCV (Binance API pública)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ PostgreSQL / Repository │ ──► Persistência e indexação temporal
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Feature Engineering   │ ──► 50+ indicadores técnicos, osciladores e regime
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Machine Learning Models │ ──► XGBoost / Ensembles / Calibração de Probabilidades
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│    Signal Generator     │ ──► Decisão quantitativa (BUY / SELL / HOLD)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   Risk Manager (Gate)   │ ──► Trava de Stop Loss, Drawdown, Daily Loss e Kill Switches
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│    Execution Engine     │
└───────────┬─────────────┘
            ├─────────────────────────────────────────┐
            ▼                                         ▼
┌─────────────────────────┐               ┌─────────────────────────┐
│      Paper Trading      │               │     Testnet Sandbox     │
│  (Simulação Totalmente  │               │   (Binance Testnet API  │
│    Segura em Memória)   │               │   com Filtros de Lote)  │
└─────────────────────────┘               └─────────────────────────┘
```

---

## 📦 Stack Tecnológica

| Componente | Tecnologia | Finalidade |
| :--- | :--- | :--- |
| **Linguagem** | Python 3.12+ | Tipagem estática moderna, alta performance |
| **API Web** | FastAPI + Uvicorn | Endpoints REST assíncronos e documentação OpenAPI interativa |
| **Configuração** | Pydantic Settings v2 | Validação estrita de variáveis de ambiente e segurança |
| **Banco de Dados** | PostgreSQL 16 + SQLAlchemy 2.0 | ORM assíncrono com AsyncPG para séries temporais e auditoria |
| **Data Science & ML** | Pandas, NumPy, Scikit-Learn, XGBoost | Extração de features, treino supervisionado e rotulagem |
| **Cliente HTTP** | HTTPX (Async) | Requisições não-bloqueantes para feeds de mercado |
| **Testes Automatizados**| Pytest, Pytest-AsyncIO, AIOSqlite | Suíte de testes automatizados com cobertura assíncrona |
| **Infraestrutura** | Docker & Docker Compose | Orquestração reproduzível do banco de dados |

---

## 📂 Estrutura do Repositório

```text
trading_ai/
├── app/
│   ├── api/                     # Camada de API REST
│   │   ├── routes/              # Roteadores base e healthcheck
│   │   ├── v1/                  # Endpoints v1 (status, portfolio, trading, risk)
│   │   └── schemas.py           # Modelos Pydantic de entrada e saída
│   ├── backtesting/             # Motor de backtesting event-driven
│   │   ├── engine.py            # Simulação barra a barra com taxas e slippage
│   │   ├── metrics.py           # Cálculo de Sharpe, Sortino, Win Rate, Drawdown
│   │   ├── report.py            # Geração de tear sheets e relatórios
│   │   └── signals.py           # Geração de sinais baseados em probabilidades
│   ├── config/                  # Configurações e logging
│   │   ├── settings.py          # Settings validados com trava de segurança
│   │   └── logging.py           # Sanitização e mascaramento de chaves
│   ├── data/                    # Pipeline de dados de mercado
│   │   ├── collectors/          # Ingestão de dados históricos e em tempo real
│   │   ├── preprocessing/       # Rotulagem (Triple Barrier, Peak) e splitters
│   │   └── storage/             # Modelos SQLAlchemy e repositórios de dados
│   ├── execution/               # Motores de execução desacoplados
│   │   ├── paper/               # Conta virtual e simulação paper trading
│   │   └── testnet/             # Conexão à Testnet com filtros de lote e preço
│   ├── features/                # Engenharia de atributos quantitativos
│   │   ├── indicators.py        # 50+ indicadores (RSI, MACD, ATR, Bollinger, etc.)
│   │   ├── pipeline.py          # Pipeline alinhado sem vazamento de dados
│   │   ├── regime.py            # Detecção de regime macro e volatilidade
│   │   └── repository.py        # Persistência de features pré-computadas
│   ├── models/                  # Inteligência Artificial e Model Registry
│   │   ├── inference/           # Predição em tempo real e batch
│   │   ├── registry/            # Versionamento, auditoria e modelo campeão ativo
│   │   └── training/            # Otimização autônoma, walk-forward e validação
│   ├── portfolio/               # Rastreamento de posições e patrimônio
│   ├── risk/                    # Motor de conformidade e gestão de risco
│   │   ├── manager.py           # Validação de ordens pré-execução
│   │   └── state.py             # Estado de carteira e limites dinâmicos
│   ├── strategy/                # Fachada e convenções de geração de sinais
│   └── main.py                  # Entrypoint da aplicação FastAPI
├── tests/                       # 94 testes automatizados cobrindo todos os módulos
├── scripts/                     # Scripts operacionais e de experimentação
├── reports/                     # Relatórios quantitativos consolidados (.md)
├── data/                        # Diretório para armazenamento local (.gitkeep)
├── models/                      # Diretório de modelos serializados (.gitkeep)
├── notebooks/                   # Análises exploratórias em Jupyter (.gitkeep)
├── .env.example                 # Modelo sanitizado de variáveis de ambiente
├── docker-compose.yml           # Serviço PostgreSQL 16
├── pyproject.toml               # Metadados do projeto e configurações de testes
├── requirements.txt             # Dependências diretas do projeto
└── README.md
```

---

## ⚡ Como Iniciar

### 1. Clonar o Repositório e Criar o Ambiente Virtual

```bash
# Criar ambiente virtual Python 3.12
python -m venv .venv

# Ativar no Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Ou ativar no Linux/macOS:
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Configurar as Variáveis de Ambiente

Copie o template de exemplo para o arquivo local `.env`:

```bash
cp .env.example .env
```

> [!NOTE]
> O arquivo `.env` já vem pré-configurado por padrão com `TRADING_MODE=paper`, parâmetros conservadores de risco e credenciais locais para o banco de dados Docker. Ele é estritamente ignorado pelo `.gitignore`.

### 3. Subir o PostgreSQL via Docker

```bash
docker compose up -d postgres
```

### 4. Inicializar o Esquema do Banco de Dados

Cria todas as tabelas relacionais do sistema (`market_candles`, `features`, `model_versions`, `predictions`, `signals`, `orders`, `fills`, `positions`, `portfolio_snapshots`, `risk_events`, `trades`):

```bash
python scripts/init_db.py
```

### 5. Executar a Suíte de Testes Automatizados

```bash
pytest -v
```

Todos os 94 testes unitários e de integração devem passar com sucesso.

### 6. Iniciar a API Web FastAPI

```bash
uvicorn app.main:app --reload --port 8000
```

Acesse a documentação interativa OpenAPI no navegador: **`http://localhost:8000/docs`**

---

## 🛠️ Guia de Scripts e Fluxo Operacional

O pipeline quantitativo é executado através de comandos CLI modulares:

### 1. Coleta de Dados Históricos
Baixa velas OHLCV diretamente da API pública da exchange e armazena no PostgreSQL:

```bash
python scripts/download_candles.py --symbols BTC/USDT ETH/USDT SOL/USDT --timeframes 1h 15m --days 60
```

### 2. Cálculo e Persistência de Features
Gera mais de 50 atributos matemáticos e quantitativos (RSI, MACD, Bandas de Bollinger, ATR, distâncias de médias, Choppiness Index, inclinações de regressão):

```bash
python scripts/build_features.py --symbols BTC/USDT --timeframes 1h
```

### 3. Preparação do Dataset e Rotulagem
Constrói os alvos supervisionados (*Triple Barrier Method*, retorno futuro e picos) com garantia de ausência de viés temporal (*look-ahead bias*):

```bash
python scripts/build_dataset.py --symbol BTC/USDT --timeframe 1h --horizon 16 --target-return 0.008 --label-method peak
```

### 4. Treinamento do Modelo e Registro no Model Registry
Treina o modelo XGBoost e registra a versão no banco de dados com auditoria completa de métricas:

```bash
python scripts/train_model.py --symbol BTC/USDT --timeframe 1h --horizon 16 --target-return 0.008 --set-active
```

### 5. Validação Temporal Walk-Forward
Avalia a estabilidade do modelo em múltiplos cortes temporais sequenciais com embargo temporal:

```bash
python scripts/run_walk_forward.py --symbol BTC/USDT --timeframe 1h --splits 4 --horizon 16
```

### 6. Backtest Event-Driven
Simula a execução histórica de ordens considerando slippage de 5 bps e corretagem real:

```bash
python scripts/run_backtest.py --symbol BTC/USDT --timeframe 1h --initial-capital 10000 --buy-threshold 0.55 --stop-loss 0.015 --take-profit 0.035
```

### 7. Paper Trading em Tempo Real
Executa o motor de simulação em tempo real, computando features a cada nova barra e operando na carteira virtual:

```bash
python scripts/run_paper_trading.py --symbol BTC/USDT --timeframe 1h --initial-cash 1000 --steps 50
```

### 8. Verificação de Conectividade com a Testnet
Inspeciona saldo virtual e ordens ativas no ambiente sandbox da Binance Testnet:

```bash
python scripts/check_testnet_account.py
```

### 9. Otimização Autônoma & Comparação de Arquiteturas
Pesquisa exaustiva de hiperparâmetros e algoritmos (XGBoost, Random Forest, ExtraTrees, HistGradientBoosting):

```bash
python scripts/run_autonomous_research_loop.py --symbol BTC/USDT --timeframe 1h --days 60
```

### 10. Simulação Contínua Multi-Ativo
Executa backtest institucional simultâneo em carteira ponderada com BTC, ETH e SOL:

```bash
python scripts/run_multi_asset_mass_backtest.py
```

---

## 🛡️ Gestão de Risco & Kill Switches

O módulo `RiskManager` atua como um portão independente obrigatório que intercepta todas as ordens antes do envio ao motor de execução:

- **Tamanho Máximo por Posição (`MAX_POSITION_SIZE_PCT`)**: Limite percentual do patrimônio alocado em uma única operação.
- **Risco Máximo por Operação (`MAX_RISK_PER_TRADE_PCT`)**: Determina o lote a partir da distância até o Stop Loss (ex: máximo de 0.5% ou 1.0% de perda da banca total).
- **Trava de Perda Diária (*Daily Loss Kill-Switch*)**: Pausa imediata de novas entradas caso o prejuízo no dia atinja o limite configurado (ex: 2.0% a 4.0%).
- **Trava de Rebaixamento Máximo (*Drawdown Kill-Switch*)**: Desativação automática se o rebaixamento acumulado ultrapassar o teto estipulado (ex: 10.0% a 12.0%).
- **Trava de Perdas Consecutivas**: Bloqueio de novas operações caso ocorram N prejuízos seguidos (padrão: 4 trades consecutivos).

---

## 🌐 Endpoints da API REST

A API do sistema provê monitoramento, telemetria de carteira e inspeção de risco:

| Método | Endpoint | Descrição |
| :--- | :--- | :--- |
| `GET` | `/health` | Checagem de integridade e conectividade do sistema e banco |
| `GET` | `/api/v1/status` | Modo de operação, versão do modelo ativo e status de segurança |
| `GET` | `/api/v1/portfolio/state` | Patrimônio atual, saldo livre, posições abertas e drawdown |
| `GET` | `/api/v1/portfolio/positions`| Listagem detalhada de posições abertas e PnL não-realizado |
| `GET` | `/api/v1/portfolio/trades` | Histórico consolidado de operações finalizadas |
| `GET` | `/api/v1/trading/signals` | Últimos sinais de decisão emitidos pelo modelo |
| `POST`| `/api/v1/risk/check` | Simulação prévia de validação de ordem pelo Risk Manager |
| `GET` | `/api/v1/risk/events` | Histórico de rejeições e acionamentos de travas de risco |

---

## 🧪 Qualidade de Código & Testes

A suíte conta com **94 testes automatizados** cobrindo todos os módulos do pipeline:

```bash
pytest -v --durations=10
```

- Validação estrita de segurança e bloqueio de produção (`tests/test_config.py`)
- Ingestão, conectividade e sanitização de dados (`tests/test_data_collector.py`, `tests/test_candle_repository.py`)
- Pipeline de features e regimes matemáticos (`tests/test_features.py`)
- Rotulagem e particionamento temporal (`tests/test_dataset_and_labels.py`)
- Treinamento e registro de modelos (`tests/test_model_training.py`, `tests/test_multi_model_and_autonomous_optimizer.py`)
- Validação Walk-Forward (`tests/test_walk_forward.py`)
- Motor de Backtesting e métricas financeiras (`tests/test_backtest.py`)
- Gestão de risco e Kill Switches (`tests/test_risk_manager.py`)
- Execução em Paper Trading e Testnet (`tests/test_paper_trading.py`, `tests/test_testnet_execution.py`)
- Endpoints da API FastAPI (`tests/test_health_api.py`, `tests/test_api_v1.py`)

---

## ⚖️ Aviso Legal (*Disclaimer*)

Este software foi desenvolvido para fins estritamente **acadêmicos, educacionais e de pesquisa quantitativa**. Não constitui aconselhamento financeiro, recomendação de investimento ou incentivo a operações em mercados reais. Rentabilidade passada não é garantia de rentabilidade futura.
