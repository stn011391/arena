# Stock Arena — Real 3-Year Walk-Forward Backtest Specification

## Objective
Replace all synthetic/random market curves and demo performance with a real OHLCV walk-forward simulation.

## Backtest window
- Evaluation start: 2023-08-24
- Evaluation end: latest completed Taiwan trading day
- Initial capital: NT$1,000,000 per robot
- Maximum simultaneous holdings: 5
- Long-only in v1
- Odd-lot trading allowed

## No-look-ahead rule
On evaluation day D, a robot may use only market data that was publicly available before its simulated execution on D. It must never use D close/high/low/volume to decide a D morning trade.

Indicators requiring lookback must use pre-start warm-up data. The portfolio evaluation still begins on 2023-08-24; warm-up history is not counted in performance.

## Daily walk-forward loop
For each actual Taiwan trading day from 2023-08-24 forward:
1. Load historical OHLCV available before that day's execution.
2. Each robot independently scores the stock universe using its frozen strategy version.
3. Generate desired Top-5 portfolio and exits.
4. Execute using the configured next-observable execution-price convention.
5. Apply transaction costs and taxes.
6. Mark portfolio to market.
7. Persist all orders, fills, positions, cash, equity and reasons.
8. Advance exactly one trading day and repeat.

## Robots
1. Momentum Hunter — relative strength + MA trend + momentum persistence.
2. Breakout Sniper — consolidation + prior-high breakout + volume confirmation.
3. Reversal Hunter — oversold / mean-reversion setup with trend/risk filters.
4. Quant Master — multi-factor ranking across momentum, trend, liquidity, volatility and relative strength.
5. Risk Master — quality of trend adjusted for volatility/drawdown; prioritizes stable compounding.

## Required metrics
- Final equity
- Total return
- CAGR
- Completed trades
- Winning completed trades
- Losing completed trades
- Win rate = winning completed trades / completed trades
- Average win
- Average loss
- Profit factor
- Maximum drawdown
- Sharpe ratio
- Exposure
- Turnover

## Audit requirements
Every completed trade must retain:
- Robot
- Strategy version
- Symbol/name
- Signal date/time convention
- Buy date/price
- Sell date/price
- Quantity
- Fees/tax
- Net P&L
- Net return
- Holding days
- Entry reasons
- Exit reason

## Data integrity
The UI must display REAL only when the underlying historical OHLCV dataset is present and validated. If real data is missing, the system must show DATA NOT READY and must not manufacture a performance number.

## Data source
Use exchange-authoritative TWSE/TPEx historical data where licensing/access permits. Do not replace missing historical prices with random or synthetic series.
