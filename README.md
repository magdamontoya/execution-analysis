Measuring what trading execution actually costs, tick by tick.

A trader's P&L report shows the result. It does not show how much of that result was lost to spread, slippage and bad timing, because measuring that requires reconstructing the order book at the millisecond each trade was filled.

This toolkit does exactly that. It reads a raw trade history from any broker, crosses every fill against tick data, and quantifies the gap between the price you got and the price that was available.

What it found:

Run against a real 396-trade account over 58 million ticks:

Metric	Result
Spread paid	20.6% of everything the account had won
Avoidable share	Just over half, by switching from market to limit orders
Worst execution hour	Spread 4.9x its normal level
Cost vs. result, by instrument	One instrument cost 3.7x in spread what it returned

The headline is the method, not the number: execution cost was roughly half of the account's total loss, and none of it was visible in the broker's report.

How it works
lector.py        Parses trade history — MT4, MT5, CSV, Excel, any broker layout
ejecucion.py     Crosses each fill against tick data; computes spread paid,
                 slippage vs. available price, and limit-vs-market counterfactual
diagnostico.py   Breakdown by duration, size, hour and instrument;
                 Monte Carlo resampling of the real return distribution
informe_pdf.py   Six-page PDF report with charts

Input is a trade history plus tick data. Output is a report.

Method notes

The analysis makes three choices that matter:

Ticks, not candles. Exit logic resolved on 5-minute candles inflates results badly — in testing, a strategy showing 77% hit rate on candles dropped to 53.6% when the same trades were resolved tick by tick, because within a single candle you cannot know whether the target or the stop was touched first.

Resampling, not assumptions. Ruin probability comes from resampling the account's own return distribution 4,000 times, not from assuming a win rate.

A random baseline. Any rule tested is compared against equivalent random entries, with periods split in half. Of several thousand configurations tested across six instruments, almost none survived both the random baseline and out-of-sample validation — which is itself the useful finding.

Stack

Python · pandas · numpy · pyarrow · MetaTrader5 · matplotlib · reportlab Dashboard and web layers: Streamlit, FastAPI, Plotly

Why it exists

I built this to understand my own losses. The assumption was that the analysis was wrong. The data said the analysis was roughly a coin flip either way, and that the money was going somewhere else entirely: into how trades were entered and how long they were held.

That is a measurable problem, and measuring it is what this repository does.
