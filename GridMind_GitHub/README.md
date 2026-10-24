# ⚡ GridMind
Decides every 30 minutes when a grid battery should charge or discharge, from live GB electricity market data
(Elexon), and explains why. Built at OxHack 26 for the HRT challenge: Best use of Data for Predictions & Decision Making.

## Results on 78 unseen days (real Elexon data, Aug - Oct 2026)
- GBP 541k vs GBP 111k for a fixed charge-at-night / sell-at-peak schedule (+389%)
- About GBP 50,650 per MW per year; 33% of the theoretical maximum (perfect hindsight)
- Losing days: 15% vs 44% for the fixed schedule
- Edge over the fixed schedule: +GBP 5,520/day (95% block-bootstrap interval GBP 3,597 to 7,169)
- Multi-horizon ML forecast: 13% lower error than the typical daily shape across the 22-hour plan

## Files
- `streamlit_app.py`: the web app (Live decision, Replay a day, Results, Explainability, How it works)
- `gridmind_core.py`: data download, cleaning, features, models and the battery optimiser (exported from our notebook)
- `gridmind_model.joblib`: trained models and the settings chosen on validation months
- `gridmind_results.json`, `gridmind_replay.csv`, `gridmind_snapshot.json`: backtest results, replay data, last live decision
- `Ox_hacks.ipynb`: the full pipeline with outputs (data, cleaning, EDA, model comparison, backtest, explainability)

## Run locally
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Data: Elexon Insights Solution (BMRS), used under the BMRS Data Licence.
