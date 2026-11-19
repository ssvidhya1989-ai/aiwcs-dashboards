# AI Assisted Workshop Costing System: dashboards

Two versions of the same costing dashboard, on DUMMY data:

| App | Main file |
|---|---|
| New: AI alert of the day, AI impact scoreboard, Without AI vs With AI replay, hover tiles | `app_new.py` |
| Original | `app_original.py` |

Deploy each on Streamlit Community Cloud (share.streamlit.io) with this repository and the main file above (Python 3.11).
Optional: add `ANTHROPIC_API_KEY` under the app's Secrets to let the costing assistant use Claude; without it the built-in reader answers.

Run locally: `pip install -r requirements.txt` then `streamlit run app_new.py`.

Advisory AI only: no AI output posts an accounting entry. All figures are computed from the dummy data in `data/`.
