"""
Work in progress (WIP): cost built up on units still on the shop floor, as on any date.
Valued from dated labour bookings (labour cost + labour on-cost) and issue notes (material + stores on-cost; returns are
negative). For a dispatched unit the four elements add up to its cost in target_coach_cost.
"""
import numpy as np
import pandas as pd
import aiwcs_logic as L

ELEMENTS = ["labour_cost", "labour_oncost", "material_value", "stores_oncost"]


def ledger(d) -> pd.DataFrame:
    """One row per job, shop and day with the four cost elements in rupees."""
    lb = d["labour_bookings"][["job_no", "shop_no", "date", "labour_cost", "labour_oncost"]]
    mi = d["material_issues"][d["material_issues"].job_no != ""][["job_no", "shop_no", "date", "value", "stores_oncost"]]
    mi = mi.rename(columns={"value": "material_value"})
    g = pd.concat([lb, mi], ignore_index=True).fillna(0)
    return g.groupby(["job_no", "shop_no", "date"], as_index=False)[ELEMENTS].sum()


def on_floor(d, as_on) -> pd.DataFrame:
    """Units received on or before the date and not dispatched by it."""
    j = d["poh_jobs"]
    as_on = pd.Timestamp(as_on)
    return j[(j.date_received <= as_on) & (j.date_dispatched > as_on)]


def wip_as_on(d, led: pd.DataFrame, as_on) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(units, shops): cost to date per unit on the floor, and the same by shop."""
    as_on = pd.Timestamp(as_on)
    jobs = on_floor(d, as_on)
    g = led[led.job_no.isin(jobs.job_no) & (led.date <= as_on)]
    units = g.groupby("job_no")[ELEMENTS].sum().reindex(jobs.job_no).fillna(0).reset_index()
    units["cost_to_date"] = units[ELEMENTS].sum(axis=1)
    units = units.merge(jobs[["job_no", "coach_no", "rs_type_code", "date_received", "date_dispatched", "fy"]], on="job_no")
    units["days_on_floor"] = (as_on - units.date_received).dt.days + 1
    shops = g.groupby("shop_no")[ELEMENTS].sum().reset_index()
    shops["shop"] = shops.shop_no.map(L.SHOPS)
    return units, shops


def month_end_trend(d, led: pd.DataFrame) -> pd.DataFrame:
    """WIP value and units on the floor at every month end in the data."""
    j = d["poh_jobs"]
    ends = pd.date_range(j.date_received.min(), j.date_dispatched.max(), freq="ME")
    tot = led.assign(cost=led[ELEMENTS].sum(axis=1)).groupby(["job_no", "date"]).cost.sum().reset_index()
    rows = []
    for e in ends:
        jobs = on_floor(d, e).job_no
        rows.append({"month_end": e, "units": len(jobs), "wip": tot[tot.job_no.isin(jobs) & (tot.date <= e)].cost.sum()})
    return pd.DataFrame(rows)


def daily_curve(d, led: pd.DataFrame, job_no: str) -> pd.DataFrame:
    """Cumulative cost by element for one unit, every calendar day from receipt to dispatch."""
    j = d["poh_jobs"].set_index("job_no").loc[job_no]
    days = pd.date_range(j.date_received, j.date_dispatched, freq="D")
    g = led[led.job_no == job_no].groupby("date")[ELEMENTS].sum().reindex(days).fillna(0).cumsum()
    g.index.name = "date"
    g = g.reset_index()
    g["day"] = np.arange(1, len(g) + 1)
    g["total"] = g[ELEMENTS].sum(axis=1)
    return g


def ai_estimates(d, model, job_nos) -> pd.Series:
    """AI estimate at receipt (same model as the Cost prediction page), priced at the unit's FY rates."""
    f = L.feature_frame(d)
    f = f[f.job_no.isin(job_nos)].reset_index(drop=True)
    if f.empty:
        return pd.Series(dtype=float)
    q = L.predict_quantities(model, L.design_matrix(f, model["types"]))
    return pd.Series([L.price_prediction(d, q.iloc[i], fy)["total"] for i, fy in enumerate(f.fy)], index=f.job_no.values)
