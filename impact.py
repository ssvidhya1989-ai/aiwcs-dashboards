"""
AI impact: what the advisory AI changes, measured on the costing data.
- warning_history: replays the Railway-held test year day by day. Each unit gets its AI estimate at receipt (model not trained
  on it); the AI warning fires on the first day the cost to date passes that estimate. Without AI the cost is known 10 days
  after the month-end of dispatch.
- ml_only: units flagged by the IsolationForest that no rule check flagged.
- reasons: plain-language reasons behind a unit's warning (shop vs AI shop estimate, largest material item, condition at receipt).
"""
import numpy as np
import pandas as pd
import aiwcs_logic as L
import wip as W

REPORT_LAG_DAYS = 10          # cost known 10 days after month-end today (problem statement)
AHEAD = None                  # None: warn only when the estimate is passed (0.25 = also 'spending ahead of time')


def usual_days(d) -> pd.Series:
    j = d["poh_jobs"]
    j = j[j.split == "train"]
    return ((j.date_dispatched - j.date_received).dt.days + 1).groupby(j.rs_type_code).median()


def first_warning(cum: pd.Series, est: float, usual: float):
    """cum: cumulative cost indexed by date from receipt. Returns the first date the warning rule fires, or None."""
    used = cum.values / est
    day = np.arange(1, len(cum) + 1)
    hit = (used > 1) if AHEAD is None else (used > 1) | (used - day / usual > AHEAD)
    return cum.index[hit.argmax()] if hit.any() else None


def warning_history(d, led: pd.DataFrame, model) -> pd.DataFrame:
    te = model["test"][["job_no", "predicted", "actual"]]
    j = d["poh_jobs"].set_index("job_no")
    us = usual_days(d)
    tot = led.assign(c=led[W.ELEMENTS].sum(axis=1)).groupby(["job_no", "date"]).c.sum()
    rows = []
    for r in te.itertuples():
        u = j.loc[r.job_no]
        days = pd.date_range(u.date_received, u.date_dispatched, freq="D")
        cum = tot.loc[r.job_no].reindex(days).fillna(0).cumsum()
        w = first_warning(cum, r.predicted, us.get(u.rs_type_code, us.median()))
        known = u.date_dispatched + pd.offsets.MonthEnd(0) + pd.Timedelta(days=REPORT_LAG_DAYS)
        rows.append({"job_no": r.job_no, "rs_type_code": u.rs_type_code, "estimate": r.predicted, "actual": r.actual,
                     "received": u.date_received, "dispatched": u.date_dispatched, "warned": w, "known_without_ai": known})
    h = pd.DataFrame(rows)
    h["overrun"] = h.actual > h.estimate * 1.05
    h["caught"] = h.warned.notna() & (h.warned < h.dispatched)
    h["lead_days"] = (h.dispatched - h.warned).dt.days
    h["earlier_days"] = (h.known_without_ai - h.warned).dt.days
    h["warn_day"] = (h.warned - h.received).dt.days + 1
    return h


def summary(h: pd.DataFrame) -> dict:
    o = h[h.overrun]
    c = o[o.caught]
    return {"units": len(h), "overruns": len(o), "caught": len(c), "at_risk": float((c.actual - c.estimate).sum()),
            "lead_days": float(c.lead_days.median()) if len(c) else 0, "earlier_days": float(c.earlier_days.median()) if len(c) else 0,
            "false_alarms": int((h.caught & ~h.overrun).sum())}


def ml_only(d, rule_flags: pd.DataFrame) -> pd.DataFrame:
    m = L.ml_flags(d)
    m = m[m.flag.astype(str).isin(["True", "-1", "1", "Flag", "Yes"])] if m.flag.dtype != bool else m[m.flag]
    return m[~m.job_no.isin(set(rule_flags.job_no))]


COND = {"corrosion_trough_floor": ("Corrosion, trough floor", 1), "corrosion_sole_bar": ("Corrosion, sole bar", 1),
        "corrosion_side_wall": ("Corrosion, side wall", 1), "corrosion_roof": ("Corrosion, roof", 1),
        "wheel_dia_mm": ("Wheel diameter (mm)", -1), "flange_thickness_mm": ("Flange thickness (mm)", -1),
        "brake_disc_wear_mm": ("Brake disc wear (mm)", 1), "missing_damaged_fittings": ("Missing / damaged fittings", 1)}


def reasons(d, led: pd.DataFrame, model, job_no: str, as_on) -> list[str]:
    """Up to three plain-language reasons behind a unit's warning."""
    as_on = pd.Timestamp(as_on)
    j = d["poh_jobs"].set_index("job_no").loc[job_no]
    out = []
    # 1. shop furthest beyond its AI shop estimate
    f = L.feature_frame(d)
    f = f[f.job_no == job_no].reset_index(drop=True)
    p = L.price_prediction(d, L.predict_quantities(model, L.design_matrix(f, model["types"])).iloc[0], j.fy)["by_shop"]
    act = led[(led.job_no == job_no) & (led.date <= as_on)].groupby("shop_no")[W.ELEMENTS].sum().sum(axis=1)
    p["actual"] = p.shop_no.map(act).fillna(0)
    p["over"] = p.actual - p.shop_cost
    s = p.sort_values("over", ascending=False).iloc[0]
    if s.over > 0:
        out.append(f"<b>{s.shop}</b>: ₹ {s.actual/1e5:,.2f} lakh booked so far against an AI estimate of "
                   f"₹ {s.shop_cost/1e5:,.2f} lakh for the whole overhaul (<b>+{s.over/max(s.shop_cost, 1):.0%}</b>)")
    # 2. largest material item, and whether the standard BOM covers it
    mi = d["material_issues"]
    mi = mi[(mi.job_no == job_no) & (mi.date <= as_on)]
    if len(mi):
        g = mi.groupby("pl_no").agg(value=("value", "sum"), qty=("quantity", "sum")).sort_values("value", ascending=False)
        pl, top = g.index[0], g.iloc[0]
        desc = d["item_master"].set_index("pl_no").description.get(pl, str(pl))
        bom = d["standard_bom"]
        std = bom[(bom.rs_type_code == j.rs_type_code) & (bom.pl_no == pl)].std_qty_per_poh
        note = "not in the standard BOM" if std.empty or not std.iloc[0] else f"{top.qty / std.iloc[0]:.1f}× the standard quantity"
        out.append(f"Largest item drawn: <b>{desc}</b> ₹ {top.value/1e5:,.2f} lakh ({note})")
    # 3. condition at receipt worse than most units of this type
    cf = d["condition_findings"].merge(d["poh_jobs"][["job_no", "rs_type_code"]], on="job_no")
    peers = cf[cf.rs_type_code == j.rs_type_code]
    me = cf[cf.job_no == job_no].iloc[0]
    worst = []
    for col, (label, sign) in COND.items():
        pct = ((peers[col] * sign) < me[col] * sign).mean()
        worst.append((pct, label, me[col]))
    worst = [w for w in sorted(worst, reverse=True) if w[0] >= 0.8][:2]
    if worst:
        out.append("Condition at receipt: " + " · ".join(f"<b>{lab} {val:g}</b> (worse than {pct:.0%} of this type)" for pct, lab, val in worst))
    return out
