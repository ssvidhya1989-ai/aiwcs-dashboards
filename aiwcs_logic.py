"""
AI-WCS dashboard — data and model logic, aligned to BRD v0.4
(Rolling stock POH, shop-wise costing, on-cost method).

Cost per unit POH = Direct material + Direct labour + Labour on-cost + Stores on-cost   (BRD §6)
Variance           = material used vs required; labour time consumed vs allowed time   (BRD §6)

Open-source only: pandas, numpy, scikit-learn. No UI code here, so it can be reused from Jupyter.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import cross_val_predict

SHOPS = {1: "Inspection shop", 2: "Lifting shop", 3: "Bogie shop", 4: "Body shop", 5: "Air Brake shop",
         6: "Electrical & Furnishing shop", 7: "Paint shop", 8: "Testing shop"}
ELEMENTS = ["material_value", "labour_cost", "labour_oncost", "stores_oncost"]
ELEMENT_LABEL = {"material_value": "Direct material", "labour_cost": "Direct labour",
                 "labour_oncost": "Labour on-cost", "stores_oncost": "Stores on-cost"}
MAPE_TARGET = 0.05      # AI-01 (BRD v0.4)
MAPE_ALERT = 0.15       # AI-12 drift alert
DATE_COLS = {"poh_jobs": ["date_received", "date_dispatched"], "attendance": ["date"], "labour_bookings": ["date"],
             "idle_time": ["date"], "material_issues": ["date"], "direct_expenses": ["date"], "driver_consumption": ["date"]}
COND_MAP = {"Good": 0, "Fair": 1, "Poor": 2}


# ------------------------------------------------------------------ loading
def load_data(data_dir: str = "data") -> dict[str, pd.DataFrame]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data folder '{data_dir}' not found. Unzip the dummy CSVs into it.")
    d = {}
    for f in os.listdir(data_dir):
        if f.endswith(".csv"):
            name = f[:-4]
            df = pd.read_csv(os.path.join(data_dir, f), low_memory=False)
            for c in DATE_COLS.get(name, []):
                df[c] = pd.to_datetime(df[c])
            d[name] = df
    required = ["poh_jobs", "target_coach_cost", "target_shop_cost", "labour_bookings", "attendance", "oncost_rates"]
    missing = [r for r in required if r not in d]
    if missing:
        raise FileNotFoundError(f"Missing CSVs: {missing}. Use the v0.4 dummy data (AI-WCS_Dummy_Training_Data_v2_CSV.zip).")
    d["material_issues"]["job_no"] = d["material_issues"]["job_no"].fillna("")
    return d


def fy_of(ts: pd.Series) -> pd.Series:
    y = ts.dt.year - (ts.dt.month < 4).astype(int)
    return y.astype(str) + "-" + ((y + 1) % 100).astype(str).str.zfill(2)


# ------------------------------------------------------------------ overview
def coach_table(d) -> pd.DataFrame:
    t = d["target_coach_cost"].merge(d["poh_jobs"][["job_no", "date_received", "date_dispatched"]], on="job_no")
    t = t.merge(d["coach_master"][["coach_no", "rs_type"]], on="coach_no", how="left")
    t["month_dispatched"] = t.date_dispatched.dt.to_period("M").dt.to_timestamp()
    t["oncost_total"] = t.labour_oncost + t.stores_oncost
    return t


def kpis(t: pd.DataFrame) -> dict:
    tot = t.total_cost_per_coach.sum()
    return {"units": len(t), "avg_cost": t.total_cost_per_coach.mean() if len(t) else 0, "total_cost": tot,
            "oncost_share": (t.labour_oncost.sum() + t.stores_oncost.sum()) / tot if tot else 0,
            "avg_hours": t.man_hours.mean() if len(t) else 0}


def element_by_fy(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby("fy")[ELEMENTS].mean().reset_index()
    return g.melt(id_vars="fy", var_name="element", value_name="avg_per_unit").assign(
        element=lambda x: x.element.map(ELEMENT_LABEL))


def shop_cost_table(d, jobs: pd.Series | None = None) -> pd.DataFrame:
    s = d["target_shop_cost"].merge(d["poh_jobs"][["job_no", "rs_type_code", "fy"]], on="job_no")
    return s if jobs is None else s[s.job_no.isin(jobs)]


# ------------------------------------------------------------------ variance (BRD §6)
def shop_variance(d, job_no: str) -> pd.DataFrame:
    """Time consumed vs allowed time, and material used vs material required, by shop."""
    s = d["target_shop_cost"][d["target_shop_cost"].job_no == job_no].copy()
    j = d["poh_jobs"].set_index("job_no").loc[job_no]
    card = d["standard_cost_card"]
    card = card[(card.rs_type_code == j.rs_type_code) & (card.fy == j.fy)][["shop_no", "allowed_hours", "std_material_value"]]
    s = s.merge(card, on="shop_no", how="left")
    s["time_variance_hours"] = s.man_hours - s.allowed_hours
    s["time_efficiency"] = s.allowed_hours / s.man_hours.where(s.man_hours > 0)
    s["material_variance"] = s.material_value - s.std_material_value
    return s


# ------------------------------------------------------------------ on-cost (BRD §6) and FR-07 parallel run
def oncost_summary(d, fys) -> pd.DataFrame:
    s = shop_cost_table(d)
    s = s[s.fy.isin(fys)]
    g = s.groupby(["fy", "shop_no", "shop"])[["labour_cost", "labour_oncost", "material_value", "stores_oncost"]].sum().reset_index()
    return g.merge(d["oncost_rates"], on=["fy", "shop_no", "shop"], how="left")


def parallel_run(t: pd.DataFrame) -> pd.DataFrame:
    """FR-07: on-cost method (official) vs driver-based overhead (comparison only)."""
    x = t[["job_no", "rs_type_code", "fy", "labour_oncost", "stores_oncost", "parallel_driver_based_overhead"]].copy()
    x["oncost_method"] = x.labour_oncost + x.stores_oncost
    x["difference"] = x.parallel_driver_based_overhead - x.oncost_method
    return x


def overhead_check(d) -> pd.DataFrame:
    a = d["overhead_allocation"].groupby(["month", "pool"]).overhead.sum().rename("allocated").reset_index()
    p = d["overhead_pools"].merge(a, on=["month", "pool"], how="left")
    p["difference"] = (p.allocated - p.pool_cost).round(2)
    return p


# ------------------------------------------------------------------ labour reconciliation (FC-03)
def labour_reconciliation(d, regular_only: bool = True) -> pd.DataFrame:
    att = d["attendance"].merge(d["staff_master"][["staff_id", "shop"]], on="staff_id")
    bk = d["labour_bookings"]
    if regular_only:
        bk = bk[bk["shift"] != "Overtime"]
    b = bk.groupby(["staff_id", "date"]).hours.sum().rename("booked")
    i = d["idle_time"].groupby(["staff_id", "date"]).idle_hours.sum().rename("idle")
    x = att.set_index(["staff_id", "date"]).join(b).join(i).fillna({"booked": 0, "idle": 0}).reset_index()
    x["leave"] = x.paid_hours - x.attended_hours
    x["unaccounted"] = (x.attended_hours - x.booked - x.idle).clip(lower=0).round(2)
    x["over_booked"] = (x.booked - x.attended_hours).clip(lower=0).round(2)
    x["month"] = x.date.dt.to_period("M").dt.to_timestamp()
    return x


def recon_by_shop_month(x: pd.DataFrame) -> pd.DataFrame:
    g = x.groupby(["shop", "month"])[["paid_hours", "booked", "idle", "leave", "unaccounted", "over_booked"]].sum().reset_index()
    g["unaccounted_pct"] = (g.unaccounted / g.paid_hours.where(g.paid_hours > 0)).fillna(0)
    return g


# ------------------------------------------------------------------ material
def material_summary(d) -> pd.DataFrame:
    m = d["material_issues"][d["material_issues"].job_no != ""]
    g = m.groupby("pl_no").agg(value=("value", "sum"), issues=("issue_note_no", "count")).reset_index()
    return g.merge(d["item_master"][["pl_no", "description", "shop_no", "category"]], on="pl_no").sort_values("value", ascending=False)


# ------------------------------------------------------------------ anomaly rules (AI-04, AI-06)
def rule_flags(d, hours_factor: float = 1.35, repeat_threshold: int = 2) -> pd.DataFrame:
    out = []
    jobs = d["poh_jobs"].set_index("job_no")
    x = labour_reconciliation(d, regular_only=False)
    for r in x[x.over_booked > 0].itertuples():
        out.append(["R1 Hours booked beyond attendance", "", f"{r.staff_id} {r.date:%d-%m-%Y}",
                    f"Booked {r.booked:.1f} h vs attended {r.attended_hours:.1f} h ({r.status})", "High"])
    e = d["direct_expenses"].merge(jobs[["date_dispatched"]], left_on="job_no", right_index=True)
    for r in e[e.date > e.date_dispatched].itertuples():
        out.append(["R2 Expense after dispatch", r.job_no, r.expense_id,
                    f"Rs {r.amount:,.0f} dated {r.date:%d-%m-%Y}; dispatched {r.date_dispatched:%d-%m-%Y}", "High"])
    m = d["material_issues"]
    mi = m[(m.job_no != "") & (m.txn_type == "ISSUE")].merge(jobs[["date_dispatched"]], left_on="job_no", right_index=True)
    for r in mi[mi.date > mi.date_dispatched].itertuples():
        out.append(["R3 Issue to dispatched unit", r.job_no, r.issue_note_no, f"{r.pl_no} x {r.quantity} on {r.date:%d-%m-%Y}", "High"])
    rep = mi.groupby(["job_no", "pl_no"]).size()
    for (jn, pl), n in rep[rep > repeat_threshold].items():
        out.append(["R4 Repeated drawal of same item", jn, pl, f"{n} issue notes for {pl}", "Medium"])
    for r in m[m.job_no == ""].itertuples():
        out.append(["R5 Issue without job number / work order", "", r.issue_note_no, f"{r.pl_no} x {r.quantity}", "Low"])
    s = d["target_shop_cost"].merge(jobs[["rs_type_code"]], left_on="job_no", right_index=True)
    med = s.groupby(["rs_type_code", "shop_no"]).man_hours.transform("median")
    for r in s[s.man_hours > med * hours_factor].itertuples():
        out.append(["R6 Excess hours vs comparable units", r.job_no, f"shop {r.shop_no}", f"{r.man_hours:.0f} h in {r.shop}", "Medium"])
    for r in x[x.unaccounted > 0].itertuples():
        out.append(["R7 Unaccounted labour hours", "", f"{r.staff_id} {r.date:%d-%m-%Y}",
                    f"{r.unaccounted:.1f} h attended but neither booked nor idle", "Low"])
    f = pd.DataFrame(out, columns=["rule", "job_no", "ref", "detail", "severity"])
    f.insert(0, "flag_id", [f"F{i+1:05d}" for i in range(len(f))])
    return f


def ml_flags(d, contamination: float = 0.02) -> pd.DataFrame:
    s = d["target_shop_cost"].merge(d["poh_jobs"][["job_no", "rs_type_code"]], on="job_no")
    X = s.pivot_table(index="job_no", columns="shop_no", values=["man_hours", "material_value"]).fillna(0)
    X.columns = [f"{a}_{SHOPS[b]}" for a, b in X.columns]
    typ = s.drop_duplicates("job_no").set_index("job_no").rs_type_code.reindex(X.index)
    Xn = X / X.groupby(typ).transform("median").replace(0, 1)
    iso = IsolationForest(contamination=contamination, random_state=0).fit(Xn)
    res = pd.DataFrame({"job_no": X.index, "rs_type_code": typ.values, "anomaly_score": -iso.score_samples(Xn),
                        "flag": iso.predict(Xn) == -1, "main_driver": Xn.sub(1).abs().idxmax(axis=1).values})
    return res.sort_values("anomaly_score", ascending=False)


def evaluate_rules(flags: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Recall against the planted-anomaly answer key (testing only)."""
    rows = []
    refs = set(flags.ref.str.replace(r" (\d\d)-(\d\d)-(\d{4})$", r" \3-\2-\1", regex=True)) | set(flags.ref)
    for t, g in key.groupby("anomaly_type"):
        if t.startswith("Unaccounted") or t.startswith("Missing job"):
            hit = g.record_ref.isin(refs)
        elif t.startswith("Hours booked while on leave"):
            hit = pd.Series([(flags.rule.str.startswith("R1")).sum() >= len(g)] * len(g))
        else:
            hit = g.job_no.isin(flags.job_no)
        rows.append([t, len(g), int(hit.sum()), float(hit.mean())])
    return pd.DataFrame(rows, columns=["anomaly_type", "planted", "caught", "recall"])


# ------------------------------------------------------------------ cost prediction (AI-01, AI-02, AI-12)
FEATURES_NUM = ["age_years", "km_since_last_poh", "months_since_last_poh", "previous_poh_count", "ioh_count_since_last_poh",
                "corrosion_trough_floor", "corrosion_sole_bar", "corrosion_side_wall", "corrosion_roof",
                "wheel_dia_mm", "flange_thickness_mm", "brake_disc_wear_mm", "missing_damaged_fittings",
                "air_spring", "damper", "cbc_coupler", "coastal", "accident"]


def feature_frame(d) -> pd.DataFrame:
    j = d["poh_jobs"].merge(d["condition_findings"], on="job_no").merge(
        d["coach_master"][["coach_no", "year_built", "coastal_region"]], on="coach_no")
    f = pd.DataFrame({"job_no": j.job_no, "rs_type_code": j.rs_type_code, "fy": j.fy, "split": j.split,
                      "date_dispatched": j.date_dispatched})
    f["age_years"] = j.date_received.dt.year - j.year_built
    for c in ["km_since_last_poh", "months_since_last_poh", "previous_poh_count", "ioh_count_since_last_poh",
              "corrosion_trough_floor", "corrosion_sole_bar", "corrosion_side_wall", "corrosion_roof",
              "wheel_dia_mm", "flange_thickness_mm", "brake_disc_wear_mm", "missing_damaged_fittings"]:
        f[c] = j[c]
    f["air_spring"] = j.air_spring_condition.map(COND_MAP)
    f["damper"] = j.damper_condition.map(COND_MAP)
    f["cbc_coupler"] = j.cbc_coupler_condition.map(COND_MAP)
    f["coastal"] = (j.coastal_region == "Y").astype(int)
    f["accident"] = (j.accident_history == "Y").astype(int)
    return f


def design_matrix(f: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    X = f[FEATURES_NUM].astype(float).copy()
    for v in types:
        X[f"type_{v}"] = (f.rs_type_code == v).astype(int)
    return X


def base_price_material_by_shop(d) -> pd.DataFrame:
    """Material per job and shop re-priced at the first year's Book Average Rate (removes inflation)."""
    m = d["material_issues"][d["material_issues"].job_no != ""].copy()
    bar = d["book_average_rates"]
    base = bar[bar.fy == bar.fy.min()].set_index("pl_no").book_average_rate
    m["base_value"] = m.quantity.astype(float) * m.pl_no.map(base) * np.where(m.txn_type == "RETURN", -1, 1)
    return m.pivot_table(index="job_no", columns="shop_no", values="base_value", aggfunc="sum").fillna(0)


def price_factors(d, fy: str) -> dict:
    bar = d["book_average_rates"]
    if fy not in set(bar.fy):
        fy = bar.fy.max()
    base_fy = bar.fy.min()
    mat_index = (bar[bar.fy == fy].set_index("pl_no").book_average_rate / bar[bar.fy == base_fy].set_index("pl_no").book_average_rate).mean()
    t = d["target_coach_cost"][d["target_coach_cost"].fy == fy]
    oc = d["oncost_rates"][d["oncost_rates"].fy == fy].set_index("shop_no")
    return {"labour_rate": t.labour_cost.sum() / t.man_hours.sum(), "material_index": float(mat_index),
            "loc_pct": oc.labour_oncost_pct.to_dict(), "soc_pct": oc.stores_oncost_pct.to_dict()}


def price_prediction(d, q: pd.Series, fy: str) -> dict:
    """Turn predicted quantities (hours and base-price material per shop) into rupees at FY rates."""
    pf = price_factors(d, fy)
    rows = []
    for k, name in SHOPS.items():
        hrs = max(0.0, float(q[f"hours_{k}"]))
        mat = max(0.0, float(q[f"mat_{k}"])) * pf["material_index"]
        lab = hrs * pf["labour_rate"]
        rows.append({"shop_no": k, "shop": name, "hours": hrs, "material_value": mat, "labour_cost": lab,
                     "labour_oncost": lab * pf["loc_pct"].get(k, 0) / 100, "stores_oncost": mat * pf["soc_pct"].get(k, 0) / 100})
    by_shop = pd.DataFrame(rows)
    by_shop["shop_cost"] = by_shop[ELEMENTS].sum(axis=1)
    out = {e: by_shop[e].sum() for e in ELEMENTS}
    out.update({"hours": by_shop.hours.sum(), "total": by_shop.shop_cost.sum(), "by_shop": by_shop})
    return out


def train_cost_model(d, random_state: int = 0) -> dict:
    """One gradient-boosting model per shop for man-hours and per shop for material; trained on 'train', tested on Railway-held year."""
    f = feature_frame(d)
    types = sorted(f.rs_type_code.unique())
    X = design_matrix(f, types)
    hrs = d["target_shop_cost"].pivot_table(index="job_no", columns="shop_no", values="man_hours").reindex(f.job_no).fillna(0)
    mat = base_price_material_by_shop(d).reindex(f.job_no).fillna(0)
    tr = (f.split == "train").values
    models, cv_pred = {}, pd.DataFrame(index=f.job_no[tr].values)
    for k in SHOPS:
        for name, y in ((f"hours_{k}", hrs[k].values), (f"mat_{k}", mat[k].values if k in mat else np.zeros(len(f)))):
            m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.06, subsample=0.8, random_state=random_state)
            cv_pred[name] = cross_val_predict(m, X[tr], y[tr], cv=5)
            models[name] = m.fit(X[tr], y[tr])
    model = {"models": models, "types": types, "features": list(X.columns)}
    ttr = d["target_coach_cost"].set_index("job_no").loc[f.job_no[tr]]
    pred_tr = np.array([price_prediction(d, cv_pred.loc[jn], fy)["total"] for jn, fy in zip(f.job_no[tr], f.fy[tr])])
    ratio = ttr.total_cost_per_coach.values / pred_tr
    model["range"] = (float(np.percentile(ratio, 10)), float(np.percentile(ratio, 90)))
    te = ~tr
    P = predict_quantities(model, X[te])
    tt = d["target_coach_cost"].set_index("job_no").loc[f.job_no[te]]
    pred_te = np.array([price_prediction(d, P.iloc[i], fy)["total"] for i, fy in enumerate(f.fy[te])])
    test = pd.DataFrame({"job_no": f.job_no[te].values, "rs_type_code": f.rs_type_code[te].values,
                         "month": f.date_dispatched[te].dt.to_period("M").dt.to_timestamp().values,
                         "actual": tt.total_cost_per_coach.values, "predicted": pred_te})
    test["abs_pct_error"] = (test.predicted - test.actual).abs() / test.actual
    model["test"] = test
    model["mape"] = float(test.abs_pct_error.mean())
    model["mape_by_type"] = test.groupby("rs_type_code").abs_pct_error.mean().sort_values()
    model["monthly"] = test.groupby("month").abs_pct_error.mean().reset_index()   # AI-12 monitoring
    imp = pd.DataFrame({n: m.feature_importances_ for n, m in models.items()}, index=X.columns)
    model["importance"] = imp.mean(axis=1).sort_values(ascending=False)
    return model


def predict_quantities(model, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({n: m.predict(X[model["features"]]) for n, m in model["models"].items()}, index=X.index)


def what_if(d, model, inputs: dict, fy: str) -> dict:
    X = design_matrix(pd.DataFrame([inputs]), model["types"])
    out = price_prediction(d, predict_quantities(model, X).iloc[0], fy)
    lo, hi = model["range"]
    out["low"], out["high"] = out["total"] * lo, out["total"] * hi
    return out


# ------------------------------------------------------------------ decision log (AI-05, AI-13)
LOG_COLS = ["timestamp", "flag_id", "rule", "job_no", "ref", "decision", "remarks", "reviewer"]


def read_log(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=LOG_COLS)


def append_log(path: str, entry: dict) -> None:
    pd.DataFrame([entry], columns=LOG_COLS).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
