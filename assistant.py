"""
AI-WCS Costing Assistant, Claude edition (BRD v0.4: AI-08 natural-language queries, AI-10 human in the loop,
AI-13 logging).

How it works (safe by design, same as the original):
  1. UNDERSTAND the question  -> a small structured "query spec" (metric, aggregation, filters, grouping).
       - Built-in rule parser: works offline (English + common Hindi words). It passes the AI-08 test on its own.
       - Optional Claude (Anthropic API): fills the same query spec through a strict JSON schema.
         Claude ONLY fills the spec; it never writes code and never sees the costing tables.
  2. COMPUTE the answer with pandas from the costing tables (no AI involved in the numbers).
  3. SHOW the answer, the query used and the source rows (AI-08 acceptance criterion).
     When Claude is on, it rewrites the computed result as a short reply, using only the numbers it is given.

Claude needs ANTHROPIC_API_KEY (or an `ant auth login` profile). Model: AIWCS_CLAUDE_MODEL (default claude-opus-5).
"""
from __future__ import annotations
import json, os, re
import pandas as pd
import aiwcs_logic as L

CLAUDE_MODEL = os.environ.get("AIWCS_CLAUDE_MODEL", "claude-opus-5")
# Server-side fallback: if a request is declined by a safety classifier, the API re-runs it on a suitable model.
_BETAS = ["server-side-fallback-2026-07-01"]
_FALLBACK = {"fallbacks": "default"}

# ------------------------------------------------------------------ vocabulary
TYPE_WORDS = [  # order matters: "non-ac" before "ac"
    (r"lhb\s*(non[\s-]*ac|nac)", "LHBNAC"), (r"lhb\s*ac", "LHBAC"),
    (r"icf\s*(non[\s-]*ac|nac)", "ICFNAC"), (r"icf\s*ac", "ICFAC"),
    (r"vande\s*bharat|\bvb\b", "VB"), (r"\bdemu", "DEMU"), (r"\bemu", "EMU"),
    (r"wagon", "WAGON"), (r"\bloco", "LOCO"),
]
SHOP_WORDS = {"inspection": 1, "lifting": 2, "bogie": 3, "body": 4, "air brake": 5, "brake shop": 5,
              "electrical": 6, "furnishing": 6, "paint": 7, "testing": 8}
HINDI = {"लागत": " cost ", "खर्च": " cost ", "औसत": " average ", "कुल": " total ", "श्रम": " labour ", "मजदूरी": " labour ",
         "सामग्री": " material ", "कितने": " how many ", "कितना": " how much ", "घंटे": " hours ", "शॉप": " shop ",
         "वर्ष": " fy ", "सबसे अधिक": " most ", "सबसे ज्यादा": " most ", "सबसे कम": " lowest ", "डिब्बे": " coaches ",
         "वैगन": " wagon ", "इंजन": " loco ", "ओवरहेड": " on-cost ", "ऑनकॉस्ט": " on-cost "}
ITEM_ALIASES = {"air spring": "PL10010003", "wheelset": "PL10010001", "wheel set": "PL10010001", "bearing": "PL10010002",
                "brake disc": "PL10010008", "brake pad": "PL10010009", "damper": "PL10010005", "yaw damper": "PL10010007",
                "paint": "PL10050001", "primer": "PL10050002", "electrode": "PL10020003", "welding wire": "PL10020003",
                "steel sheet": "PL10020001", "trough floor sheet": "PL10020002", "cbc knuckle": "PL10030005",
                "coupler kit": "PL10030004", "led": "PL10040001", "cable": "PL10040002", "cushion": "PL10040003",
                "window glass": "PL10040004", "ac unit": "PL10040006", "dg set": "PL10040007"}

METRICS = {  # name: (description, default aggregation)
    "units": ("units received for POH", "count"),
    "total_cost": ("total POH cost", "mean"),
    "material_value": ("direct material cost", "mean"),
    "labour_cost": ("direct labour cost", "mean"),
    "labour_oncost": ("labour on-cost", "sum"),
    "stores_oncost": ("stores on-cost", "sum"),
    "oncost_total": ("labour + stores on-cost", "sum"),
    "man_hours": ("man-hours booked", "sum"),
    "km": ("km run since last POH", "mean"),
    "floor_days": ("working days on the floor (receipt to dispatch)", "mean"),
    "item_value": ("value of an item drawn (net of returns, Book Average Rate)", "sum"),
    "item_qty": ("quantity of an item", "sum"),
    "idle_hours": ("approved idle hours", "sum"),
    "labour_oncost_rate": ("labour on-cost rate (% of direct labour)", "mean"),
    "stores_oncost_rate": ("stores on-cost rate (% of material)", "mean"),
    "blank_job_issues": ("material issue notes without job number / work order", "count"),
}
AGGS = {"mean": "average", "sum": "total", "count": "number", "max": "highest", "min": "lowest", "argmax": "highest", "argmin": "lowest"}
GROUPS = ["shop", "rs_type", "fy", "month"]


# ------------------------------------------------------------------ 1a. rule-based understanding
def _normalise(q: str) -> str:
    for h, e in HINDI.items():
        q = q.replace(h, e)
    q = q.lower().replace("on cost", "on-cost").replace("oncost", "on-cost").replace("man hours", "man-hours")
    return re.sub(r"\s+", " ", q)


def _amount(q: str):
    m = re.search(r"(more|greater|above|over|less|below|under)\s+than\s+(?:rs\.?|₹|inr)?\s*([\d.,]+)\s*(lakh|crore|cr|l)?", q) or \
        re.search(r"(above|over|below|under)\s+(?:rs\.?|₹|inr)?\s*([\d.,]+)\s*(lakh|crore|cr|l)?", q)
    if not m:
        return None
    v = float(m.group(2).replace(",", ""))
    v *= {"lakh": 1e5, "l": 1e5, "crore": 1e7, "cr": 1e7}.get(m.group(3) or "", 1)
    return (">" if m.group(1) in ("more", "greater", "above", "over") else "<", v)


def rule_parse(question: str, d) -> dict:
    q = _normalise(question)
    spec = {"metric": None, "agg": None, "group_by": None, "fy": [], "rs_type": [], "shop": [], "item": None,
            "threshold": None, "issue_only": False}
    spec["fy"] = sorted({f"{a}-{b}" for a, b in re.findall(r"(20\d\d)\s*[-/]\s*(\d\d)\b", q)})
    if not spec["fy"]:
        yrs = re.findall(r"\b(20\d\d)\b", q)
        spec["fy"] = sorted({f"{y}-{(int(y)+1) % 100:02d}" for y in yrs})
    for pat, code in TYPE_WORDS:
        if re.search(pat, q) and code not in spec["rs_type"]:
            if code in ("EMU",) and "DEMU" in spec["rs_type"] and not re.search(r"(^|[^d])emu", q):
                continue
            spec["rs_type"].append(code)
    if not spec["rs_type"] and re.search(r"\bac coach", q) and "non" not in q:
        spec["rs_type"] = ["ICFAC", "LHBAC", "VB"]
    # item
    for alias, pl in sorted(ITEM_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in q and not (alias == "paint" and "shop" in q):
            spec["item"] = pl; break
    for w, k in SHOP_WORDS.items():
        if re.search(rf"\b{w}\b", q) and (spec["item"] is None or w != "paint") and k not in spec["shop"]:
            spec["shop"].append(k)
    # metric
    if re.search(r"(without|no|blank|missing) (job|work ?order)", q):
        spec["metric"] = "blank_job_issues"
    elif "idle" in q:
        spec["metric"] = "idle_hours"
    elif "on-cost rate" in q or ("on-cost" in q and ("rate" in q or "%" in q or "percent" in q)):
        spec["metric"] = "stores_oncost_rate" if "stores" in q else "labour_oncost_rate"
    elif "labour on-cost" in q:
        spec["metric"] = "labour_oncost"
    elif "stores on-cost" in q:
        spec["metric"] = "stores_oncost"
    elif "on-cost" in q or "overhead" in q:
        spec["metric"] = "oncost_total"
    elif spec["item"]:
        spec["metric"] = "item_qty" if re.search(r"how many|number of|quantity|qty", q) else "item_value"
        spec["issue_only"] = bool(re.search(r"\bissued\b", q)) and "net" not in q
    elif re.search(r"\bkm\b|kilomet", q):
        spec["metric"] = "km"
    elif re.search(r"on the floor|floor time|turn ?round|days", q):
        spec["metric"] = "floor_days"
    elif re.search(r"man-hours|manhours|hours", q):
        spec["metric"] = "man_hours"
    elif "labour" in q or "labor" in q:
        spec["metric"] = "labour_cost"
    elif "material" in q:
        spec["metric"] = "material_value"
    elif re.search(r"how many (units|coaches|wagons|locos|rolling stock|vehicles)|number of (units|coaches)|received|turned out", q) and "cost" not in q:
        spec["metric"] = "units"
    elif "cost" in q or "expenditure" in q or "spent" in q:
        spec["metric"] = "total_cost"
    # threshold -> count of units
    spec["threshold"] = _amount(q)
    if spec["threshold"] and re.search(r"how many|number of|count", q):
        spec["agg"] = "count"
        if spec["metric"] in (None, "units"):
            spec["metric"] = "total_cost"
    # grouping / ranking
    if re.search(r"which shop|shop-?wise|by shop|each shop|per shop", q):
        spec["group_by"] = "shop"
    elif re.search(r"which (rolling stock|type|coach|variant)|by type|type-?wise|each type|per type", q):
        spec["group_by"] = "rs_type"
    elif re.search(r"monthly|by month|month-?wise|trend", q):
        spec["group_by"] = "month"
    elif re.search(r"by (year|fy)|year-?wise|each year", q):
        spec["group_by"] = "fy"
    if spec["agg"] is None:
        if re.search(r"\b(most|highest|maximum|max|costliest|largest)\b", q):
            spec["agg"] = "argmax" if spec["group_by"] else "max"
        elif re.search(r"\b(least|lowest|minimum|min|cheapest|smallest)\b", q):
            spec["agg"] = "argmin" if spec["group_by"] else "min"
        elif re.search(r"average|avg|mean|per (unit|coach)", q):
            spec["agg"] = "mean"
        elif re.search(r"total|sum|overall|how much", q):
            spec["agg"] = "sum"
        elif re.search(r"how many|number of|count", q):
            spec["agg"] = "count" if spec["metric"] in ("units", "blank_job_issues", None) else "sum"
    spec["_matched"] = spec["metric"] is not None or bool(spec["item"])
    if spec["metric"] is None:
        spec["metric"] = "total_cost"
    if spec["agg"] is None:
        spec["agg"] = METRICS[spec["metric"]][1]
    return spec


# ------------------------------------------------------------------ 1b. optional Claude understanding
_client = None


def claude_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


def _spec_schema(d) -> dict:
    types = sorted({c for _, c in TYPE_WORDS})
    return {
        "type": "object",
        "properties": {
            "metric": {"type": "string", "enum": list(METRICS)},
            "agg": {"type": "string", "enum": list(AGGS)},
            "group_by": _nullable({"type": "string", "enum": GROUPS}),
            "fy": {"type": "array", "items": {"type": "string"}},
            "rs_type": {"type": "array", "items": {"type": "string", "enum": types}},
            "shop": {"type": "array", "items": {"type": "integer", "enum": list(L.SHOPS)}},
            "item": _nullable({"type": "string", "enum": sorted(set(ITEM_ALIASES.values()))}),
            "threshold": _nullable({
                "type": "object",
                "properties": {"op": {"type": "string", "enum": [">", "<"]}, "amount_rupees": {"type": "number"}},
                "required": ["op", "amount_rupees"], "additionalProperties": False}),
            "issue_only": {"type": "boolean"},
        },
        "required": ["metric", "agg", "group_by", "fy", "rs_type", "shop", "item", "threshold", "issue_only"],
        "additionalProperties": False,
    }


def _parse_prompt(d) -> str:
    types = d["coach_master"].drop_duplicates("rs_type_code").set_index("rs_type_code").rs_type.to_dict()
    return ("You convert Indian Railways workshop costing questions (English or Hindi) into a JSON query spec. "
            "Fill every field from the schema.\n"
            f"Metric meanings: {json.dumps({k: v[0] for k, v in METRICS.items()})}\n"
            f"Rolling stock type codes: {json.dumps(types)}\n"
            f"Shop numbers: {json.dumps(L.SHOPS)}\n"
            f"Item PL codes: {json.dumps(ITEM_ALIASES)}\n"
            "Rules: financial years are written 'YYYY-YY' (e.g. '2024-25'); a bare year 2024 means '2024-25'. "
            "Use agg 'argmax'/'argmin' with group_by for 'which ... highest/lowest'. "
            "threshold is for 'more/less than <amount>' (convert lakh = 1e5, crore = 1e7 rupees), else null. "
            "issue_only is true only if the question says 'issued'. Leave lists empty when not mentioned.")


def claude_parse(question: str, d, model: str = CLAUDE_MODEL) -> dict:
    """Ask Claude to fill the query spec. Raises on any failure (caller falls back to rule_parse)."""
    r = claude_client().beta.messages.create(
        model=model, max_tokens=2000, betas=_BETAS, **_FALLBACK,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": _spec_schema(d)}},
        system=_parse_prompt(d),
        messages=[{"role": "user", "content": question}])
    if r.stop_reason != "end_turn":
        raise RuntimeError(f"Claude stopped: {r.stop_reason}")
    spec = json.loads(next(b.text for b in r.content if b.type == "text"))
    th = spec.get("threshold")
    spec["threshold"] = [th["op"], th["amount_rupees"]] if th else None
    return validate(spec)


def validate(spec: dict) -> dict:
    """Never trust model output: keep only allowed values."""
    out = {"metric": spec.get("metric") if spec.get("metric") in METRICS else "total_cost",
           "agg": spec.get("agg") if spec.get("agg") in AGGS else None,
           "group_by": spec.get("group_by") if spec.get("group_by") in GROUPS else None,
           "fy": [f for f in (spec.get("fy") or []) if re.fullmatch(r"20\d\d-\d\d", str(f))],
           "rs_type": [t for t in (spec.get("rs_type") or []) if t in {c for _, c in TYPE_WORDS}],
           "shop": [int(s) for s in (spec.get("shop") or []) if str(s).isdigit() and int(s) in L.SHOPS],
           "item": spec.get("item") if spec.get("item") in set(ITEM_ALIASES.values()) else None,
           "threshold": None, "issue_only": bool(spec.get("issue_only")), "_matched": True}
    th = spec.get("threshold")
    if isinstance(th, (list, tuple)) and len(th) == 2 and th[0] in (">", "<"):
        try: out["threshold"] = (th[0], float(th[1]))
        except (TypeError, ValueError): pass
    if out["agg"] is None:
        out["agg"] = METRICS[out["metric"]][1]
    if out["agg"] in ("argmax", "argmin") and out["group_by"] is None:
        out["agg"] = "max" if out["agg"] == "argmax" else "min"
    return out


# ------------------------------------------------------------------ 2. compute (pandas only)
def _frames(d):
    u = L.coach_table(d).merge(d["poh_jobs"][["job_no", "km_since_last_poh"]], on="job_no")
    import numpy as np
    u["floor_days"] = np.busday_count(u.date_received.values.astype("datetime64[D]"), u.date_dispatched.values.astype("datetime64[D]"),
                                      weekmask="1111110")  # working days, Monday to Saturday
    u["km"] = u.km_since_last_poh
    u["total_cost"] = u.total_cost_per_coach
    u["month"] = u.month_dispatched.dt.strftime("%Y-%m")
    s = L.shop_cost_table(d).merge(d["coach_master"].drop_duplicates("rs_type_code")[["rs_type_code", "rs_type"]], on="rs_type_code")
    s["total_cost"] = s.shop_cost
    s["oncost_total"] = s.labour_oncost + s.stores_oncost
    s = s.merge(u[["job_no", "month"]], on="job_no")
    u["oncost_total"] = u.labour_oncost + u.stores_oncost
    return u, s


def run(spec: dict, d) -> dict:
    u, s = _frames(d)
    m, agg, g = spec["metric"], spec["agg"], spec["group_by"]
    tnames = d["coach_master"].drop_duplicates("rs_type_code").set_index("rs_type_code").rs_type.to_dict()
    where = []
    # choose the table the metric lives in
    if m in ("item_value", "item_qty", "blank_job_issues"):
        df = d["material_issues"].copy()
        df["fy"] = L.fy_of(df.date)
        df = df.merge(d["poh_jobs"][["job_no", "rs_type_code"]], on="job_no", how="left")
        df["rs_type"] = df.rs_type_code.map(tnames)
        df["shop"] = df.shop_no.map(L.SHOPS); df["month"] = df.date.dt.strftime("%Y-%m")
        if m == "blank_job_issues":
            df = df[df.job_no == ""]; col = "issue_note_no"; agg = "count"
        else:
            if spec["item"]:
                df = df[df.pl_no == spec["item"]]
                where.append(f"item = {d['item_master'].set_index('pl_no').description.get(spec['item'], spec['item'])}")
            if m == "item_qty":
                if spec["issue_only"]:
                    df = df[df.txn_type == "ISSUE"]; where.append("issues only")
                df = df.assign(qty=df.quantity.astype(float) * df.txn_type.map({"ISSUE": 1, "RETURN": -1}).where(~pd.Series(spec["issue_only"], index=df.index), 1))
                col = "qty"
            else:
                col = "value"
            if agg in ("mean", "count"):
                agg = "sum"
        level = "material issue notes"
    elif m == "idle_hours":
        df = d["idle_time"].merge(d["staff_master"][["staff_id", "shop", "primary_shop_no"]], on="staff_id")
        df["fy"] = L.fy_of(df.date); df["month"] = df.date.dt.strftime("%Y-%m"); df["shop_no"] = df.primary_shop_no
        col, level = "idle_hours", "staff-days"
        if agg in ("mean", "count"):
            agg = "sum"
    elif m in ("labour_oncost_rate", "stores_oncost_rate"):
        df = d["oncost_rates"].copy(); col = m.replace("_rate", "_pct"); level = "on-cost rate master"
    elif m in ("units", "km", "floor_days") or (not spec["shop"] and g != "shop"):
        df, level = u, "units"
        col = "job_no" if m == "units" else m
        if m == "units":
            agg = "count" if agg not in ("argmax", "argmin") else agg
    else:
        df, level = s, "unit-shop rows"
        col = m if m not in ("units",) else "job_no"
    # filters
    if spec["fy"] and "fy" in df:
        df = df[df.fy.isin(spec["fy"])]; where.append("FY " + ", ".join(spec["fy"]))
    if spec["rs_type"] and "rs_type_code" in df:
        df = df[df.rs_type_code.isin(spec["rs_type"])]; where.append("type " + ", ".join(tnames.get(t, t) for t in spec["rs_type"]))
    if spec["shop"] and "shop_no" in df:
        df = df[df.shop_no.isin(spec["shop"])]; where.append(", ".join(L.SHOPS[k] for k in spec["shop"]))
    if spec["threshold"] and col in df and m not in ("item_value", "item_qty"):
        op, v = spec["threshold"]
        df = df[df[col] > v] if op == ">" else df[df[col] < v]; where.append(f"{L.ELEMENT_LABEL.get(col, col)} {op} ₹{v:,.0f}")
        if agg not in ("argmax", "argmin"):
            agg, col = "count", "job_no" if "job_no" in df else col
    query = {"table": level, "metric": METRICS[m][0], "aggregation": AGGS[agg], "filters": where or ["none"], "group by": g or "-"}
    if df.empty:
        return {"answer": "No records match this question.", "table": df.head(0), "query": query, "rows": df.head(0)}
    fn = {"mean": "mean", "sum": "sum", "count": "count", "max": "max", "min": "min", "argmax": "sum", "argmin": "sum"}[agg]
    if agg in ("argmax", "argmin") and (m in ("labour_oncost_rate", "stores_oncost_rate", "km", "floor_days")
                                        or (g == "rs_type" and m in ("total_cost", "material_value", "labour_cost", "man_hours"))):
        fn = "mean"   # compare types per unit, not by how many units they had
    if g:
        gcol = {"shop": "shop", "rs_type": "rs_type", "fy": "fy", "month": "month"}[g]
        if gcol not in df:
            gcol = "fy"
        tab = df.groupby(gcol)[col].agg(fn).reset_index().rename(columns={col: METRICS[m][0]})
        val_col = METRICS[m][0]
        if fn == "mean" and agg in ("argmax", "argmin"):
            query["aggregation"] = f"{AGGS[agg]} average per unit"
        if agg in ("argmax", "argmin"):
            row = tab.loc[tab[val_col].idxmax() if agg == "argmax" else tab[val_col].idxmin()]
            per = " per unit (average)" if fn == "mean" and m not in ("labour_oncost_rate", "stores_oncost_rate") else ""
            ans = f"**{row[gcol]}** has the {AGGS[agg]} {METRICS[m][0]}{per}: **{_fmt(m, row[val_col])}**"
        else:
            ans = f"{AGGS[agg].capitalize()} {METRICS[m][0]} by {g.replace('rs_type', 'rolling stock type')}:"
        tab = tab.sort_values(val_col, ascending=agg == "argmin")
    else:
        if agg in ("max", "min") and col in df:
            idx = df[col].idxmax() if agg == "max" else df[col].idxmin()
            v = df.loc[idx, col]; who = df.loc[idx, "job_no"] if "job_no" in df else ""
            ans = f"The {AGGS[agg]} {METRICS[m][0]} is **{_fmt(m, v)}**" + (f" ({who})" if who else "")
        else:
            v = df[col].agg(fn)
            if agg == "count":
                ans = f"Number of {METRICS[m][0] if m in ('units', 'blank_job_issues') else level}: **{_fmt('units', v)}**"
            else:
                ans = f"{AGGS[agg].capitalize()} {METRICS[m][0]}: **{_fmt(m, v)}**"
            if agg == "mean":
                ans += f" (over {len(df):,} {level})"
        tab = None
    return {"answer": ans, "table": tab, "query": query, "rows": df.head(200), "value": None if g else v}


def _fmt(metric, v):
    if metric in ("units", "blank_job_issues"):
        return f"{int(v):,}"
    if metric in ("man_hours", "idle_hours"):
        return f"{v:,.1f} hours"
    if metric == "km":
        return f"{v:,.0f} km"
    if metric == "floor_days":
        return f"{v:,.1f} days"
    if metric.endswith("_rate"):
        return f"{v:,.1f}%"
    if metric == "item_qty":
        return f"{v:,.1f}"
    return f"₹ {v:,.0f}" + (f"  (₹ {v/1e5:,.2f} lakh)" if abs(v) >= 1e5 else "")


def ask(question: str, d, use_llm: bool = False, model: str = CLAUDE_MODEL) -> dict:
    source = "rule parser"
    spec = None
    if use_llm:
        try:
            spec = claude_parse(question, d, model); source = f"Claude ({model})"
        except Exception as e:  # no key / network / bad JSON -> fall back
            source = f"rule parser (Claude unavailable: {type(e).__name__})"
    if spec is None:
        spec = rule_parse(question, d)
    out = run(spec, d)
    out["spec"], out["understood_by"] = spec, source
    return out


def acceptance_test(d, tol: float = 0.01) -> pd.DataFrame:
    """AI-08: compare assistant answers with the expected answers in ai_test_questions."""
    rows = []
    for r in d["ai_test_questions"].itertuples():
        res = ask(r.question, d)
        exp = r.expected_answer
        try:
            e = float(exp); got = res.get("value")
            ok = got is not None and abs(float(got) - e) <= max(tol * abs(e), 0.5)
        except (TypeError, ValueError):
            ok = str(exp).lower() in res["answer"].lower()
        rows.append([r.q_id, r.question, exp, re.sub(r"\*\*", "", res["answer"]), "PASS" if ok else "CHECK"])
    return pd.DataFrame(rows, columns=["q_id", "question", "expected", "assistant answer", "result"])


# ------------------------------------------------------------------ 4. Claude chat
GLOSSARY = """AI-WCS: AI-Assisted Workshop Costing System for Periodic Overhaul (POH) of rolling stock (ICF and LHB coaches,
Vande Bharat, EMU, DEMU, wagons, locos), Ministry of Railways (BRD v0.4).
Cost per unit POH = Direct material + Direct labour + Labour on-cost + Stores on-cost.
Labour on-cost = a % of direct labour (shop overheads such as supervision, indirect labour, machinery upkeep, power).
Stores on-cost = a % of direct material (stores handling, storage, inspection, procurement overheads).
Material is valued at Book Average Rate (from IMMS). Labour hours come from shop-floor booking validated against WISE/WAMS attendance.
Variance = Actual - Standard: material used vs material required (standard BOM), labour time consumed vs allowed time.
Shops: Inspection, Lifting, Bogie, Body, Air Brake, Electrical & Furnishing, Paint, Testing. Each job is one unit under POH against a work order.
WIP: units not yet turned out. WMS: Workshop Manufacturing Suspense. WGR: workshop costing returns.
AI modules are advisory: cost prediction (AI-01, target error 5% or less), anomaly detection (AI-04), variance drafts and questions (AI-07, AI-08).
No AI output posts an accounting entry; every decision is logged."""

SYSTEM = ("You are the AI Assisted Workshop Costing assistant for an Indian Railways workshop. Answer briefly and clearly in the language of the "
          "question (English or Hindi). Use Indian number style (lakh, crore). Never invent figures: use only numbers given to you in "
          "COMPUTED RESULT. If no computed result is given and the user asks for figures, ask them to mention the year, rolling stock "
          "type or shop. Background:\n" + GLOSSARY)


def claude_status(model: str = CLAUDE_MODEL) -> str | None:
    """None when Claude is usable, else a short reason. Uses the Models API (no tokens are spent)."""
    try:
        import anthropic
    except ImportError:
        return "The anthropic package is not installed (pip install anthropic)."
    try:
        claude_client().with_options(timeout=8, max_retries=0).models.retrieve(model)
    except anthropic.AuthenticationError:
        return "Anthropic credentials were rejected. Check ANTHROPIC_API_KEY."
    except anthropic.NotFoundError:
        return f"Model '{model}' is not available to this API key."
    except anthropic.APIConnectionError:
        return "Cannot reach the Anthropic API (network or proxy)."
    except anthropic.APIError as e:
        return f"Anthropic API error: {type(e).__name__}."
    except (anthropic.AnthropicError, TypeError):   # no credentials configured at all
        return "No Anthropic credentials. Set ANTHROPIC_API_KEY and restart."
    return None


def chat_stream(messages, system: str = SYSTEM, model: str = CLAUDE_MODEL):
    """Yield Claude's reply as it is written."""
    with claude_client().beta.messages.stream(
            model=model, max_tokens=4000, betas=_BETAS, **_FALLBACK,
            output_config={"effort": "low"}, system=system, messages=messages) as s:
        yield from s.text_stream
        if s.get_final_message().stop_reason == "refusal":
            yield "\n\n_(Claude declined to answer this one.)_"


def converse(question: str, d, history: list, use_llm: bool, model: str = CLAUDE_MODEL):
    """Answer one chat turn. Returns (result_or_None, reply_text_or_stream)."""
    spec = rule_parse(question, d)
    ql = question.lower()
    definitional = re.match(r"\s*(what is|what are|what's|explain|define|meaning of|why|how (does|do|is)|difference between)", ql) \
        and not (spec["fy"] or spec["rs_type"] or spec["shop"]) \
        and not re.search(r"average|avg|total|how many|highest|lowest|most|least|by shop|monthly|which", ql)
    data_q = not definitional and (spec.get("_matched") or bool(spec["fy"] or spec["rs_type"] or spec["shop"]))
    # the built-in reader interprets questions it recognises (it passes the AI-08 test); Claude interprets the rest
    res = ask(question, d, use_llm=use_llm and not spec.get("_matched"), model=model) if data_q else None
    if not use_llm:
        if res is None:
            return None, ("I answer questions about the costing data, for example *Average POH cost of LHB AC coaches in 2024-25* "
                          "or *Which shop has the highest labour on-cost?* Switch on Claude in the sidebar for general questions.")
        return res, res["answer"]
    msgs = [{"role": h["role"], "content": h["content"]} for h in history[-6:]]
    while msgs and msgs[0]["role"] != "user":   # the API needs the conversation to start with the user
        msgs.pop(0)
    if res is not None:
        tab = res["table"].head(15).to_string(index=False) if res["table"] is not None else ""
        msgs.append({"role": "user", "content": f"QUESTION: {question}\nCOMPUTED RESULT: {re.sub(r'[*]', '', res['answer'])}\n{tab}\n"
                                                f"HOW COMPUTED: {res['query']}\nWrite a short answer using only these numbers."})
    else:
        msgs.append({"role": "user", "content": question})
    return res, chat_stream(msgs, model=model)
