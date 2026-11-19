"""
AI-WCS Dashboard — AI-Assisted Workshop Costing System for ROLLING STOCK POH (BRD v0.4), Claude edition.
Same design and outputs as the original app.py; the costing assistant uses Claude instead of a local Ollama model.
Run:  streamlit run app.py
Data: CSVs from AI-WCS_Dummy_Training_Data_v2_CSV.zip in ./data (or set AIWCS_DATA).
Claude (optional): set ANTHROPIC_API_KEY. Without it the assistant uses its built-in reader.
"""
import os, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import aiwcs_logic as L
import assistant as A
import wip as W

st.set_page_config(page_title="AI Assisted Workshop Costing", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("AIWCS_DATA", os.path.join(HERE, "data"))
LOG_PATH = os.environ.get("AIWCS_LOG", os.path.join(HERE, "decision_log.csv"))

# Categorical palette stepped for a black background; colour follows the entity, never its rank.
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#2fa84f", "#9085e9", "#e66767"]
SHOP_COLORS = dict(zip(L.SHOPS.values(), PALETTE))
ELEMENT_COLORS = dict(zip(["Direct material", "Direct labour", "Labour on-cost", "Stores on-cost"], PALETTE))
FAMILY = {"ICFNAC": "ICF coaches", "ICFAC": "ICF coaches", "LHBNAC": "LHB coaches", "LHBAC": "LHB coaches",
          "VB": "Vande Bharat", "EMU": "EMU / DEMU", "DEMU": "EMU / DEMU", "WAGON": "Wagons", "LOCO": "Locos"}
FAMILY_COLORS = dict(zip(["ICF coaches", "LHB coaches", "Vande Bharat", "EMU / DEMU", "Wagons", "Locos"], PALETTE))
BG, CARD, INK, MUTED = "#000000", "#101018", "#f2f2f5", "#a8a8b8"

st.markdown(f"""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background-color: {BG}; color: {INK}; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0b1a33 0%, #1a0b2e 55%, #000000 100%); }}
[data-testid="stSidebar"] * {{ color: {INK} !important; }}
h1, h2, h3, p, label, span, div {{ color: {INK}; }}
.banner {{ background: linear-gradient(90deg, #3987e5 0%, #9085e9 35%, #d55181 70%, #d95926 100%);
          border-radius: 14px; padding: 18px 24px; margin-bottom: 18px; }}
.banner h1 {{ color: #ffffff !important; margin: 0; font-size: 2rem; }}
.banner p {{ color: #ffffff !important; margin: 4px 0 0 0; opacity: .92; }}
.kpi {{ background: {CARD}; border-radius: 14px; padding: 14px 16px; border: 1px solid #23233a; height: 100%; }}
.kpi .bar {{ height: 6px; border-radius: 4px; margin-bottom: 10px; }}
.kpi .lab {{ color: {MUTED}; font-size: .85rem; }}
.kpi .val {{ color: #ffffff; font-size: 1.6rem; font-weight: 700; margin-top: 2px; }}
.pass {{ color: #2fa84f !important; font-weight: 700; }} .fail {{ color: #e66767 !important; font-weight: 700; }}
[data-testid="stExpander"], [data-testid="stForm"] {{ background: {CARD}; border-radius: 12px; border: 1px solid #23233a; }}
.stTabs [data-baseweb="tab"] {{ color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ color: #ffffff; border-bottom-color: #d55181 !important; }}

/* dropdown & input boxes on the dark theme (from app2) */
div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {{ background-color: {CARD} !important; border: 1px solid #23233a !important; }}
div[data-baseweb="select"] span, input[type="text"], input[type="number"] {{ color: {INK} !important; -webkit-text-fill-color: {INK} !important; }}
div[data-baseweb="popover"], div[data-baseweb="popover"] > div, ul[role="listbox"], ul[data-baseweb="menu"], div[role="listbox"] {{ background-color: {CARD} !important; }}
li[role="option"], li[data-baseweb="option"] {{ background-color: {CARD} !important; color: {INK} !important; }}
li[role="option"]:hover, li[data-baseweb="option"]:hover {{ background-color: #23233a !important; color: #ffffff !important; }}

/* floating AI button & chat container (from app2) */
.st-key-float_ask {{ position: fixed !important; bottom: 30px !important; right: 30px !important; width: 65px !important; z-index: 99999 !important; }}
.st-key-float_ask button {{ width: 65px !important; height: 65px !important; border-radius: 50% !important; background-color: #3987e5 !important; background-image: linear-gradient(135deg, #3987e5 0%, #9085e9 100%) !important; border: 2px solid #ffffff !important; box-shadow: 0px 0px 15px rgba(57, 135, 229, 0.8) !important; padding: 0 !important; display: flex !important; align-items: center !important; justify-content: center !important; }}
.st-key-float_ask button p {{ font-size: 30px !important; margin: 0 !important; color: white !important; }}
.st-key-float_ask button svg {{ display: none !important; }}
div[data-testid="stPopoverBody"] {{ width: 460px !important; max-width: 92vw !important; background-color: {CARD} !important; border: 1px solid #3987e5 !important; border-radius: 16px !important; padding: 0 14px 12px 14px !important; box-shadow: 0px 10px 40px rgba(0,0,0,0.8) !important; }}

/* chatbot: header, bubbles, starter questions */
.fc-head {{ background: linear-gradient(90deg, #3987e5 0%, #9085e9 35%, #d55181 70%, #d95926 100%); margin: 0 -14px 6px -14px;
           padding: 12px 16px; border-radius: 15px 15px 0 0; }}
.fc-head .t {{ color: #fff !important; font-weight: 700; font-size: 1.05rem; }}
.fc-head .s {{ color: #fff !important; opacity: .9; font-size: .78rem; margin-top: 2px; }}
.fc-head .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
.fc-hello {{ color: {MUTED} !important; font-size: .88rem; margin: 4px 0 8px 0; }}
.st-key-float_box button {{ background: #161624 !important; border: 1px solid #2c2c48 !important; border-radius: 12px !important;
    text-align: left !important; justify-content: flex-start !important; }}
.st-key-float_box button:hover {{ border-color: #9085e9 !important; }}
.st-key-float_box button p {{ font-size: .85rem !important; color: {INK} !important; }}
[data-testid="stChatMessage"] {{ background: transparent !important; padding: 4px 0 !important; gap: 8px; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ flex-direction: row-reverse; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{
    background: linear-gradient(135deg, #3987e5 0%, #9085e9 100%); border-radius: 16px 16px 4px 16px; padding: 8px 14px;
    flex: 0 1 auto; max-width: 85%; margin-left: auto; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] * {{ color: #fff !important; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {{
    background: #161624; border: 1px solid #23233a; border-radius: 16px 16px 16px 4px; padding: 8px 14px; }}
[data-testid="stChatMessageAvatarUser"] {{ background: #3987e5 !important; }}
[data-testid="stChatMessageAvatarAssistant"] {{ background: linear-gradient(135deg, #d55181 0%, #d95926 100%) !important; }}
.st-key-float_form [data-testid="InputInstructions"] {{ display: none !important; }}
.st-key-float_box button > div, .st-key-float_box button [data-testid="stMarkdownContainer"] {{ justify-content: flex-start !important; text-align: left !important; width: 100%; }}
.st-key-float_form [data-testid="stForm"] {{ border: 1px solid #2c2c48; border-radius: 14px; padding: 6px 8px; }}
</style>
""", unsafe_allow_html=True)


def banner(title, subtitle):
    st.markdown(f'<div class="banner"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(col, label, value, color):
    col.markdown(f'<div class="kpi"><div class="bar" style="background:{color}"></div>'
                 f'<div class="lab">{label}</div><div class="val">{value}</div></div>', unsafe_allow_html=True)


def rs(x):
    return f"₹ {x/1e5:,.2f} lakh"


def style(fig, y_title=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor=CARD, plot_bgcolor=CARD,
                      margin=dict(l=14, r=14, t=48, b=14), legend_title_text="", hovermode="closest",
                      font=dict(family="Arial", size=13, color=INK), title_font=dict(size=16, color="#ffffff"),
                      hoverlabel=dict(bgcolor="#1c1c2a", font_color="#ffffff"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)", title=y_title)
    fig.update_xaxes(showgrid=False, linecolor="rgba(255,255,255,0.2)")
    return fig


@st.cache_data(show_spinner="Loading data ...")
def get_data(path):
    return L.load_data(path)


@st.cache_resource(show_spinner="Training cost model (one-time, about 30 s) ...")
def get_model(path):
    return L.train_cost_model(get_data(path))


@st.cache_data(show_spinner="Running anomaly checks ...")
def get_flags(path, factor):
    return L.rule_flags(get_data(path), hours_factor=factor)


@st.cache_data
def get_recon(path):
    return L.labour_reconciliation(get_data(path))


@st.cache_data(show_spinner="Building the dated cost ledger ...")
def get_ledger(path):
    return W.ledger(get_data(path))


@st.cache_data(show_spinner="Valuing month-end WIP ...")
def get_wip_trend(path):
    return W.month_end_trend(get_data(path), get_ledger(path))


@st.cache_data(ttl=300, show_spinner="Checking Claude ...")
def claude_status(model):
    return A.claude_status(model)


try:
    d = get_data(DATA_DIR)
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("AI Assisted Workshop Costing")
st.sidebar.caption("Rolling stock POH costing · BRD v0.4 · DUMMY DATA")
PAGES = ["Overview", "Work in progress", "Costing assistant", "Unit cost & variance", "On-cost", "Labour reconciliation",
         "Material", "Anomaly review", "Cost prediction (AI)"]
page = st.sidebar.radio("Page", PAGES)
fys = sorted(d["poh_jobs"].fy.unique())
sel_fy = st.sidebar.multiselect("Financial year", fys, default=fys)
type_names = d["coach_master"].drop_duplicates("rs_type_code").set_index("rs_type_code").rs_type.to_dict()
types = sorted(d["poh_jobs"].rs_type_code.unique())
sel_type = st.sidebar.multiselect("Rolling stock type", types, default=types, format_func=lambda c: type_names.get(c, c))
st.sidebar.markdown("---")
st.sidebar.caption("Cost = Direct material + Direct labour + Labour on-cost + Stores on-cost (BRD §6). "
                   "Advisory AI only: no AI output posts an accounting entry (AI-10).")

T = L.coach_table(d)
T = T[T.fy.isin(sel_fy) & T.rs_type_code.isin(sel_type)]
T["family"] = T.rs_type_code.map(FAMILY)

# ------------------------------------------------------------------ pages
if page == "Overview":
    banner("Overview", "BR-01 cost per unit POH, shop-wise · KPIs from BRD section 3")
    k = L.kpis(T)
    c = st.columns(5)
    kpi(c[0], "Units turned out", f"{k['units']:,}", PALETTE[0])
    kpi(c[1], "Average cost per unit", rs(k["avg_cost"]), PALETTE[1])
    kpi(c[2], "Total POH cost", f"₹ {k['total_cost']/1e7:,.2f} crore", PALETTE[2])
    kpi(c[3], "On-cost share", f"{k['oncost_share']:.1%}", PALETTE[3])
    kpi(c[4], "Average man-hours per unit", f"{k['avg_hours']:,.0f}", PALETTE[4])
    st.write("")
    if T.empty:
        st.info("No units match the filters."); st.stop()
    left, right = st.columns(2)
    m = T.groupby("month_dispatched").total_cost_per_coach.mean().reset_index()
    fig = px.line(m, x="month_dispatched", y="total_cost_per_coach", markers=True, color_discrete_sequence=[PALETTE[0]],
                  title="Average cost per unit by month of dispatch", labels={"month_dispatched": "Month", "total_cost_per_coach": "₹ per unit"})
    fig.update_traces(line_width=3, marker_size=8, fill="tozeroy", fillcolor="rgba(57,135,229,0.15)")
    left.plotly_chart(style(fig, "₹ per unit"), width="stretch")
    e = L.element_by_fy(T)
    fig = px.bar(e, x="fy", y="avg_per_unit", color="element", color_discrete_map=ELEMENT_COLORS,
                 title="Average cost per unit by element", labels={"fy": "Financial year", "avg_per_unit": "₹ per unit"})
    right.plotly_chart(style(fig, "₹ per unit"), width="stretch")
    left, right = st.columns(2)
    v = T.groupby(["rs_type_code", "family"]).agg(units=("job_no", "count"), avg_cost=("total_cost_per_coach", "mean")).reset_index()
    v["type"] = v.rs_type_code.map(type_names)
    fig = px.bar(v.sort_values("avg_cost"), x="avg_cost", y="type", orientation="h", color="family", color_discrete_map=FAMILY_COLORS,
                 title="Average cost per unit by rolling stock type", hover_data=["units"], labels={"avg_cost": "₹ per unit", "type": ""})
    left.plotly_chart(style(fig), width="stretch")
    sc = L.shop_cost_table(d, T.job_no).groupby("shop")[L.ELEMENTS].sum().reset_index()
    sc = sc.melt(id_vars="shop", var_name="element", value_name="rs").assign(element=lambda x: x.element.map(L.ELEMENT_LABEL))
    fig = px.bar(sc, x="shop", y="rs", color="element", color_discrete_map=ELEMENT_COLORS,
                 title="Shop-wise cost (BR-01)", labels={"shop": "", "rs": "₹"})
    right.plotly_chart(style(fig, "₹"), width="stretch")

elif page == "Work in progress":
    banner("Work in progress", "Cost built up on units still on the floor, valued from dated labour bookings and issue notes · "
                               "as on any date, not after month-end")
    led = get_ledger(DATA_DIR)
    j = d["poh_jobs"]
    lo, hi = j.date_received.min().date(), (j.date_dispatched.max() - pd.Timedelta(days=1)).date()
    a, b = st.columns([1, 3])
    as_on = a.date_input("WIP as on", value=min(max(dt.date(2025, 11, 30), lo), hi), min_value=lo, max_value=hi)
    b.caption("Units on the floor = received on or before this date and not yet dispatched. Cost to date = labour cost + "
              "labour on-cost from labour bookings, and material + stores on-cost from issue notes (returns credited), up to "
              "this date. The rolling stock type filter applies; the financial year filter does not.")
    units, shops = W.wip_as_on(d, led, as_on)
    units = units[units.rs_type_code.isin(sel_type)]
    shops = W.wip_as_on(d, led[led.job_no.isin(units.job_no)], as_on)[1] if len(units) else shops.iloc[0:0]
    if units.empty:
        st.info("No units on the floor on this date for the selected types."); st.stop()

    # AI estimate at receipt, and how far each unit has run against it and against the usual time on the floor
    units["ai_estimate"] = units.job_no.map(W.ai_estimates(d, get_model(DATA_DIR), units.job_no))
    done = j[j.date_dispatched <= pd.Timestamp(as_on)].assign(days=lambda x: (x.date_dispatched - x.date_received).dt.days + 1)
    usual = done.groupby("rs_type_code").days.median()
    units["usual_days"] = units.rs_type_code.map(usual).fillna(done.days.median() if len(done) else 30)
    units["cost_used"] = units.cost_to_date / units.ai_estimate
    units["time_used"] = units.days_on_floor / units.usual_days
    units["expected_dispatch"] = units.date_received + pd.to_timedelta(units.usual_days.round() - 1, unit="D")
    units["status"] = np.select([units.cost_used > 1, units.cost_used - units.time_used > 0.25],
                                ["⛔ Past AI estimate", "⚠ Spending ahead of time"], "✅ On track")
    units["rs_type"] = units.rs_type_code.map(type_names)

    c = st.columns(5)
    kpi(c[0], "Units on the floor", f"{len(units):,}", PALETTE[0])
    kpi(c[1], "WIP value", f"₹ {units.cost_to_date.sum()/1e7:,.2f} crore", PALETTE[1])
    kpi(c[2], "Average cost to date per unit", rs(units.cost_to_date.mean()), PALETTE[2])
    kpi(c[3], "Average days on the floor", f"{units.days_on_floor.mean():,.0f}", PALETTE[3])
    kpi(c[4], "Units needing attention", f"{(units.status != '✅ On track').sum():,}", PALETTE[7])
    st.write("")

    left, right = st.columns(2)
    tr = get_wip_trend(DATA_DIR).copy()
    tr["label"] = np.where(tr.month_end.dt.to_period("M") == pd.Timestamp(as_on).to_period("M"), "Selected month", "Month end")
    fig = px.bar(tr, x="month_end", y=tr.wip / 1e7, color="label", hover_data={"units": True, "label": False},
                 color_discrete_map={"Month end": PALETTE[0], "Selected month": PALETTE[3]},
                 title="WIP at each month end (all types)", labels={"month_end": "Month end", "y": "₹ crore", "units": "Units on floor"})
    fig.update_layout(showlegend=False)
    left.plotly_chart(style(fig, "₹ crore"), width="stretch")
    sm = shops.melt(id_vars="shop", value_vars=W.ELEMENTS, var_name="element", value_name="rs")
    sm["element"] = sm.element.map(L.ELEMENT_LABEL)
    fig = px.bar(sm, x="shop", y="rs", color="element", color_discrete_map=ELEMENT_COLORS,
                 title=f"WIP by shop and element as on {as_on:%d %b %Y}", labels={"shop": "", "rs": "₹"})
    right.plotly_chart(style(fig, "₹"), width="stretch")

    st.subheader("Units on the floor")
    show = units.sort_values("cost_used", ascending=False)[
        ["job_no", "coach_no", "rs_type", "date_received", "expected_dispatch", "days_on_floor", "usual_days", "cost_to_date", "ai_estimate",
         "cost_used", "status"]]
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "date_received": st.column_config.DateColumn("Received", format="DD MMM YYYY"),
        "expected_dispatch": st.column_config.DateColumn("Expected dispatch", format="DD MMM YYYY",
                                                         help="Received + usual days on the floor for this type (median of past overhauls)"),
        "days_on_floor": "Days on floor", "usual_days": st.column_config.NumberColumn("Usual days", format="%d"),
        "cost_to_date": st.column_config.NumberColumn("Cost to date ₹", format="%,.0f"),
        "ai_estimate": st.column_config.NumberColumn("AI estimate at receipt ₹", format="%,.0f"),
        "cost_used": st.column_config.ProgressColumn("Share of estimate used", format="percent", min_value=0, max_value=1.2),
        "rs_type": "Type", "job_no": "Job", "coach_no": "Unit no.", "status": "Status"})
    st.caption("Status: past AI estimate = cost to date above the estimate made at receipt; spending ahead of time = share of the "
               "estimate used is more than 25 points ahead of the share of usual days on the floor. Advisory only (AI-10).")

    st.subheader("One unit, day by day")
    pick = st.selectbox("Unit", show.job_no, format_func=lambda x: f"{x} · {units.set_index('job_no').rs_type[x]}")
    u = units.set_index("job_no").loc[pick]
    cur = W.daily_curve(d, led, pick)
    cur = cur[cur.date <= pd.Timestamp(as_on)]
    cm = cur.melt(id_vars=["date", "day"], value_vars=W.ELEMENTS, var_name="element", value_name="rs")
    cm["element"] = cm.element.map(L.ELEMENT_LABEL)
    fig = px.area(cm, x="date", y="rs", color="element", color_discrete_map=ELEMENT_COLORS,
                  title=f"{pick}: cost built up from receipt to {as_on:%d %b %Y} · day {int(u.days_on_floor)}",
                  labels={"date": "", "rs": "₹ (cumulative)"})
    fig.add_hline(y=u.ai_estimate, line_dash="dot", line_color="#ffffff", annotation_text=f"AI estimate at receipt {rs(u.ai_estimate)}")
    st.plotly_chart(style(fig, "₹ (cumulative)"), width="stretch")
    c = st.columns(3)
    kpi(c[0], "Cost to date", rs(u.cost_to_date), PALETTE[1])
    kpi(c[1], "Share of AI estimate used", f"{u.cost_used:.0%}", PALETTE[3])
    kpi(c[2], "Days on floor / usual", f"{int(u.days_on_floor)} / {int(u.usual_days)}", PALETTE[0])

elif page == "Unit cost & variance":
    banner("Unit cost & variance", "BR-01, FR-06 · time consumed vs allowed time; material used vs material required, by shop")
    show = T[["job_no", "coach_no", "rs_type", "fy", "man_hours", "material_value", "labour_cost", "labour_oncost",
              "stores_oncost", "total_cost_per_coach"]]
    st.dataframe(show.sort_values("total_cost_per_coach", ascending=False), width="stretch", height=280)
    if show.empty:
        st.stop()
    job = st.selectbox("Select a job", show.job_no)
    sv = L.shop_variance(d, job)
    j = d["poh_jobs"].set_index("job_no").loc[job]
    st.subheader(f"{job} · {j.coach_no} · {type_names.get(j.rs_type_code, j.rs_type_code)} · FY {j.fy}")
    c1, c2 = st.columns(2)
    hv = sv.melt(id_vars="shop", value_vars=["man_hours", "allowed_hours"], var_name="series", value_name="hours")
    hv["series"] = hv.series.map({"man_hours": "Time consumed", "allowed_hours": "Allowed time"})
    fig = px.bar(hv, x="shop", y="hours", color="series", barmode="group",
                 color_discrete_map={"Time consumed": PALETTE[0], "Allowed time": PALETTE[3]},
                 title="Labour: time consumed vs allowed time", labels={"shop": ""})
    c1.plotly_chart(style(fig, "Man-hours"), width="stretch")
    mv = sv.melt(id_vars="shop", value_vars=["material_value", "std_material_value"], var_name="series", value_name="rs")
    mv["series"] = mv.series.map({"material_value": "Material used", "std_material_value": "Material required (standard)"})
    fig = px.bar(mv, x="shop", y="rs", color="series", barmode="group",
                 color_discrete_map={"Material used": PALETTE[1], "Material required (standard)": PALETTE[2]},
                 title="Material: used vs required", labels={"shop": "", "rs": "₹"})
    c2.plotly_chart(style(fig, "₹"), width="stretch")
    sc = sv.melt(id_vars="shop", value_vars=L.ELEMENTS, var_name="element", value_name="rs")
    sc["element"] = sc.element.map(L.ELEMENT_LABEL)
    fig = px.bar(sc, x="shop", y="rs", color="element", color_discrete_map=ELEMENT_COLORS,
                 title="Cost by shop and element", labels={"shop": "", "rs": "₹"})
    st.plotly_chart(style(fig, "₹"), width="stretch")
    st.dataframe(sv[["shop", "man_hours", "allowed_hours", "time_variance_hours", "time_efficiency", "material_value",
                     "std_material_value", "material_variance", "shop_cost"]].round(2), width="stretch")
    st.caption("Conditional items (wheelsets, air springs, corrosion steel) have no standard quantity, so they appear as material variance.")
    with st.expander("As-received condition (drives the variance)"):
        st.write(d["condition_findings"].set_index("job_no").loc[job].astype(str).to_frame("value"))

elif page == "On-cost":
    banner("On-cost", "BRD §6 · labour on-cost on direct labour and stores on-cost on direct material, by shop")
    oc = L.oncost_summary(d, sel_fy)
    tot = oc[["labour_oncost", "stores_oncost"]].sum()
    c = st.columns(3)
    kpi(c[0], "Labour on-cost charged", f"₹ {tot.labour_oncost/1e7:,.2f} crore", PALETTE[2])
    kpi(c[1], "Stores on-cost charged", f"₹ {tot.stores_oncost/1e7:,.2f} crore", PALETTE[3])
    kpi(c[2], "Average labour on-cost rate", f"{(oc.labour_oncost.sum()/oc.labour_cost.sum()):.0%} of direct labour", PALETTE[6])
    st.write("")
    left, right = st.columns(2)
    g = oc.groupby("shop")[["labour_oncost", "stores_oncost"]].sum().reset_index().melt(id_vars="shop", var_name="type", value_name="rs")
    g["type"] = g.type.map({"labour_oncost": "Labour on-cost", "stores_oncost": "Stores on-cost"})
    fig = px.bar(g, x="shop", y="rs", color="type", color_discrete_map=ELEMENT_COLORS, title="On-cost charged by shop",
                 labels={"shop": "", "rs": "₹"})
    left.plotly_chart(style(fig, "₹"), width="stretch")
    fig = px.bar(oc, x="shop", y="labour_oncost_pct", color="fy", barmode="group",
                 color_discrete_sequence=[PALETTE[0], PALETTE[6], PALETTE[4]],
                 title="Labour on-cost rate by shop and year", labels={"shop": "", "labour_oncost_pct": "% of direct labour"})
    right.plotly_chart(style(fig, "% of direct labour"), width="stretch")
    st.subheader("On-cost rates (master, FC-08)")
    st.dataframe(d["oncost_rates"][d["oncost_rates"].fy.isin(sel_fy)], width="stretch", height=260)
    with st.expander("FR-07 parallel run: on-cost method vs driver-based overhead (comparison only)"):
        pr = L.parallel_run(T)
        pr["family"] = pr.rs_type_code.map(FAMILY)
        fig = px.scatter(pr, x="oncost_method", y="parallel_driver_based_overhead", color="family", color_discrete_map=FAMILY_COLORS,
                         hover_data=["job_no"], title="Overhead per unit: on-cost method vs driver-based method",
                         labels={"oncost_method": "On-cost method ₹", "parallel_driver_based_overhead": "Driver-based ₹"})
        lo, hi = pr[["oncost_method", "parallel_driver_based_overhead"]].min().min(), pr[["oncost_method", "parallel_driver_based_overhead"]].max().max()
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color=MUTED, dash="dot"))
        st.plotly_chart(style(fig, "Driver-based ₹"), width="stretch")
        chk = L.overhead_check(d)
        st.write(f"Driver-based pools: largest difference between allocated overhead and pool cost = ₹ {chk.difference.abs().max():,.2f}")

elif page == "Labour reconciliation":
    banner("Labour reconciliation", "FC-03 · paid hours = booked to units + approved idle + leave; the rest is unaccounted")
    x = get_recon(DATA_DIR)
    x = x[L.fy_of(x.date).isin(sel_fy)]
    g = L.recon_by_shop_month(x)
    shop = st.selectbox("Shop", ["All shops"] + sorted(g.shop.unique()))
    gg = g if shop == "All shops" else g[g.shop == shop]
    m = gg.groupby("month")[["booked", "idle", "leave", "unaccounted"]].sum().reset_index()
    mm = m.melt(id_vars="month", var_name="category", value_name="hours")
    mm["category"] = mm.category.map({"booked": "Booked to units", "idle": "Approved idle", "leave": "Leave", "unaccounted": "Unaccounted"})
    fig = px.bar(mm, x="month", y="hours", color="category", title=f"Paid hours breakdown · {shop}",
                 color_discrete_map={"Booked to units": PALETTE[0], "Approved idle": PALETTE[3], "Leave": PALETTE[6], "Unaccounted": PALETTE[7]},
                 labels={"month": "Month"})
    st.plotly_chart(style(fig, "Hours"), width="stretch")
    tot = gg[["paid_hours", "booked", "idle", "leave", "unaccounted"]].sum()
    c = st.columns(4)
    kpi(c[0], "Utilisation (booked ÷ paid)", f"{tot.booked/tot.paid_hours:.1%}", PALETTE[0])
    kpi(c[1], "Approved idle", f"{tot.idle:,.0f} h", PALETTE[3])
    kpi(c[2], "Unaccounted", f"{tot.unaccounted:,.0f} h", PALETTE[7])
    kpi(c[3], "Unaccounted % of paid (target < 2%)", f"{tot.unaccounted/tot.paid_hours:.2%}", PALETTE[2])
    st.write("")
    st.subheader("Staff-days with unaccounted hours")
    st.dataframe(x[x.unaccounted > 0][["staff_id", "shop", "date", "attended_hours", "booked", "idle", "unaccounted"]],
                 width="stretch", height=250)

elif page == "Material":
    banner("Material", "FR-04, FC-04, AI-03 · drawal at Book Average Rate; used vs standard BOM")
    ms = L.material_summary(d).head(15)
    ms["shop"] = ms.shop_no.map(L.SHOPS)
    fig = px.bar(ms.sort_values("value"), x="value", y="description", orientation="h", color="shop", color_discrete_map=SHOP_COLORS,
                 title="Top 15 items by value drawn (all years), coloured by shop", labels={"value": "₹", "description": "", "shop": "Shop"})
    st.plotly_chart(style(fig), width="stretch")
    q = d["target_material_qty"].merge(d["poh_jobs"][["job_no", "rs_type_code", "fy"]], on="job_no")
    q = q[q.fy.isin(sel_fy) & q.rs_type_code.isin(sel_type)]
    avg = q.groupby(["rs_type_code", "pl_no"]).net_qty.mean().rename("avg_qty_per_unit").reset_index()
    cmp_ = avg.merge(d["standard_bom"][["rs_type_code", "pl_no", "std_qty_per_poh"]], on=["rs_type_code", "pl_no"], how="left")
    cmp_["description"] = cmp_.pl_no.map(d["item_master"].set_index("pl_no").description)
    cmp_["used_vs_required"] = cmp_.avg_qty_per_unit / cmp_.std_qty_per_poh
    st.subheader("Average quantity per unit vs standard BOM (material used vs required)")
    st.dataframe(cmp_.round(2).sort_values(["rs_type_code", "pl_no"]), width="stretch", height=320)
    m = d["material_issues"]
    kpi(st.columns(3)[0], "Issue notes without job number / work order (FR-04)", f"{int((m.job_no == '').sum())}", PALETTE[7])

elif page == "Anomaly review":
    banner("Anomaly review", "AI-04 to AI-06, AI-10 to AI-13 · rule checks + ML flags; an officer closes each flag with remarks")
    factor = st.slider("R6 sensitivity: flag a unit-shop when hours exceed the type median ×", 1.2, 2.0, 1.35, 0.05)
    flags = get_flags(DATA_DIR, factor)
    log = L.read_log(LOG_PATH)
    flags["status"] = flags.flag_id.map(log.drop_duplicates("flag_id", keep="last").set_index("flag_id").decision.astype(str)).fillna("Open")
    c = st.columns(4)
    kpi(c[0], "Total flags", f"{len(flags):,}", PALETTE[0])
    kpi(c[1], "Open", f"{int((flags.status == 'Open').sum()):,}", PALETTE[3])
    kpi(c[2], "High severity open", f"{int(((flags.status == 'Open') & (flags.severity == 'High')).sum()):,}", PALETTE[7])
    kpi(c[3], "Closed", f"{int((flags.status != 'Open').sum()):,}", PALETTE[2])
    st.write("")
    rc = flags.rule.value_counts().reset_index(); rc.columns = ["rule", "flags"]
    fig = px.bar(rc, x="flags", y="rule", orientation="h", color="rule", color_discrete_sequence=PALETTE,
                 title="Flags by rule", labels={"rule": ""})
    fig.update_layout(showlegend=False)
    st.plotly_chart(style(fig), width="stretch")
    rules = st.multiselect("Rule", sorted(flags.rule.unique()), default=sorted(flags.rule.unique()))
    status = st.radio("Show", ["Open", "Closed", "All"], horizontal=True)
    view = flags[flags.rule.isin(rules)]
    view = view if status == "All" else (view[view.status == "Open"] if status == "Open" else view[view.status != "Open"])
    st.dataframe(view, width="stretch", height=300)
    st.subheader("Close a flag")
    if len(view):
        fid = st.selectbox("Flag", view.flag_id)
        row = flags.set_index("flag_id").loc[fid]
        st.write(f"**{row.rule}** · {row.detail}")
        with st.form("review"):
            decision = st.radio("Decision", ["Justified", "Corrected", "Referred to Accounts"], horizontal=True)
            remarks = st.text_area("Remarks (mandatory)")
            reviewer = st.text_input("Reviewer (designation)")
            if st.form_submit_button("Record decision"):
                if not remarks.strip() or not reviewer.strip():
                    st.error("Remarks and reviewer are mandatory (AI-05).")
                else:
                    L.append_log(LOG_PATH, {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), "flag_id": fid,
                                            "rule": row.rule, "job_no": row.job_no, "ref": row.ref, "decision": decision,
                                            "remarks": remarks, "reviewer": reviewer})
                    st.success("Decision recorded in the append-only decision log."); st.rerun()
    with st.expander("ML flags: unusual shop-wise cost profiles (IsolationForest)"):
        st.dataframe(L.ml_flags(d).head(25), width="stretch")
    with st.expander("Decision log (AI-13)"):
        st.dataframe(log, width="stretch")
    if "anomaly_answer_key" in d:
        with st.expander("Testing only: recall against planted anomalies (do not show to developer)"):
            st.dataframe(L.evaluate_rules(flags, d["anomaly_answer_key"]), width="stretch")

elif page == "Cost prediction (AI)":
    banner("Cost prediction (AI)", "AI-01, AI-02, AI-12 · predicts man-hours and material by shop, then prices them with on-cost at FY rates")
    model = get_model(DATA_DIR)
    met = model["mape"] <= L.MAPE_TARGET
    c = st.columns(4)
    kpi(c[0], f"MAPE on Railway-held test year (target ≤ {L.MAPE_TARGET:.0%})", f"{model['mape']:.1%}", PALETTE[2] if met else PALETTE[7])
    kpi(c[1], "AI-01 acceptance", "Met" if met else "Not yet met", PALETTE[2] if met else PALETTE[7])
    kpi(c[2], "Prediction range (80%)", f"{model['range'][0]:.0%} to {model['range'][1]:.0%}", PALETTE[6])
    kpi(c[3], "Test units", f"{len(model['test'])}", PALETTE[0])
    st.write("")
    tab1, tab2, tab3, tab4 = st.tabs(["Estimate a unit", "Test results", "Monitoring (AI-12)", "What drives cost"])
    with tab1:
        st.write("Enter the as-received inspection. The estimate is advisory: it is stored beside the actual cost and never posts entries.")
        a, b, cc = st.columns(3)
        inp = {"rs_type_code": a.selectbox("Rolling stock type", model["types"], format_func=lambda x: type_names.get(x, x)),
               "age_years": a.number_input("Age (years)", 1, 40, 8),
               "km_since_last_poh": a.number_input("Km since last POH", 50000, 900000, 480000, step=10000),
               "months_since_last_poh": a.number_input("Months since last POH", 6, 48, 24),
               "previous_poh_count": a.number_input("Previous POHs", 0, 15, 2),
               "ioh_count_since_last_poh": a.number_input("IOH since last POH", 0, 3, 1),
               "corrosion_trough_floor": b.slider("Corrosion: trough floor (0-3)", 0, 3, 1),
               "corrosion_sole_bar": b.slider("Corrosion: sole bar (0-3)", 0, 3, 1),
               "corrosion_side_wall": b.slider("Corrosion: side wall (0-3)", 0, 3, 1),
               "corrosion_roof": b.slider("Corrosion: roof (0-3)", 0, 3, 1),
               "missing_damaged_fittings": b.number_input("Missing / damaged fittings", 0, 30, 2),
               "wheel_dia_mm": cc.number_input("Wheel diameter (mm)", 800, 1100, 890),
               "flange_thickness_mm": cc.number_input("Flange thickness (mm)", 20.0, 33.0, 27.5, step=0.1),
               "brake_disc_wear_mm": cc.number_input("Brake disc wear (mm)", 0.0, 10.0, 4.0, step=0.1),
               "air_spring": L.COND_MAP[cc.selectbox("Air spring", list(L.COND_MAP))],
               "damper": L.COND_MAP[cc.selectbox("Dampers", list(L.COND_MAP))],
               "cbc_coupler": L.COND_MAP[cc.selectbox("Coupler", list(L.COND_MAP))],
               "coastal": int(cc.checkbox("Coastal / humid region")),
               "accident": int(cc.checkbox("Accident damage"))}
        price_fy = st.selectbox("Price at rates of FY", fys, index=len(fys) - 1)
        r = L.what_if(d, model, inp, price_fy)
        kpi(st.columns(3)[0], "Expected POH cost", rs(r["total"]), PALETTE[4])
        st.write(f"Likely range: **{rs(r['low'])}** to **{rs(r['high'])}** · expected man-hours **{r['hours']:,.0f}**")
        bs = r["by_shop"]
        left, right = st.columns(2)
        bsm = bs.melt(id_vars="shop", value_vars=L.ELEMENTS, var_name="element", value_name="rs")
        bsm["element"] = bsm.element.map(L.ELEMENT_LABEL)
        fig = px.bar(bsm, x="shop", y="rs", color="element", color_discrete_map=ELEMENT_COLORS,
                     title="Expected cost by shop and element", labels={"shop": "", "rs": "₹"})
        left.plotly_chart(style(fig, "₹"), width="stretch")
        br = pd.DataFrame({"element": [L.ELEMENT_LABEL[e] for e in L.ELEMENTS], "rs": [r[e] for e in L.ELEMENTS]})
        fig = px.pie(br, names="element", values="rs", hole=0.55, color="element", color_discrete_map=ELEMENT_COLORS,
                     title="Expected cost by element")
        right.plotly_chart(style(fig), width="stretch")
        st.dataframe(bs[["shop", "hours", "material_value", "labour_cost", "labour_oncost", "stores_oncost", "shop_cost"]].round(0),
                     width="stretch")
    with tab2:
        te = model["test"].copy(); te["family"] = te.rs_type_code.map(FAMILY)
        fig = px.scatter(te, x="actual", y="predicted", color="family", color_discrete_map=FAMILY_COLORS,
                         title="Test year: predicted vs actual cost per unit", hover_data=["job_no", "rs_type_code", "abs_pct_error"],
                         labels={"actual": "Actual ₹", "predicted": "Predicted ₹"})
        lo, hi = te[["actual", "predicted"]].min().min(), te[["actual", "predicted"]].max().max()
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color=MUTED, dash="dot"))
        fig.update_traces(marker=dict(size=10, line=dict(width=1.5, color=CARD)), selector=dict(mode="markers"))
        st.plotly_chart(style(fig, "Predicted ₹"), width="stretch")
        mt = model["mape_by_type"].reset_index(); mt.columns = ["rs_type_code", "mape"]
        mt["type"] = mt.rs_type_code.map(type_names); mt["family"] = mt.rs_type_code.map(FAMILY)
        fig = px.bar(mt, x="mape", y="type", orientation="h", color="family", color_discrete_map=FAMILY_COLORS,
                     title=f"Error by rolling stock type (target ≤ {L.MAPE_TARGET:.0%})", labels={"mape": "MAPE", "type": ""})
        fig.add_vline(x=L.MAPE_TARGET, line_dash="dot", line_color="#ffffff")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style(fig), width="stretch")
    with tab3:
        mo = model["monthly"]
        fig = px.line(mo, x="month", y="abs_pct_error", markers=True, color_discrete_sequence=[PALETTE[4]],
                      title="Monthly prediction error on dispatched units", labels={"month": "Month", "abs_pct_error": "MAPE"})
        fig.add_hline(y=L.MAPE_TARGET, line_dash="dot", line_color=PALETTE[2], annotation_text="Target 5%")
        fig.add_hline(y=L.MAPE_ALERT, line_dash="dot", line_color=PALETTE[7], annotation_text="Alert 15%")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig, "MAPE"), width="stretch")
        bad = (mo.abs_pct_error > L.MAPE_ALERT).astype(int)
        two_in_row = bool(((bad + bad.shift(1, fill_value=0)) == 2).any())
        (st.error if two_in_row else st.success)("AI-12 drift alert: MAPE above 15% for two consecutive months" if two_in_row
                                                 else "AI-12: no drift alert (MAPE has not exceeded 15% for two consecutive months)")
    with tab4:
        imp = model["importance"].head(12).reset_index(); imp.columns = ["factor", "importance"]
        fig = px.bar(imp.sort_values("importance"), x="importance", y="factor", orientation="h", color="importance",
                     color_continuous_scale=["#1a2a4a", "#3987e5", "#9085e9", "#d55181"],
                     title="Factors that most influence predicted cost (AI-02)")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style(fig), width="stretch")

elif page == "Costing assistant":
    banner("Costing assistant", "AI-08 · chat in English or Hindi; figures are always computed from the costing data, with the working shown")
    ASK_LOG = os.environ.get("AIWCS_ASK_LOG", os.path.join(HERE, "assistant_log.csv"))
    with st.sidebar.expander("AI model (Claude)", expanded=True):
        llm_model = st.text_input("Model", A.CLAUDE_MODEL)
        problem = claude_status(llm_model)
        if problem:
            st.caption(problem + " The assistant still answers data questions with its built-in reader.")
            use_llm = False
        else:
            use_llm = st.checkbox("Use Claude", value=True)
    mode = st.radio("View", ["Chat", "Acceptance test (AI-08)", "Variance notes & documents"], horizontal=True)
    if mode == "Chat":
        if "chat" not in st.session_state:
            st.session_state["chat"] = []
        chat = st.session_state["chat"]
        c1, c2 = st.columns([4, 1])
        c1.caption(f"Answering with: {'Claude ' + llm_model if use_llm else 'built-in reader (no AI model)'}")
        if c2.button("Clear chat"):
            chat.clear()
        EXAMPLES = ["Average POH cost of LHB AC coaches in 2024-25", "Which shop has the highest labour on-cost in 2024-25?",
                    "Which rolling stock type is costliest?", "How many units cost more than 25 lakh in 2025-26?",
                    "Monthly man-hours in Paint shop 2024-25", "What is stores on-cost?",
                    "Total approved idle hours in Bogie shop 2024-25", "LHB AC कोच की औसत लागत 2024-25"]
        picked = None
        if not chat:
            st.caption("Try an example:")
            ec = st.columns(4)
            for k, ex in enumerate(EXAMPLES):
                if ec[k % 4].button(ex, key=f"ex{k}", width="stretch"):
                    picked = ex

        def show_result(res, n):
            tab = res["table"]
            if tab is not None and len(tab) > 1:
                gcol, vcol = tab.columns[0], tab.columns[1]
                chart = tab.sort_values(gcol) if gcol in ("month", "fy") else tab
                fig = (px.line(chart, x=gcol, y=vcol, markers=True, color_discrete_sequence=[PALETTE[n % 8]]) if gcol == "month"
                       else px.bar(chart, x=gcol, y=vcol, color=gcol if gcol == "shop" else None, color_discrete_map=SHOP_COLORS,
                                   color_discrete_sequence=[PALETTE[n % 8]]))
                fig.update_layout(showlegend=False, title=vcol.capitalize())
                st.plotly_chart(style(fig), width="stretch", key=f"chart{n}")
            with st.expander("How this was computed (from the data, not by the AI)"):
                st.markdown(res["answer"])
                st.write(f"Question understood by: {res['understood_by']}")
                st.json(res["query"])
                if tab is not None:
                    st.dataframe(tab, width="stretch")
                st.dataframe(res["rows"], width="stretch", height=200)

        for n, msg in enumerate(chat):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("res") is not None:
                    show_result(msg["res"], n)

        question = st.chat_input("Ask about costs, on-cost, man-hours, material, idle time ...") or picked
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    res, reply = A.converse(question, d, chat, use_llm, llm_model)
                    if isinstance(reply, str):
                        st.markdown(reply); text = reply
                    elif hasattr(st, "write_stream"):
                        text = st.write_stream(reply)
                    else:
                        box, text = st.empty(), ""
                        for part in reply:
                            text += part; box.markdown(text)
                except Exception as e:
                    res = A.ask(question, d)
                    text = res["answer"] + f"\n\n_(Claude error: {type(e).__name__}; answered with the built-in reader.)_"
                    st.markdown(text)
                if res is not None:
                    show_result(res, len(chat) + 1)
            chat.append({"role": "user", "content": question})
            chat.append({"role": "assistant", "content": text, "res": res})
            pd.DataFrame([[dt.datetime.now().isoformat(timespec="seconds"), question, llm_model if use_llm else "built-in",
                           str(res["query"]) if res is not None else "general (no data query)", A.re.sub(r"[*]", "", str(text))]],
                         columns=["timestamp", "question", "answered_with", "query", "answer"]).to_csv(
                ASK_LOG, mode="a", header=not os.path.exists(ASK_LOG), index=False)
    elif mode == "Acceptance test (AI-08)":
        st.write("Runs every question in the test set through the assistant's data engine and compares with the expected answer (AI-08 needs 20 to match).")
        if st.button("Run acceptance test"):
            res = A.acceptance_test(d)
            ok = int((res.result == "PASS").sum())
            kpi(st.columns(3)[0], "Questions answered correctly", f"{ok} of {len(res)}", PALETTE[2] if ok >= 20 else PALETTE[7])
            st.write("")
            st.dataframe(res, width="stretch", height=420)
        else:
            st.dataframe(d["ai_test_questions"], width="stretch", height=380)
    else:
        for r in d["variance_note_samples"].itertuples():
            st.markdown(f"**{r.job_no}** · {r.variance_note}")
        st.subheader("Document library index (AI-09)")
        st.dataframe(d["document_library_index"], width="stretch")

# ==================================================================
# FLOATING AI CHATBOT (every page except "Costing assistant", which has the full chat)
# ==================================================================
QUICK_QUESTIONS = ["Average POH cost of LHB AC coaches in 2024-25", "Which shop has the highest labour on-cost in 2024-25?",
                   "Monthly man-hours in Paint shop 2024-25", "What is stores on-cost?"]


def log_ask(question, used, res, text):
    """AI-13: every question and answer goes to the append-only assistant log."""
    path = os.environ.get("AIWCS_ASK_LOG", os.path.join(HERE, "assistant_log.csv"))
    pd.DataFrame([[dt.datetime.now().isoformat(timespec="seconds"), question, used,
                   str(res["query"]) if res is not None else "general (no data query)", A.re.sub(r"[*]", "", str(text))]],
                 columns=["timestamp", "question", "answered_with", "query", "answer"]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False)


def float_result(res, n):
    """Compact working under an answer: mini chart for grouped answers, then how it was computed."""
    tab = res["table"]
    if tab is not None and len(tab) > 1:
        gcol, vcol = tab.columns[0], tab.columns[1]
        chart = tab.sort_values(gcol) if gcol in ("month", "fy") else tab
        fig = (px.line(chart, x=gcol, y=vcol, markers=True, color_discrete_sequence=[PALETTE[n % 8]]) if gcol == "month"
               else px.bar(chart.iloc[::-1], x=vcol, y=gcol, orientation="h", color=gcol if gcol == "shop" else None,
                           color_discrete_map=SHOP_COLORS, color_discrete_sequence=[PALETTE[n % 8]]))
        height = 220 if gcol == "month" else 40 + 26 * len(chart)
        style(fig).update_layout(showlegend=False, height=height, margin=dict(l=4, r=4, t=6, b=4), font_size=11)
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
        st.plotly_chart(fig, width="stretch", key=f"fchart{n}", config={"displayModeBar": False})
    with st.expander("How this was computed"):
        st.caption(f"Understood by: {res['understood_by']} · numbers computed by pandas, not by the AI")
        st.json(res["query"], expanded=False)
        if tab is not None:
            st.dataframe(tab, width="stretch", hide_index=True)


if page != "Costing assistant":
    fchat = st.session_state.setdefault("float_chat", [])
    claude_ok = claude_status(A.CLAUDE_MODEL) is None
    with st.container(key="float_ask"):
        with st.popover("💬"):
            use_claude = claude_ok and st.session_state.get("float_use_claude", True)
            engine = f"Claude · {A.CLAUDE_MODEL}" if use_claude else "Built-in reader (no AI model)"
            dot = "#2fa84f" if use_claude else "#c98500"
            st.markdown(f'<div class="fc-head"><div class="t">AI Costing Assistant</div>'
                        f'<div class="s"><span class="dot" style="background:{dot}"></span>'
                        f'{engine} · figures computed from the costing data</div></div>', unsafe_allow_html=True)
            t1, t2 = st.columns([3, 1], vertical_alignment="center")
            t1.toggle("Use Claude", value=claude_ok, disabled=not claude_ok, key="float_use_claude",
                      help="Claude reads the question and writes the reply. Without an API key the built-in reader answers.")
            if t2.button("Clear", key="float_clear", width="stretch", disabled=not fchat):
                fchat.clear()
                st.rerun()

            box = st.container(height=420, border=False, key="float_box")
            picked = None
            with box:
                if not fchat:
                    st.markdown('<div class="fc-hello">Hi! Ask me about POH costs, on-cost, man-hours, material or idle time, '
                                'in English or Hindi. Try one of these:</div>', unsafe_allow_html=True)
                    for i, q in enumerate(QUICK_QUESTIONS):
                        if st.button(q, key=f"fq{i}", width="stretch"):
                            picked = q
                for n, msg in enumerate(fchat):
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        if msg.get("res") is not None:
                            float_result(msg["res"], n)

            with st.container(key="float_form"):
                with st.form("floating_chat_form", clear_on_submit=True, border=False):
                    c1, c2 = st.columns([5, 1], vertical_alignment="center")
                    new_q = c1.text_input("Ask", label_visibility="collapsed", placeholder="Ask about costs, on-cost, man-hours ...")
                    sent = c2.form_submit_button("➤", type="primary", width="stretch")

            question = picked or (new_q.strip() if sent and new_q.strip() else None)
            if question:
                with box:
                    with st.chat_message("user"):
                        st.markdown(question)
                    with st.chat_message("assistant"):
                        try:
                            res, reply = A.converse(question, d, fchat, use_claude, A.CLAUDE_MODEL)
                            if isinstance(reply, str):
                                st.markdown(reply)
                                text = reply
                            else:
                                text = st.write_stream(reply)
                        except Exception as e:
                            res = A.ask(question, d)
                            text = res["answer"] + f"\n\n_(Claude error: {type(e).__name__}; answered with the built-in reader.)_"
                            st.markdown(text)
                fchat += [{"role": "user", "content": question}, {"role": "assistant", "content": text, "res": res}]
                log_ask(question, A.CLAUDE_MODEL if use_claude else "built-in", res, text)
                st.rerun()
