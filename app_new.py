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
import impact as I

st.set_page_config(page_title="AI Assisted Workshop Costing", layout="wide", initial_sidebar_state="collapsed")

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

/* command centre: hero, headline tiles, feature tiles */
.kpi {{ transition: transform .15s ease, border-color .15s ease; }}
.kpi:hover {{ transform: translateY(-2px); border-color: #3a3a5c; }}
.hero {{ position: relative; overflow: hidden; border-radius: 18px; padding: 26px 30px; margin-bottom: 18px;
        background: radial-gradient(circle at 85% 20%, rgba(213,81,129,.35) 0%, rgba(0,0,0,0) 45%),
                    radial-gradient(circle at 10% 90%, rgba(57,135,229,.35) 0%, rgba(0,0,0,0) 50%), #0c0c16;
        border: 1px solid #23233a; }}
.hero .eyebrow {{ color: #d4af37 !important; letter-spacing: .18em; font-size: .78rem; font-weight: 700; }}
.hero h1 {{ color: #fff !important; font-size: 2.3rem; margin: 4px 0 2px 0; line-height: 1.15; }}
.hero .sub {{ color: {MUTED} !important; font-size: 1rem; }}
.hero .chips {{ margin-top: 12px; }}
.chip {{ display: inline-block; padding: 3px 10px; margin: 0 6px 6px 0; border-radius: 999px; font-size: .78rem;
        background: #161624; border: 1px solid #2c2c48; color: {INK} !important; }}
.chip .dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
.big {{ background: {CARD}; border: 1px solid #23233a; border-radius: 16px; padding: 18px 20px; height: 100%; position: relative; overflow: hidden; }}
.big::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--c); }}
.big .lab {{ color: {MUTED} !important; font-size: .85rem; text-transform: uppercase; letter-spacing: .06em; }}
.big .val {{ color: #fff !important; font-size: 2.1rem; font-weight: 800; margin: 4px 0 2px 0; line-height: 1.1; }}
.big .note {{ color: {MUTED} !important; font-size: .82rem; }}
.sect {{ color: #fff !important; font-size: 1.15rem; font-weight: 700; margin: 22px 0 10px 0; }}
.sect span {{ color: {MUTED} !important; font-weight: 400; font-size: .9rem; margin-left: 8px; }}
.ms {{ font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1; letter-spacing: normal;
      text-transform: none; white-space: nowrap; direction: ltr; font-feature-settings: 'liga'; }}
/* feature tile = one card; the whole card is the button; it enlarges on hover */
div[class*="st-key-tile_"] {{ position: relative; border-radius: 18px; z-index: 1;
    transition: transform .28s cubic-bezier(.2,.8,.2,1), z-index 0s .28s; }}
div[class*="st-key-tile_"]:hover {{ transform: scale(1.09) translateY(-4px); z-index: 60; transition: transform .28s cubic-bezier(.2,.8,.2,1); }}
.tile {{ position: relative; overflow: hidden; background: linear-gradient(160deg, #161628 0%, {CARD} 65%); border: 1px solid #23233a;
        border-radius: 18px; padding: 18px 20px 16px 20px; min-height: 268px; transition: border-color .28s, box-shadow .28s, background .28s; }}
.tile::before {{ content: ""; position: absolute; inset: 0 0 auto 0; height: 3px; background: var(--c); opacity: .85; }}
.tile::after {{ content: ""; position: absolute; right: -50px; top: -50px; width: 150px; height: 150px; border-radius: 50%;
               background: radial-gradient(circle, var(--c) 0%, transparent 70%); opacity: .14; transition: opacity .28s, transform .5s; }}
div[class*="st-key-tile_"]:hover .tile {{ border-color: var(--c); background: linear-gradient(160deg, #1b1b33 0%, #111120 65%);
    box-shadow: 0 22px 60px rgba(0,0,0,.85), 0 0 34px color-mix(in srgb, var(--c) 40%, transparent); }}
div[class*="st-key-tile_"]:hover .tile::after {{ opacity: .38; transform: scale(1.5); }}
.tile .top {{ display: flex; align-items: center; justify-content: space-between; }}
.tile .ico {{ width: 46px; height: 46px; border-radius: 14px; display: flex; align-items: center; justify-content: center;
             background: color-mix(in srgb, var(--c) 22%, transparent); border: 1px solid color-mix(in srgb, var(--c) 55%, transparent);
             transition: transform .3s, box-shadow .3s; }}
.tile .ico .ms {{ font-size: 26px; color: var(--c) !important; }}
div[class*="st-key-tile_"]:hover .ico {{ transform: rotate(-6deg) scale(1.12); box-shadow: 0 0 22px var(--c); }}
.tile .st {{ font-size: .68rem; font-weight: 800; letter-spacing: .06em; padding: 3px 10px; border-radius: 999px;
            border: 1px solid var(--s); color: var(--s) !important; background: color-mix(in srgb, var(--s) 12%, transparent); }}
.tile .st::before {{ content: ""; display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--s);
                    margin-right: 6px; vertical-align: middle; animation: pulse 1.8s ease-in-out infinite; }}
@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: .25; }} }}
.tile .name {{ color: {MUTED} !important; font-weight: 700; font-size: .8rem; margin-top: 14px; text-transform: uppercase; letter-spacing: .08em; }}
.tile .val {{ color: #fff !important; font-size: 1.85rem; font-weight: 800; line-height: 1.15; margin-top: 3px; }}
.tile .what {{ color: {MUTED} !important; font-size: .8rem; margin-top: 5px; min-height: 2.4em; }}
.tile .meter {{ margin-top: 12px; }}
.tile .meter .track {{ height: 6px; border-radius: 6px; background: #23233a; overflow: hidden; }}
.tile .meter .fill {{ height: 100%; border-radius: 6px; background: var(--c); width: var(--w); transition: width .6s; }}
.tile .meter .cap {{ display: flex; justify-content: space-between; font-size: .72rem; margin-top: 5px; }}
.tile .meter .cap span {{ color: {MUTED} !important; }} .tile .meter .cap b {{ color: {INK} !important; }}
.tile .open {{ margin-top: 12px; font-size: .8rem; font-weight: 700; color: var(--c) !important; opacity: .75; transition: opacity .28s, letter-spacing .28s; }}
div[class*="st-key-tile_"]:hover .open {{ opacity: 1; letter-spacing: .04em; }}
div[class*="st-key-go_"] {{ position: absolute !important; inset: 0; width: 100% !important; height: 100% !important; z-index: 5; }}
div[class*="st-key-go_"] > div, div[class*="st-key-go_"] .stButton {{ width: 100% !important; height: 100% !important; }}
div[class*="st-key-go_"] button {{ width: 100% !important; height: 100% !important; opacity: 0; cursor: pointer; }}
div[class*="st-key-tile_"] {{ margin-bottom: 14px; }}

/* count-up numbers (pure CSS: animated integer custom properties) */
@property --n {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
@keyframes cu {{ from {{ --n: 0; }} }}
.cn {{ --n: var(--to); counter-reset: n var(--n); animation: cu 1.6s cubic-bezier(.15,.85,.25,1) both; }}
.cn::after {{ content: counter(n); }}
.cn.d2::after {{ content: counter(n, decimal-leading-zero); }}

/* sparklines in the headline cards */
.big .spark {{ margin-top: 10px; height: 46px; }}
.big .spark svg {{ width: 100%; height: 46px; display: block; overflow: visible; }}
.big .spark .cap {{ color: {MUTED} !important; font-size: .7rem; margin-top: 2px; }}
.big {{ transition: transform .2s ease, border-color .2s ease; }}
.big:hover {{ transform: translateY(-3px); border-color: #3a3a5c; }}

/* AI alert of the day */
.st-key-alertbox {{ position: relative; margin: 2px 0 14px 0; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 4.2rem !important; }}
.seg {{ display: flex; height: 14px; border-radius: 7px; overflow: hidden; margin-top: 16px; gap: 2px; }}
.seg div {{ height: 100%; }}
.segl {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; font-size: .74rem; }}
.segl span {{ color: {MUTED} !important; }} .segl b {{ color: {INK} !important; }}
.segl i {{ display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 5px; }}
.alert {{ display: flex; align-items: center; gap: 16px; padding: 14px 20px; border-radius: 14px;
         background: linear-gradient(90deg, rgba(230,103,103,.18) 0%, rgba(230,103,103,.04) 60%, {CARD} 100%);
         border: 1px solid rgba(230,103,103,.45); border-left: 5px solid #e66767; transition: box-shadow .25s, transform .25s; }}
.alert.ok {{ background: linear-gradient(90deg, rgba(47,168,79,.16) 0%, {CARD} 70%); border-color: rgba(47,168,79,.45); border-left-color: #2fa84f; }}
.st-key-alertbox:hover .alert {{ box-shadow: 0 0 30px rgba(230,103,103,.35); transform: translateY(-2px); }}
.alert .ring {{ position: relative; width: 44px; height: 44px; flex: none; border-radius: 50%; background: rgba(230,103,103,.2);
               display: flex; align-items: center; justify-content: center; }}
.alert.ok .ring {{ background: rgba(47,168,79,.2); }}
.alert .ring::after {{ content: ""; position: absolute; inset: 0; border-radius: 50%; border: 2px solid #e66767; animation: ping 1.8s ease-out infinite; }}
.alert.ok .ring::after {{ border-color: #2fa84f; }}
@keyframes ping {{ 0% {{ transform: scale(1); opacity: .9; }} 100% {{ transform: scale(1.7); opacity: 0; }} }}
.alert .ring .ms {{ font-size: 24px; color: #ff8a8a !important; }}
.alert.ok .ring .ms {{ color: #5fd47d !important; }}
.alert .tx {{ flex: 1; }}
.alert .tx .h {{ font-size: .72rem; font-weight: 800; letter-spacing: .12em; color: #ff8a8a !important; }}
.alert.ok .tx .h {{ color: #5fd47d !important; }}
.alert .tx .m {{ font-size: 1.02rem; color: #fff !important; margin-top: 2px; }}
.alert .tx .m b {{ color: #fff !important; }}
.alert .tx .m .hl {{ color: #ffb4b4 !important; font-weight: 800; }}
.alert .go {{ font-weight: 700; font-size: .85rem; color: #ff8a8a !important; white-space: nowrap; }}
.alert.ok .go {{ color: #5fd47d !important; }}

/* AI impact: scoreboard */
.imp {{ position: relative; border-radius: 16px; padding: 16px 18px; height: 100%; border: 1px solid transparent;
       background: linear-gradient({CARD}, {CARD}) padding-box,
                   linear-gradient(135deg, #3987e5, #9085e9 45%, #d55181 80%, #d95926) border-box; transition: transform .2s; }}
.imp:hover {{ transform: translateY(-3px); }}
.imp .k {{ color: {MUTED} !important; font-size: .72rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }}
.imp .v {{ font-size: 1.9rem; font-weight: 800; line-height: 1.15; margin-top: 6px;
          background: linear-gradient(90deg, #ffffff 0%, #c9c3ff 100%); -webkit-background-clip: text; background-clip: text;
          -webkit-text-fill-color: transparent; }}
.imp .v small {{ font-size: 1rem; -webkit-text-fill-color: {MUTED}; }}
.imp .n {{ color: {MUTED} !important; font-size: .78rem; margin-top: 4px; }}
.imp .was {{ display: inline-block; margin-top: 8px; font-size: .72rem; padding: 2px 8px; border-radius: 6px;
            background: rgba(230,103,103,.12); color: #ff9c9c !important; text-decoration: line-through; text-decoration-color: rgba(255,156,156,.6); }}
.impnote {{ color: {MUTED} !important; font-size: .76rem; margin: 8px 0 4px 2px; }}

/* Without AI vs With AI lanes */
.lanes {{ background: {CARD}; border: 1px solid #23233a; border-radius: 16px; padding: 18px 22px 8px 22px; }}
.lanes .ttl {{ color: #fff !important; font-weight: 800; font-size: 1.02rem; }}
.lanes .sub {{ color: {MUTED} !important; font-size: .8rem; margin: 2px 0 10px 0; }}
.lane {{ display: flex; align-items: center; gap: 14px; margin: 44px 0 40px 0; }}
.lane .ln {{ width: 92px; flex: none; font-size: .72rem; font-weight: 800; letter-spacing: .1em; color: {MUTED} !important; }}
.lane.ai .ln {{ color: #c9c3ff !important; }}
.lane .tk {{ position: relative; flex: 1; height: 10px; border-radius: 6px; background: #1a1a2a; }}
.lane .sg {{ position: absolute; top: 0; height: 100%; border-radius: 6px; transform-origin: left; animation: grow 1.4s cubic-bezier(.2,.8,.2,1) both; }}
.lane .sg.grey {{ background: #4a4a60; }}
.lane .sg.dash {{ background: repeating-linear-gradient(90deg, #e66767 0 6px, transparent 6px 11px); opacity: .8; animation-delay: .9s; }}
.lane .sg.ai {{ background: linear-gradient(90deg, #3987e5, #9085e9 60%, #d55181); }}
@keyframes grow {{ from {{ transform: scaleX(0); }} }}
.lane .mk {{ position: absolute; top: 50%; width: 16px; height: 16px; margin: -8px 0 0 -8px; border-radius: 50%;
            background: #0b0b14; border: 3px solid #a8a8b8; z-index: 2; animation: pop .4s ease both; }}
.lane .mk.blue {{ border-color: #3987e5; }} .lane .mk.red {{ border-color: #e66767; background: #e66767; }}
.lane .mk.pink {{ border-color: #d55181; background: #d55181; box-shadow: 0 0 0 0 rgba(213,81,129,.7); animation: pop .4s ease both, halo 1.6s ease-out 1.4s infinite; }}
@keyframes pop {{ from {{ transform: scale(0); }} }}
@keyframes halo {{ 0% {{ box-shadow: 0 0 0 0 rgba(213,81,129,.7); }} 100% {{ box-shadow: 0 0 0 14px rgba(213,81,129,0); }} }}
.lane .lb {{ position: absolute; top: 16px; transform: translateX(-50%); white-space: nowrap; font-size: .72rem; text-align: center;
            color: {MUTED} !important; line-height: 1.25; }}
.lane .lb.up {{ top: auto; bottom: 16px; }}
.lane .lb b {{ color: {INK} !important; }} .lane .lb.red b {{ color: #ff8a8a !important; }} .lane .lb.pink b {{ color: #ff9ccb !important; }}
.lane .lb.l {{ transform: none; }} .lane .lb.r {{ transform: translateX(-100%); text-align: right; }}
.gain {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 0 0 12px 0; }}
.gain span {{ font-size: .8rem; padding: 5px 12px; border-radius: 999px; background: rgba(144,133,233,.14); border: 1px solid rgba(144,133,233,.4); }}
.gain span b {{ color: #fff !important; }}

/* why the AI warned */
.why {{ background: {CARD}; border: 1px solid #23233a; border-radius: 16px; padding: 18px 20px; height: 100%; }}
.why .ttl {{ color: #fff !important; font-weight: 800; font-size: 1.02rem; display: flex; align-items: center; gap: 8px; }}
.why .ttl .ms {{ color: #d55181 !important; font-size: 22px; }}
.why .sub {{ color: {MUTED} !important; font-size: .8rem; margin: 2px 0 12px 0; }}
.why ol {{ list-style: none; padding: 0; margin: 0; counter-reset: r; }}
.why li {{ position: relative; padding: 10px 12px 10px 46px; margin-bottom: 8px; border-radius: 12px; background: #151524;
          border: 1px solid #23233a; font-size: .86rem; line-height: 1.45; animation: slide .5s ease both; }}
.why li:nth-child(2) {{ animation-delay: .15s; }} .why li:nth-child(3) {{ animation-delay: .3s; }}
@keyframes slide {{ from {{ opacity: 0; transform: translateX(14px); }} }}
.why li::before {{ counter-increment: r; content: counter(r); position: absolute; left: 12px; top: 10px; width: 24px; height: 24px;
                  border-radius: 50%; background: linear-gradient(135deg, #d55181, #d95926); color: #fff; font-weight: 800;
                  font-size: .8rem; display: flex; align-items: center; justify-content: center; }}
.why li b {{ color: #fff !important; }}
.why .act {{ margin-top: 10px; font-size: .78rem; color: {MUTED} !important; border-top: 1px dashed #2c2c48; padding-top: 10px; }}
.why .act b {{ color: #5fd47d !important; }}

/* top navigation + filter button */
header[data-testid="stHeader"] {{ background: rgba(0,0,0,.85) !important; backdrop-filter: blur(8px);
    border-bottom: 1px solid #23233a; }}
.st-key-topbar {{ margin-bottom: 4px; }}
.st-key-topbar .brand {{ display: flex; align-items: center; gap: 12px; }}
.st-key-topbar .brand .mono {{ width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
    font-weight: 800; color: #fff !important; background: linear-gradient(135deg, #3987e5 0%, #9085e9 50%, #d55181 100%); }}
.st-key-topbar .brand .t {{ font-weight: 800; color: #fff !important; font-size: 1.02rem; line-height: 1.2; }}
.st-key-topbar .brand .s {{ color: {MUTED} !important; font-size: .74rem; }}
.foot {{ color: {MUTED} !important; font-size: .76rem; text-align: center; margin: 34px 0 10px 0; padding-top: 12px; border-top: 1px solid #23233a; }}
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


@st.cache_data
def get_vs_standard(path):
    """Per job: man-hours and material against the approved standard cost card (all shops)."""
    d = get_data(path)
    s = d["target_shop_cost"].merge(d["poh_jobs"][["job_no", "rs_type_code", "fy"]], on="job_no")
    s = s.merge(d["standard_cost_card"][["rs_type_code", "fy", "shop_no", "allowed_hours", "std_material_value"]],
                on=["rs_type_code", "fy", "shop_no"], how="left")
    return s.groupby("job_no")[["man_hours", "allowed_hours", "material_value", "std_material_value"]].sum()


@st.cache_data(show_spinner="Replaying the test year with and without AI ...")
def get_history(path):
    return I.warning_history(get_data(path), get_ledger(path), get_model(path))


@st.cache_data
def get_ml_only(path):
    return I.ml_only(get_data(path), get_flags(path, 1.35))


@st.cache_data
def get_reasons(path, job_no, as_on):
    return I.reasons(get_data(path), get_ledger(path), get_model(path), job_no, as_on)


def lanes_html(u, est, curve_total):
    """Two timelines for one unit: when the overrun becomes known without AI, and with AI."""
    t0, t1 = u.received, u.known_without_ai + pd.Timedelta(days=2)
    span = (t1 - t0).days or 1
    pos = lambda t: (t - t0).days / span * 100
    r, w, dp, kn = pos(u.received), pos(u.warned), pos(u.dispatched), pos(u.known_without_ai)
    day_w = int(u.warn_day)
    return f"""<div class="lanes"><div class="ttl">Without AI vs With AI · {u.job_no}</div>
<div class="sub">{type_names.get(u.rs_type_code, u.rs_type_code)} · AI estimate at receipt {rs(est)} · final cost {rs(u.actual)}
 (<b>{u.actual / est - 1:+.0%}</b>)</div>
<div class="lane"><div class="ln">WITHOUT AI</div><div class="tk">
 <div class="sg grey" style="left:{r:.1f}%;width:{dp - r:.1f}%"></div><div class="sg dash" style="left:{dp:.1f}%;width:{kn - dp:.1f}%"></div>
 <div class="mk" style="left:{r:.1f}%"></div><div class="lb l" style="left:0%"><b>Received</b><br>{u.received:%d %b}</div>
 <div class="mk" style="left:{dp:.1f}%"></div><div class="lb up" style="left:{dp:.1f}%"><b>Dispatched</b><br>{u.dispatched:%d %b}</div>
 <div class="mk red" style="left:{kn:.1f}%"></div><div class="lb r red" style="left:{kn:.1f}%"><b>Cost known · overrun found</b><br>{u.known_without_ai:%d %b %Y}</div>
</div></div>
<div class="lane ai"><div class="ln">WITH AI</div><div class="tk">
 <div class="sg ai" style="left:{r:.1f}%;width:{dp - r:.1f}%"></div>
 <div class="mk blue" style="left:{r:.1f}%"></div><div class="lb l" style="left:0%"><b>AI estimate</b><br>day 1 · {rs(est)}</div>
 <div class="mk pink" style="left:{w:.1f}%"></div><div class="lb up pink" style="left:{w:.1f}%"><b>AI warning · day {day_w}</b><br>{u.warned:%d %b} · estimate passed</div>
 <div class="mk" style="left:{dp:.1f}%"></div><div class="lb" style="left:{dp:.1f}%"><b>Dispatched</b><br>{u.dispatched:%d %b}</div>
</div></div>
<div class="gain"><span>Overrun seen <b>{int(u.earlier_days)} days earlier</b></span><span><b>{int(u.lead_days)} days</b> before the coach leaves</span>
<span>Officer can act while it is <b>still on the floor</b></span></div></div>"""


STATUS_COL = {"good": "#2fa84f", "warn": "#c98500", "bad": "#e66767", "info": "#9085e9"}


def countup(x, decimals=0, prefix="", suffix=""):
    """A number that counts up from zero when the card appears (CSS only)."""
    if decimals:
        whole, frac = divmod(round(x * 100), 100)
        num = f'<span class="cn" style="--to:{whole}"></span>.<span class="cn d2" style="--to:{frac}"></span>'
    else:
        num = f'<span class="cn" style="--to:{round(x)}"></span>'
    return f"{prefix}{num}{suffix}"


def spark(values, color, caption=""):
    """Inline SVG trend line with a soft area fill."""
    v = [float(x) for x in values]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1
    pts = [(i / (len(v) - 1) * 100, 40 - (x - lo) / span * 34) for i, x in enumerate(v)]
    line = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    gid = "g" + color.strip("#")
    return (f'<div class="spark"><svg viewBox="0 0 100 44" preserveAspectRatio="none">'
            f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{color}" stop-opacity=".45"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="0,44 {line} 100,44" fill="url(#{gid})"/>'
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>'
            f'</svg><div class="cap">{caption}</div></div>')


def spark_bars(values, color, caption=""):
    """Inline SVG mini bars (largest first)."""
    v = [float(x) for x in values][:8]
    if not v:
        return ""
    hi = max(v) or 1
    w = 100 / len(v)
    bars = "".join(f'<rect x="{i * w + w * .15:.2f}" y="{44 - x / hi * 40:.2f}" width="{w * .7:.2f}" height="{x / hi * 40:.2f}" rx="1" fill="{color}" '
                   f'opacity="{1 - i * .08:.2f}"/>' for i, x in enumerate(v))
    return f'<div class="spark"><svg viewBox="0 0 100 44" preserveAspectRatio="none">{bars}</svg><div class="cap">{caption}</div></div>'


def seg_bar(counts, colors):
    """Segmented bar + legend (e.g. open flags by severity)."""
    tot = counts.sum() or 1
    order = [k for k in colors if k in counts.index]
    bar = "".join(f'<div style="flex:{counts[k] / tot:.4f};background:{colors[k]}"></div>' for k in order)
    leg = "".join(f'<span><i style="background:{colors[k]}"></i>{k} <b>{int(counts[k]):,}</b></span>' for k in order)
    return f'<div class="seg">{bar}</div><div class="segl">{leg}</div>'


def big(col, label, value, note, color, extra=""):
    col.markdown(f'<div class="big" style="--c:{color}"><div class="lab">{label}</div><div class="val">{value}</div>'
                 f'<div class="note">{note}</div>{extra}</div>', unsafe_allow_html=True)


def go_to(target):
    st.session_state["_goto"] = target


def tile(col, target, icon, value, what, status, color, meter=None):
    """A feature tile: the whole card opens the page; it enlarges on hover. meter = (share 0..1, caption, value text)."""
    label, kind = status
    bar = ""
    if meter:
        share, cap, txt = meter
        bar = (f'<div class="meter"><div class="track"><div class="fill" style="--w:{max(0, min(share, 1)) * 100:.0f}%"></div></div>'
               f'<div class="cap"><span>{cap}</span><b>{txt}</b></div></div>')
    n = PAGES.index(target)
    with col.container(key=f"tile_{n}"):
        st.markdown(f'<div class="tile" style="--c:{color};--s:{STATUS_COL[kind]}"><div class="top"><div class="ico"><span class="ms">{icon}</span></div>'
                    f'<div class="st">{label}</div></div><div class="name">{target}</div><div class="val">{value}</div>'
                    f'<div class="what">{what}</div>{bar}<div class="open">Open {target} &rarr;</div></div>',
                    unsafe_allow_html=True)
        st.button(f"Open {target}", key=f"go_{n}", on_click=go_to, args=(target,))


try:
    d = get_data(DATA_DIR)
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

# ------------------------------------------------------------------ navigation (top bar) and filters
PAGES = ["Overview", "Work in progress", "Costing assistant", "Unit cost & variance", "On-cost", "Labour reconciliation",
         "Material", "Anomaly review", "Cost prediction (AI)"]
NAV_ICON = {"Overview": "dashboard", "Work in progress": "factory", "Costing assistant": "forum", "Unit cost & variance": "query_stats",
            "On-cost": "calculate", "Labour reconciliation": "engineering", "Material": "inventory_2", "Anomaly review": "report",
            "Cost prediction (AI)": "model_training"}


def _page_fn(slug):
    def f():
        pass
    f.__name__ = slug
    return f


PAGE_OBJ = {t: st.Page(_page_fn("p_" + str(i)), title=t, icon=f":material/{NAV_ICON[t]}:", default=(i == 0),
                       url_path=None if i == 0 else t.lower().replace(" & ", "-").replace(" (ai)", "").replace(" ", "-"))
            for i, t in enumerate(PAGES)}
page = st.navigation(list(PAGE_OBJ.values()), position="sidebar", expanded=True).title
if st.session_state.get("_goto"):
    st.switch_page(PAGE_OBJ[st.session_state.pop("_goto")])

fys = sorted(d["poh_jobs"].fy.unique())
type_names = d["coach_master"].drop_duplicates("rs_type_code").set_index("rs_type_code").rs_type.to_dict()
types = sorted(d["poh_jobs"].rs_type_code.unique())
with st.container(key="topbar"):
    tb1, tb2 = st.columns([5, 1], vertical_alignment="center")
    tb1.markdown('<div class="brand"><div class="mono">AI</div><div><div class="t">AI Assisted Workshop Costing</div>'
                 '<div class="s">Rolling stock POH costing · BRD v0.4 · DUMMY DATA</div></div></div>', unsafe_allow_html=True)
    n_fy, n_ty = len(st.session_state.get("f_fy", fys)), len(st.session_state.get("f_type", types))
    with tb2.popover(f"Filters · {n_fy} FY · {n_ty} types", icon=":material/tune:", width="stretch"):
        sel_fy = st.multiselect("Financial year", fys, default=fys, key="f_fy")
        sel_type = st.multiselect("Rolling stock type", types, default=types, format_func=lambda c: type_names.get(c, c), key="f_type")

T = L.coach_table(d)
T = T[T.fy.isin(sel_fy) & T.rs_type_code.isin(sel_type)]
T["family"] = T.rs_type_code.map(FAMILY)

# ------------------------------------------------------------------ pages
if page == "Overview":
    k = L.kpis(T)
    if T.empty:
        st.info("No units match the filters."); st.stop()

    # --- live numbers for the tiles (all cached)
    j = d["poh_jobs"]
    as_on = min(max(dt.date(2025, 11, 30), j.date_received.min().date()), (j.date_dispatched.max() - pd.Timedelta(days=1)).date())
    wu = W.wip_as_on(d, get_ledger(DATA_DIR), as_on)[0]
    wu = wu[wu.rs_type_code.isin(sel_type)]
    vs = get_vs_standard(DATA_DIR).reindex(T.job_no).dropna()
    time_var = vs.man_hours.sum() / vs.allowed_hours.sum() - 1 if len(vs) else 0
    mat_var = vs.material_value.sum() / vs.std_material_value.sum() - 1 if len(vs) else 0
    rc = get_recon(DATA_DIR)
    rc = rc[L.fy_of(rc.date).isin(sel_fy)]
    paid = rc.paid_hours.sum() if "paid_hours" in rc else 0
    util = rc.booked.sum() / paid if paid else 0
    unacc = rc.unaccounted.sum() / paid if paid else 0
    flags = get_flags(DATA_DIR, 1.35)
    log = L.read_log(LOG_PATH)
    closed = set(log.flag_id) if len(log) else set()
    open_f = flags[~flags.flag_id.isin(closed)]
    high_open = int((open_f.severity == "High").sum())
    model = get_model(DATA_DIR)
    ask_log = os.environ.get("AIWCS_ASK_LOG", os.path.join(HERE, "assistant_log.csv"))
    n_asked = len(pd.read_csv(ask_log)) if os.path.exists(ask_log) else 0
    no_job = int((d["material_issues"].job_no == "").sum())

    fy_txt = ", ".join(sel_fy) if len(sel_fy) <= 3 else f"{sel_fy[0]} to {sel_fy[-1]}"
    st.markdown(f"""<div class="hero"><div class="eyebrow">ROLLING STOCK POH · REAL-TIME COSTING</div>
<h1>AI Assisted Workshop Costing System</h1>
<div class="sub">Every labour booking and every material issue, costed on the day it happens · cost, WIP and variance on any date</div>
<div class="chips"><span class="chip"><span class="dot" style="background:#2fa84f"></span>Live ledger · WIP as on {as_on:%d %b %Y}</span>
<span class="chip">FY {fy_txt}</span><span class="chip">{len(sel_type)} rolling stock types</span>
<span class="chip">On-cost method · BRD §6</span><span class="chip"><span class="dot" style="background:#9085e9"></span>Advisory AI · officer decides</span>
<span class="chip">DUMMY DATA</span></div></div>""", unsafe_allow_html=True)

    # AI alert of the day: the unit on the floor furthest past its AI estimate
    al = wu.copy()
    if len(al):
        al["ai_estimate"] = al.job_no.map(W.ai_estimates(d, model, al.job_no))
        done = j[j.date_dispatched <= pd.Timestamp(as_on)].assign(days=lambda x: (x.date_dispatched - x.date_received).dt.days + 1)
        al["usual_days"] = al.rs_type_code.map(done.groupby("rs_type_code").days.median()).fillna(done.days.median())
        al["cost_used"] = al.cost_to_date / al.ai_estimate
        u = al.loc[al.cost_used.idxmax()]
        disp = u.date_received + pd.Timedelta(days=round(u.usual_days) - 1)
        hot = u.cost_used > 0.9
        msg = (f'<b>{u.job_no}</b> · {type_names.get(u.rs_type_code, u.rs_type_code)} {u.get("coach_no", "")} has used '
               f'<span class="hl">{u.cost_used:.0%} of its AI estimate</span> ({rs(u.cost_to_date)} of {rs(u.ai_estimate)}) '
               f'on day {int(u.days_on_floor)} of a usual {int(u.usual_days)} · expected dispatch {disp:%d %b %Y}') if hot else \
              f'All {len(al)} units on the floor are within their AI estimate · highest {u.cost_used:.0%} ({u.job_no})'
        with st.container(key="alertbox"):
            st.markdown(f'<div class="alert{"" if hot else " ok"}"><div class="ring"><span class="ms">'
                        f'{"notifications_active" if hot else "verified"}</span></div><div class="tx"><div class="h">'
                        f'AI ALERT OF THE DAY · AS ON {as_on:%d %b %Y}</div><div class="m">{msg}</div></div>'
                        f'<div class="go">Review in Work in progress &rarr;</div></div>', unsafe_allow_html=True)
            st.button("Review", key="go_alert", on_click=go_to, args=("Work in progress",))

    mo = T.groupby("month_dispatched").total_cost_per_coach.agg(["sum", "mean"]).sort_index().tail(12)
    trw = get_wip_trend(DATA_DIR).sort_values("month_end").tail(12)
    c = st.columns(4)
    big(c[0], "Total POH cost", countup(k["total_cost"] / 1e7, 2, "₹ ", " Cr"), f"{k['units']:,} units turned out", PALETTE[2],
        spark(mo["sum"], PALETTE[2], "cost by month of dispatch · last 12 months"))
    big(c[1], "Average cost per unit", countup(k["avg_cost"] / 1e5, 2, "₹ ", " lakh"), f"{k['avg_hours']:,.0f} man-hours per unit",
        PALETTE[1], spark(mo["mean"], PALETTE[1], "average per unit · last 12 months"))
    big(c[2], "Work in progress now", countup(wu.cost_to_date.sum() / 1e7, 2, "₹ ", " Cr"),
        f"{len(wu):,} units on the floor · as on {as_on:%d %b}", PALETTE[0], spark(trw.wip, PALETTE[0], "WIP at month end · last 12 months"))
    big(c[3], "High-severity flags open", countup(high_open), f"{len(open_f):,} of {len(flags):,} flags awaiting an officer", PALETTE[7],
        seg_bar(open_f.severity.value_counts(), {"High": "#e66767", "Medium": "#c98500", "Low": "#6b6b80"}))

    # ---------------- AI impact
    hist = get_history(DATA_DIR)
    sm = I.summary(hist)
    mlo = get_ml_only(DATA_DIR)
    st.markdown('<div class="sect">AI impact<span>what changes with AI · replayed day by day on the Railway-held test year '
                f'({sm["units"]} units the model never trained on)</span></div>', unsafe_allow_html=True)
    ic = st.columns(5)
    imp = [("Cost known", "Same day", "every booking costed on the day it happens", "10 days after month-end"),
           ("Overruns seen before dispatch", f'{sm["caught"]} <small>of {sm["overruns"]}</small>',
            "units that finished > 5% above their AI estimate", "seen after the coach left"),
           ("Warning before dispatch", f'{sm["lead_days"]:.0f} <small>days</small>',
            f'median · {sm["earlier_days"]:.0f} days before the month-end report', "0 days"),
           ("Cost above estimate, seen early", f'₹ {sm["at_risk"]/1e7:,.2f} <small>Cr</small>',
            f'on those {sm["caught"]} units · cost at risk, not a saving', "found at month-end"),
           ("Found only by AI", f'{len(mlo)} <small>units</small>', "unusual cost profiles no rule check raised (Isolation Forest)",
            "not looked for")]
    for col, (k_, v_, n_, was) in zip(ic, imp):
        col.markdown(f'<div class="imp"><div class="k">{k_}</div><div class="v">{v_}</div><div class="n">{n_}</div>'
                     f'<div class="was">Without AI: {was}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="impnote">AI warning = first day the cost to date passes the AI estimate made at receipt. '
                'Estimates are advisory; an officer reviews every warning. DUMMY DATA.</div>', unsafe_allow_html=True)

    cand = hist[hist.overrun & hist.caught].sort_values("earlier_days", ascending=False)
    if len(cand):
        pick = st.selectbox("Replay a unit", cand.job_no, key="replay_unit",
                            format_func=lambda x: f"{x} · {type_names.get(cand.set_index('job_no').rs_type_code[x], '')} · "
                                                  f"{cand.set_index('job_no').actual[x] / cand.set_index('job_no').estimate[x] - 1:+.0%} over estimate")
        u = cand.set_index("job_no").loc[pick]
        u = u.copy(); u["job_no"] = pick
        left, right = st.columns([3, 2], gap="medium")
        cur = W.daily_curve(d, get_ledger(DATA_DIR), pick)
        left.markdown(lanes_html(u, u.estimate, cur), unsafe_allow_html=True)
        fig = px.area(cur, x="date", y="total", color_discrete_sequence=[PALETTE[6]], labels={"date": "", "total": "₹ (cumulative)"})
        fig.add_hline(y=u.estimate, line_dash="dot", line_color="#ffffff", annotation_text=f"AI estimate {rs(u.estimate)}",
                      annotation_position="top left")
        for x, txt, colr, anc in [(u.warned, f"AI warning · day {int(u.warn_day)}", PALETTE[4], "right"),
                                  (u.dispatched, "Dispatched", MUTED, "left"), (u.known_without_ai, "Cost known without AI", PALETTE[7], "right")]:
            fig.add_vline(x=x, line_dash="dash", line_color=colr)
            fig.add_annotation(x=x, y=1.0, yref="paper", yanchor="bottom", text=f"<b>{txt}</b>", showarrow=False, xanchor=anc,
                               font=dict(color=colr, size=11))
        fig.update_xaxes(range=[u.received - pd.Timedelta(days=1), u.known_without_ai + pd.Timedelta(days=2)])
        fig.update_layout(height=290, showlegend=False, margin=dict(l=14, r=14, t=34, b=14))
        left.plotly_chart(style(fig, "₹ (cumulative)"), width="stretch")
        rs_ = get_reasons(DATA_DIR, pick, u.warned.date())
        items = "".join(f"<li>{x}</li>" for x in rs_) or "<li>Cost to date passed the AI estimate.</li>"
        right.markdown(f'<div class="why"><div class="ttl"><span class="ms">psychology_alt</span>Why the AI warned</div>'
                       f'<div class="sub">{pick} on day {int(u.warn_day)} ({u.warned:%d %b %Y}), when the cost to date passed the '
                       f'AI estimate of {rs(u.estimate)}</div><ol>{items}</ol>'
                       f'<div class="act"><b>Advisory:</b> the AI points to where the money is going; the officer reviews, '
                       f'questions the bookings and decides. Nothing is posted by the AI.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sect">Costing features<span>click a tile to open it</span></div>', unsafe_allow_html=True)
    met = model["mape"] <= L.MAPE_TARGET
    r1, r2 = st.columns(4), st.columns(4)
    tr = get_wip_trend(DATA_DIR)
    wip_now = wu.cost_to_date.sum()
    peak = tr.wip.max() if len(tr) else wip_now
    n_issue = len(d["material_issues"])
    eff = vs.allowed_hours.sum() / vs.man_hours.sum() if len(vs) else 0
    tile(r1[0], "Work in progress", "factory", f"₹ {wip_now/1e7:,.2f} Cr",
         f"{len(wu):,} units on the floor, valued day by day", ("LIVE", "good"), PALETTE[0],
         (wip_now / peak if peak else 0, "vs highest month-end WIP", f"₹ {peak/1e7:,.2f} Cr"))
    tile(r1[1], "Unit cost & variance", "query_stats", f"{time_var:+.1%} time",
         f"Man-hours vs allowed time · material {mat_var:+.1%} vs standard",
         ("OVER STANDARD", "warn") if time_var > 0.05 or mat_var > 0.05 else ("WITHIN STANDARD", "good"), PALETTE[1],
         (eff, "time efficiency (allowed ÷ consumed)", f"{eff:.0%}"))
    tile(r1[2], "On-cost", "calculate", f"{k['oncost_share']:.1%}",
         f"of cost is labour + stores on-cost", ("BRD §6", "info"), PALETTE[3],
         (k["oncost_share"], "on-cost charged", f"₹ {T.oncost_total.sum()/1e7:,.2f} Cr"))
    tile(r1[3], "Labour reconciliation", "engineering", f"{util:.1%} utilised",
         f"Booked ÷ paid hours · unaccounted {unacc:.2%} (target < 2%)",
         ("UNACCOUNTED < 2%", "good") if unacc < 0.02 else ("UNACCOUNTED > 2%", "bad"), PALETTE[6],
         (util, "paid hours booked to units", f"{util:.1%}"))
    tile(r2[0], "Material", "inventory_2", f"₹ {T.material_value.sum()/1e7:,.2f} Cr",
         f"drawn at Book Average Rate · {no_job} issue notes without job no.",
         ("CHECK ISSUES", "warn") if no_job else ("ALL LINKED", "good"), PALETTE[5],
         (1 - no_job / n_issue if n_issue else 1, "issue notes linked to a job", f"{(1 - no_job / n_issue if n_issue else 1):.1%}"))
    tile(r2[1], "Anomaly review", "report", f"{len(open_f):,} open",
         f"{high_open:,} high severity · 7 rule checks + Isolation Forest",
         ("NEEDS REVIEW", "bad") if high_open else ("CLEAR", "good"), PALETTE[7],
         (1 - len(open_f) / len(flags) if len(flags) else 1, "closed by an officer", f"{len(flags) - len(open_f):,} of {len(flags):,}"))
    tile(r2[2], "Cost prediction (AI)", "model_training", f"{model['mape']:.1%} error",
         f"MAPE on held-out year (target ≤ {L.MAPE_TARGET:.0%}) · 16 gradient boosting models",
         ("TARGET MET", "good") if met else ("BELOW TARGET", "warn"), PALETTE[4],
         (1 - model["mape"], "accuracy (1 − MAPE)", f"{1 - model['mape']:.1%}"))
    tile(r2[3], "Costing assistant", "forum", "English · हिन्दी",
         f"Plain questions, figures computed from the data", ("AI", "info"), PALETTE[6],
         (1, "questions answered so far", f"{n_asked:,}"))

    st.markdown('<div class="sect">Cost trends<span>filtered by the sidebar</span></div>', unsafe_allow_html=True)
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
    with st.expander("AI model (Claude)", expanded=False):
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


st.markdown('<div class="foot">Cost = Direct material + Direct labour + Labour on-cost + Stores on-cost (BRD §6) · '
            'Advisory AI only: no AI output posts an accounting entry · DUMMY DATA</div>', unsafe_allow_html=True)

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
