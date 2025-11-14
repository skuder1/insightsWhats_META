import os
import json
import datetime as dt
from pathlib import Path
from typing import Dict, Any, List

import altair as alt
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# =========================
# Setup
# =========================
st.set_page_config(
    page_title="Controle ZapZap",
    page_icon=r"C:\Users\Auvo1\Desktop\PythonProjects\insightsWhats_META\assets\logo-2.webp", 
    layout="wide",
)
load_dotenv()

TOKEN = os.getenv("META_ACCESS_TOKEN", "")
DEFAULT_WABA_ID = os.getenv("WABA_ID", "")  
API_VERSION = os.getenv("API_VERSION", "")
BUSINESS_ID = os.getenv("BUSINESS_ID", "")
BASE = f"https://graph.facebook.com/{API_VERSION}"
PAID_TYPES = {"REGULAR"}

if not TOKEN:
    st.error("Configure o token de acesso")
    st.stop()

APP_DIR = Path(__file__).resolve().parent
WABAS_FILE = APP_DIR / "wabas.json"

if "data_by_waba" not in st.session_state:
    st.session_state.data_by_waba = {}
if "selected_waba" not in st.session_state:
    st.session_state.selected_waba = DEFAULT_WABA_ID or None

# =========================
# Helpers
# =========================

def _get_all_pages(url, params):
    out = []
    while True:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("data", []))
        # paginação
        paging = data.get("paging", {})
        next_url = paging.get("next")
        if not next_url:
            break
        url = next_url
        params = {}  
    return out

@st.cache_data(ttl=600)
def fetch_wabas_via_api(business_id: str, token: str) -> list[dict]:
    url = f"{BASE}/{business_id}/owned_whatsapp_business_accounts"
    params = {"fields": "id,name", "access_token": token}
    items = _get_all_pages(url, params)
    return [{"id": it.get("id"), "name": it.get("name") or it.get("id")} for it in items if it.get("id")]

@st.cache_data(ttl=600)
def fetch_numbers_via_api(waba_id: str, token: str) -> list[dict]:
    url = f"{BASE}/{waba_id}/phone_numbers"
    params = {"fields": "id,display_phone_number,verified_name", "access_token": token}
    items = _get_all_pages(url, params)
    res = []
    for it in items:
        res.append({
            "id": it.get("id"),
            "display": it.get("display_phone_number") or "",
            "name": it.get("verified_name") or "",
        })
    return res

def to_epoch(d: dt.date, end=False) -> int:
    dt_obj = dt.datetime.combine(d, dt.time.max if end else dt.time.min)
    return int(dt_obj.timestamp())

@st.cache_data(ttl=600)
def fetch_pricing_analytics(
    waba_id: str, start_epoch: int, end_epoch: int, token: str, granularity: str = "DAILY"
) -> Dict[str, Any]:
    fields = (
        f"pricing_analytics.start({start_epoch}).end({end_epoch})"
        f".granularity({granularity})"
        f".metric_types([\"VOLUME\",\"COST\"])"
        f".dimensions([\"PHONE\",\"PRICING_TYPE\",\"PRICING_CATEGORY\",\"TIER\"])"
    )
    url = f"{BASE}/{waba_id}"
    params = {"fields": fields, "access_token": token}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def parse_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    pa = payload.get("pricing_analytics", {})
    data = pa.get("data", [])
    points: List[Dict[str, Any]] = []
    for chunk in data:
        points.extend(chunk.get("data_points") or [])
    if not points:
        return pd.DataFrame()

    df = pd.DataFrame(points).rename(
        columns={
            "phone_number": "phone",
            "pricing_type": "pricing_type",
            "pricing_category": "pricing_category",
            "tier": "tier",
            "volume": "volume",
            "cost": "cost",
            "start": "start",
            "end": "end",
        }
    )
    df["time"] = pd.to_datetime(df.get("start"), unit="s", utc=True, errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)
    df["cost"] = pd.to_numeric(df.get("cost", 0.0), errors="coerce").fillna(0.0).astype(float)
    df["pricing_type"] = df.get("pricing_type", "").astype(str)
    df = df.sort_values("time").reset_index(drop=True)
    return df

def usd(x: float) -> str:
    return f"$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def load_wabas() -> list[dict]:
    if BUSINESS_ID:
        try:
            items = fetch_wabas_via_api(BUSINESS_ID, TOKEN)
            if items:
                return items
        except requests.RequestException as e:
            st.warning(f"Falha ao listar WABAs via API: {e}")


    try:
        if WABAS_FILE.exists():
            return json.loads(WABAS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"Falha ao ler wabas.json: {e}")


    if DEFAULT_WABA_ID:
        return [{"id": DEFAULT_WABA_ID, "name": f"Default ({DEFAULT_WABA_ID})"}]
    return []

# =========================
# Sidebar (WABA + datas + buscar)
# =========================
with st.sidebar:
    st.header("Parâmetros")

    # Carregar WABAs
    wabas = load_wabas()
    if not wabas:
        st.error("Nenhuma conta disponível.")
        st.stop()

    # Select WABA 
    waba_labels = [w.get("name") or w["id"] for w in wabas]
    waba_ids    = [w["id"] for w in wabas]
    # Seleciona atual se existir, senão a primeira
    initial_index = waba_ids.index(st.session_state.selected_waba) if st.session_state.selected_waba in waba_ids else 0
    waba_idx = st.selectbox(
    "Conta", options=list(range(len(wabas))), index=initial_index, format_func=lambda i: waba_labels[i], key="waba_select")
    current_waba_id = waba_ids[waba_idx]
    st.session_state.selected_waba = current_waba_id

    # Regra do dia 15, data sempre começa dia 15
    today = dt.date.today()
    if today.day >= 15:
        default_start = dt.date(today.year, today.month, 15)
    else:
        prev_year  = today.year if today.month > 1 else today.year - 1
        prev_month = today.month - 1 if today.month > 1 else 12
        default_start = dt.date(prev_year, prev_month, 15)

    start_date = st.date_input("Início", value=default_start, max_value=today, key="start_date")
    end_date = st.date_input("Fim", value=today, max_value=today, min_value=start_date, key="end_date")

    fetch_btn = st.button("Buscar", type="primary", key="fetch")


# =========================
# Busca
# =========================
if fetch_btn:
    if start_date > end_date:
        st.error("Data inicial não pode ser maior que a final.")
        st.stop()

    try:
        payload = fetch_pricing_analytics(
            current_waba_id,
            to_epoch(start_date, end=False),
            to_epoch(end_date, end=True),
            TOKEN,
            "DAILY",
        )
        df = parse_to_df(payload)
    except requests.HTTPError as e:
        st.error(f"Erro HTTP: {e}")
        st.stop()
    except requests.RequestException as e:
        st.error(f"Erro de rede: {e}")
        st.stop()

    st.session_state.data_by_waba[current_waba_id] = {"payload": payload, "df": df}

# =========================
# Filtro de número 
# =========================
cache = st.session_state.data_by_waba.get(current_waba_id)
df_current = None if cache is None else cache.get("df")

with st.sidebar:
    phone_options = [""]
    try:
        api_numbers = fetch_numbers_via_api(current_waba_id, TOKEN)  
    except requests.RequestException:
        api_numbers = []

    if api_numbers:
        phone_options += [n.get("display") or "" for n in api_numbers if n.get("display")]

    cache = st.session_state.data_by_waba.get(current_waba_id)
    df_current = None if cache is None else cache.get("df")
    if (not api_numbers) and df_current is not None and not df_current.empty and "phone" in df_current.columns:
        phones_from_df = df_current["phone"].dropna().astype(str).unique().tolist()
        phone_options += phones_from_df

    phone_choice = st.selectbox(
        "Filtrar número",
        options=["Todos"] + sorted(set([p for p in phone_options if p])), 
        index=0,
        key=f"phone_choice_{current_waba_id}",
    )

# =========================
# Render principal
# =========================
if df_current is None or df_current.empty:
    st.info("Selecione a conta, o período na barra lateral e clique em **Buscar**.")
else:
    df = df_current.copy()
    if phone_choice != "Todos":
        df = df[df["phone"].astype(str) == str(phone_choice)]

    # --- Cards ---
    total_vol = int(df["volume"].sum())
    paid_mask = df["pricing_type"].str.upper().isin(PAID_TYPES)
    paid_vol = int(df.loc[paid_mask, "volume"].sum())
    paid_cost = float(df.loc[paid_mask, "cost"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Volume total", f"{total_vol:,}".replace(",", "."))
    c2.metric("Mensagens pagas", f"{paid_vol:,}".replace(",", "."))
    c3.metric("Custo", usd(paid_cost))

    # --- Gráfico diário ---
    daily_total = df.groupby("time", as_index=False)[["volume"]].sum().rename(columns={"volume": "volume_total"})
    daily_paid = df.loc[paid_mask].groupby("time", as_index=False)[["volume"]].sum().rename(columns={"volume": "volume_pagas"})
    daily = pd.merge(daily_total, daily_paid, on="time", how="left").fillna(0)

    st.subheader(f"Evolução diária")

    chart = alt.Chart(daily).transform_fold(
        ["volume_total", "volume_pagas"], as_=["serie", "valor"]
    ).mark_line(point=True).encode(
        x="time:T", y="valor:Q", color="serie:N", tooltip=["time:T", "serie:N", "valor:Q"]
    )
    
    st.altair_chart(chart, use_container_width=True)

    # --- Debug ---
    with st.expander("Debug – Resposta da API"):
        st.code(json.dumps(st.session_state.data_by_waba[current_waba_id]["payload"], indent=2))
