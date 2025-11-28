import os
import json
import datetime as dt
from pathlib import Path
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import json

# =========================
# Setup
# =========================
st.set_page_config(
    page_title="Controle ZapZap",
    page_icon="assets/logo-2.webp", 
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
            st.warning(f"Falha ao consultar as contas via API: {e}")

    if DEFAULT_WABA_ID:
        return [{"id": DEFAULT_WABA_ID, "name": f"Default ({DEFAULT_WABA_ID})"}]
    return []

# =========================
# Sidebar 
# =========================
with st.sidebar:
    st.header("Parâmetros")

    # Carregar WABAs
    wabas = load_wabas()
    if not wabas:
        st.error("Nenhuma conta disponível.")
        st.stop()

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

    st.info("Carregando dados de todas as contas...")

    st.session_state.data_by_waba = {} 

    for w in wabas:
        waba_id = w["id"]
        waba_name = w.get("name", waba_id)

        try:
            payload = fetch_pricing_analytics(
                waba_id,
                to_epoch(start_date, end=False),
                to_epoch(end_date, end=True),
                TOKEN,
                "DAILY",
            )

            df = parse_to_df(payload)

            st.session_state.data_by_waba[waba_id] = {
                "payload": payload,
                "df": df,
                "name": waba_name,
            }

        except requests.HTTPError as e:
            # conta sem dados -> salva como vazia
            if e.response.status_code == 400:
                st.session_state.data_by_waba[waba_id] = {
                    "payload": None,
                    "df": pd.DataFrame(),
                    "name": waba_name,
                }
                continue
            else:
                st.warning(f"Falha ao carregar dados da conta {waba_name}: {e}")

        except Exception as e:
            st.warning(f"Erro ao carregar conta {waba_name}: {e}")

    if not st.session_state.data_by_waba:
        st.error("Nenhuma conta retornou dados para o período selecionado.")
    else:
        st.success("Dados carregados.")

tab1, tab2 = st.tabs(["Geral", "Limites"])

# =============================
#       TAB 1 (Geral)
# =============================
with tab1:

    if not st.session_state.get("data_by_waba"):
        st.info("Busque dados na barra lateral antes de visualizar o dashboard.")
        st.stop()

    todas_wabas_ids = list(st.session_state.data_by_waba.keys())

    if not todas_wabas_ids:
        st.warning("Nenhuma conta retornou dados válidos.")
        st.stop()

    # Seleção múltipla de contas 
    waba_selecionadas = st.multiselect(
        "Selecione as contas",
        todas_wabas_ids,
        default=todas_wabas_ids,
        format_func=lambda x: st.session_state.data_by_waba[x]["name"],
    )

    if not waba_selecionadas:
        waba_selecionadas = todas_wabas_ids

    # Concatena DFs das contas escolhidas
    dfs = []
    for w in waba_selecionadas:
        df_temp = st.session_state.data_by_waba[w]["df"].copy()
        if not df_temp.empty:
            df_temp["waba"] = st.session_state.data_by_waba[w]["name"]  
            dfs.append(df_temp)

    df_current = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if df_current.empty:
        st.warning("Sem dados para o período selecionado.")
        st.stop()

    # Normaliza
    df_current["phone"] = df_current["phone"].astype(str)
    df_current["date"] = df_current["time"].dt.date

    # -----------------------------------------
    # Filtro de número
    # -----------------------------------------
    numeros = sorted(df_current["phone"].astype(str).dropna().unique())

    phone_choices = st.multiselect(
        "Selecione os números",
        numeros,
        default=[] 
    )

    df_filtered = df_current.copy()

    if phone_choices:
        df_filtered = df_filtered[df_filtered["phone"].isin(phone_choices)]

    if df_filtered.empty:
        st.warning("Sem dados para esses números no período selecionado.")
        st.stop()

    df = df_filtered.copy()
    df["date"] = df["time"].dt.date

    current_waba_name = st.session_state.data_by_waba[waba_escolhida]["name"]
    df["waba"] = current_waba_name

    # -----------------------------
    # CARDS
    # -----------------------------
    total_vol = int(df["volume"].sum())
    paid_mask = df["pricing_type"].str.upper().isin(PAID_TYPES)
    paid_vol = int(df.loc[paid_mask, "volume"].sum())
    paid_cost = float(df.loc[paid_mask, "cost"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Volume total", f"{total_vol:,}".replace(",", "."), help="Total de mensagens enviadas no período")
    c2.metric("Mensagens pagas", f"{paid_vol:,}".replace(",", "."),help="Mensagens tarifadas pela Meta")
    c3.metric("Custo", usd(paid_cost),help="Custo total das mensagens pagas")

    # -----------------------------
    # GRÁFICO
    # -----------------------------
    daily_total = (
        df.groupby("date", as_index=False)["volume"]
        .sum()
        .rename(columns={"volume": "volume_total"})
    )

    daily_pagas = (
        df[paid_mask]
        .groupby("date", as_index=False)["volume"]
        .sum()
        .rename(columns={"volume": "volume_pagas"})
    )

    daily = pd.merge(daily_total, daily_pagas, on="date", how="left").fillna(0)

    fig = px.line(
        daily,
        x="date",
        y=["volume_total", "volume_pagas"],
        markers=True,
        title="Evolução Diária"
    )

    st.plotly_chart(fig, width="stretch")

    # -----------------------------
    # TABELA
    # -----------------------------
    tabela = (
        df.groupby(["date", "waba", "phone"], as_index=False)
        .agg(mensagens_pagas=("volume", "sum"))
        .sort_values(["date", "waba", "phone"])
    )

    with st.expander("Ver tabela detalhada"):
        st.dataframe(tabela, width="stretch")

        csv = tabela.to_csv(index=False).encode()
        st.download_button(
            "Baixar CSV",
            csv,
            "mensagens_pagas.csv",
            "text/csv"
        )
pass

# =============================
#       TAB 2 (Limites)
# =============================
with tab2:
    # carregar configuração de grupos
    try:
        grupos_raw = st.secrets["GRUPOS"]
        group_cfg = {nome: json.loads(cfg_str) for nome, cfg_str in grupos_raw.items()}
    except Exception as e:
        st.error(f"Erro ao carregar grupos")
        group_cfg = {}

    # pegar TODOS os df carregados 
    dfs = [v["df"] for v in st.session_state.data_by_waba.values() if v.get("df") is not None]
    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if df_all.empty:
        st.info("Busque dados antes de visualizar.")
        st.stop()

    df_all["phone"] = df_all["phone"].astype(str)
    df_all["date"] = df_all["time"].dt.date

    # ----------------------------
    # Data
    # ----------------------------
    hoje = dt.date.today()

    ini, fim = start_date, end_date

    # filtrar df geral pelo periodo
    df_ciclo = df_all[(df_all["date"] >= ini) & (df_all["date"] <= fim)]

    # ----------------------------
    # Renderização por grupo
    # ----------------------------
    for grupo, cfg in group_cfg.items():

        numeros = [str(n) for n in cfg.get("numeros", [])]
        limite = cfg.get("limite", 0)

        df_g = df_ciclo[df_ciclo["phone"].isin(numeros)]

        st.subheader(grupo)

        if df_g.empty:
            st.warning("Nenhum dado encontrado para este grupo neste período.")
            continue

        df_pagas = df_g[df_g["pricing_type"].str.upper().isin(PAID_TYPES)]

        usadas = int(df_pagas["volume"].sum())
        custo = float(df_pagas["cost"].sum())
        perc = (usadas / limite * 100) if limite > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pagas usadas", f"{usadas:,}".replace(",", "."),help="Mensagens pagas que foram enviadas")
        c2.metric("Limite", f"{limite:,}".replace(",", "."),help="Limite de mensagens pagas definido pela Auvo")
        c3.metric("Utilizado", f"{perc:.1f}%",help="Taxa de uso em relação ao limite")
        c4.metric("Custo total", usd(custo),help="Custo total das mensagens pagas")

        if limite > 0:
            st.progress(min(perc / 100, 1.0))
        else:
            st.info("Nenhum limite definido para este grupo.")

        st.markdown("---")
