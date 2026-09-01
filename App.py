from __future__ import annotations

import io
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI  # SDK OpenAI serve para Gemini, Groq e OpenRouter (todos compatíveis)

# ==========================================================
# CONFIGURAÇÕES GERAIS
# ==========================================================

APP_TITLE = "Nescau — Consumer Insights"
APP_SUBTITLE = "Como os consumidores percebem o produto?"

# --------- PROVEDORES GRATUITOS (sem cartão de crédito) ----------
# Cada preset tem: base_url (endpoint OpenAI-compatível), modelo default e onde pegar a key.
PROVIDERS: Dict[str, Dict[str, str]] = {
    "Google Gemini (AI Studio)": {
        # gemini-2.5-flash foi descontinuado para novos usuários (nov/2025).
        # Use gemini-3.6-flash, que é o recomendado pelo próprio Google.
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.6-flash",
        "key_hint": "Pegue grátis em ai.google.dev → Get API Key (começa com 'AIza...').",
        "key_prefix": "AIza",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_hint": "Pegue grátis em console.groq.com/keys (começa com 'gsk_...').",
        "key_prefix": "gsk_",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "key_hint": "Pegue grátis em openrouter.ai → Keys (começa com 'sk-or-...'). Use IDs terminados em ':free'.",
        "key_prefix": "sk-or-",
    },
}

# Limite defensivo de caracteres por comentário (evita payloads gigantes).
MAX_COMMENT_CHARS = 600

# Tamanho do lote de comentários enviado por chamada de API.
BATCH_SIZE = int(os.getenv("NESCAU_BATCH_SIZE", "40"))

# Colunas esperadas no CSV.
COL_DATA = "data"
COL_NOTA = "1-5"
COL_COMENTARIO = "comentario"
REQUIRED_COLUMNS = [COL_DATA, COL_NOTA, COL_COMENTARIO]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# 1. CARREGAR DADOS
# ==========================================================

def load_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if uploaded_file is None:
        return None, "Nenhum arquivo foi enviado."

    raw = uploaded_file.getvalue()
    if not raw or len(raw.strip()) == 0:
        return None, "O arquivo enviado está vazio."

    attempts = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "latin-1"},
        {"sep": ";", "encoding": "latin-1"},
    ]

    last_error: Optional[str] = None
    for opts in attempts:
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                sep=opts["sep"],
                encoding=opts["encoding"],
                dtype=str,
                keep_default_na=False,
                engine="python",
            )
            if df.shape[1] >= 2:
                df.columns = [str(c).strip().lower() for c in df.columns]
                return df, None
        except Exception as exc:
            last_error = str(exc)
            continue

    return None, f"Não foi possível interpretar o arquivo como CSV. Detalhe técnico: {last_error}"

# ==========================================================
# 2. VALIDAR CSV
# ==========================================================

def validate_csv(df: Optional[pd.DataFrame]) -> Tuple[bool, List[str]]:
    messages: List[str] = []

    if df is None:
        return False, ["O arquivo não pôde ser lido."]

    if df.empty:
        return False, ["O CSV não contém nenhuma linha de dados."]

    present = set(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in present]
    if missing:
        messages.append(
            "Colunas obrigatórias ausentes: "
            + ", ".join(f"'{c}'" for c in missing)
            + f". O CSV deve conter as colunas: {', '.join(REQUIRED_COLUMNS)}."
        )
        return False, messages

    return True, messages

def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    report = {
        "linhas_originais": len(df),
        "notas_invalidas": 0,
        "datas_invalidas": 0,
        "comentarios_vazios": 0,
    }

    work = df.copy()
    work[COL_NOTA] = pd.to_numeric(work[COL_NOTA], errors="coerce")
    valid_mask = work[COL_NOTA].between(1, 5)
    report["notas_invalidas"] = int((~valid_mask).sum())
    work.loc[~valid_mask, COL_NOTA] = pd.NA
    work[COL_NOTA] = work[COL_NOTA].round().astype("Int64")

    work[COL_DATA] = pd.to_datetime(work[COL_DATA], errors="coerce", dayfirst=True)
    report["datas_invalidas"] = int(work[COL_DATA].isna().sum())

    work[COL_COMENTARIO] = work[COL_COMENTARIO].astype(str).str.strip()
    empty_mask = work[COL_COMENTARIO].isin(["", "nan", "none", "null"])
    report["comentarios_vazios"] = int(empty_mask.sum())
    work = work[~empty_mask].reset_index(drop=True)

    return work, report

# ==========================================================
# 4 / 3. CALCULAR MÉTRICAS (LOCAL, PANDAS)
# ==========================================================

def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    notas = df[COL_NOTA].dropna()
    total = len(df)
    total_com_nota = int(notas.count())

    metrics: Dict[str, Any] = {
        "total_avaliacoes": total,
        "total_com_nota": total_com_nota,
        "media": round(float(notas.mean()), 2) if total_com_nota else None,
        "mediana": float(notas.median()) if total_com_nota else None,
        "distribuicao": {},
        "percentual": {},
    }

    for n in range(1, 6):
        qtd = int((notas == n).sum())
        metrics["distribuicao"][n] = qtd
        metrics["percentual"][n] = round(qtd / total_com_nota * 100, 1) if total_com_nota else 0.0

    positivas = int((notas >= 4).sum())
    negativas = int((notas <= 2).sum())
    metrics["pct_positivas"] = round(positivas / total_com_nota * 100, 1) if total_com_nota else 0.0
    metrics["pct_negativas"] = round(negativas / total_com_nota * 100, 1) if total_com_nota else 0.0

    return metrics

def build_time_series(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    valid = df.dropna(subset=[COL_DATA]).copy()
    if valid.empty:
        return None

    valid["dia"] = valid[COL_DATA].dt.date
    grouped = (
        valid.groupby("dia")
        .agg(qtd_avaliacoes=(COL_COMENTARIO, "count"), media_nota=(COL_NOTA, "mean"))
        .reset_index()
        .sort_values("dia")
    )
    grouped["media_nota"] = grouped["media_nota"].round(2)
    grouped["media_movel_7d"] = grouped["media_nota"].rolling(window=7, min_periods=1).mean().round(2)
    return grouped

# ==========================================================
# 3 / 4. PREPARAR COMENTÁRIOS (LOTES)
# ==========================================================

def prepare_comments(df: pd.DataFrame) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        texto = str(row[COL_COMENTARIO]).strip()
        if not texto: continue
        if len(texto) > MAX_COMMENT_CHARS:
            texto = texto[:MAX_COMMENT_CHARS] + "…"
        nota = row[COL_NOTA]
        items.append({
            "id": int(idx),
            "nota": int(nota) if pd.notna(nota) else None,
            "comentario": texto,
        })
    return items

def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]

# ==========================================================
# 5. PROMPTS DA IA
# ==========================================================

SYSTEM_PROMPT = (
    "Você é um analista sênior de Consumer Insights especializado em produtos de grande consumo. "
    "Você extrai DADO (o que foi observado), INTERPRETAÇÃO (o significado) e RECOMENDAÇÃO (ação sugerida). "
    "Diferencie percepção do PRODUTO da percepção da MARCA e de CAMPANHAS/PROPAGANDAS. "
    "Mapeie pontos fortes e pontos a melhorar sempre! "
    "Responda SEMPRE e exclusivamente em JSON válido, sem texto extra ou formatações Markdown fora do JSON."
)

BATCH_INSTRUCTION = """
Analise os comentários abaixo. Retorne um JSON estrito no formato abaixo.

Para CADA comentário, retorne um objeto com:
- "id": o mesmo id recebido
- "sentimento_geral": um de ["positivo","negativo","neutro","misto"]
- "intensidade": um de ["baixa","media","alta"]
- "aspectos": lista de objetos {"aspecto": "<nome curto>", "sentimento": "positivo|negativo|neutro"}
- "marca": {"mencionada": true|false, "sentimento": "positivo|negativo|neutro|na"}
- "comportamento": lista de ["recompra","recomendacao","abandono","decepcao", "expectativa_superada"]

Formato de Saída (Exemplo):
{"resultados": [{"id": 0, "sentimento_geral": "positivo", "intensidade": "alta", "aspectos": [{"aspecto": "sabor", "sentimento": "positivo"}], "marca": {"mencionada": false, "sentimento": "na"}, "comportamento": ["recompra"]}]}
"""

SYNTHESIS_INSTRUCTION = """
Com base nos dados agregados e amostra de comentários fornecidos, produza uma análise executiva em formato JSON.
Mapeie especificamente se houver opiniões sobre campanhas, propagandas ou ações de marketing.

Retorne JSON no formato EXATO abaixo:
{
  "sentimento": {"positivo": 0, "negativo": 0, "neutro": 0, "misto": 0},
  "aspectos": [
    {"nome": "Sabor", "mencoes": 0, "positivas": 0, "negativas": 0, "sentimento_predominante": "positivo", "resumo": ""}
  ],
  "pontos_positivos": [{"aspecto": "O que as pessoas mais gostaram", "mencoes": 0, "resumo": ""}],
  "pontos_negativos": [{"aspecto": "O que as pessoas não gostaram/defeitos", "mencoes": 0, "resumo": ""}],
  "oportunidades": [{"tema": "", "problema": "", "oportunidade": "", "evidencia": ""}],
  "comportamento": {"recompra": "", "recomendacao": "", "fidelidade": "", "risco_abandono": ""},
  "marca": {
    "percepcao": "Resumo de como veem a marca em si",
    "principais_pontos": ["Ponto 1"],
    "opiniao_publica_campanha": "Se houver menções a campanhas, comerciais ou propaganda, resuma a opinião pública geral sobre elas aqui. Diga se agradou ou não."
  },
  "insights": [{"titulo": "", "evidencia": "", "interpretacao": "", "importancia": ""}]
}
"""

# ==========================================================
# 6. CHAMAR A API (GEMINI / GROQ / OPENROUTER — FORMATO OPENAI)
# ==========================================================

def get_api_key(env_names: Tuple[str, ...]) -> Optional[str]:
    """Lê a chave do provedor a partir de st.secrets ou variáveis de ambiente.
    O acesso a st.secrets é protegido porque, sem secrets.toml, ele levanta exceção."""
    for name in env_names:
        try:
            if name in st.secrets:
                val = str(st.secrets[name]).strip()
                if val:
                    return val
        except Exception:
            pass
        val = os.getenv(name, "").strip()
        if val:
            return val
    return None


def _build_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def _call_llm_json(api_key: str, base_url: str, model: str, system_prompt: str, user_content: str) -> Dict[str, Any]:
    client = _build_client(api_key, base_url)

    # Pequeno delay para respeitar rate limits de planos gratuitos.
    time.sleep(1)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Tenta forçar JSON nativo; se o provedor não aceitar, refaz sem o parâmetro.
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )

    return safe_json_loads(response.choices[0].message.content)

def analyze_batches(api_key: str, base_url: str, model: str, comments: List[Dict[str, Any]], batch_size: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    resultados: List[Dict[str, Any]] = []
    avisos: List[str] = []

    batches = chunk_list(comments, batch_size)
    progress = st.progress(0.0, text=f"Interpretando comentários com {model}…")

    for i, batch in enumerate(batches):
        payload = json.dumps(batch, ensure_ascii=False)
        user_content = BATCH_INSTRUCTION + "\n\nComentários:\n" + payload
        try:
            data = _call_llm_json(api_key, base_url, model, SYSTEM_PROMPT, user_content)
            lote_res = data.get("resultados", [])
            if isinstance(lote_res, list):
                resultados.extend(lote_res)
            else:
                avisos.append(f"Lote {i + 1}: formato inesperado, ignorado.")
        except Exception as exc:
            avisos.append(f"Lote {i + 1}: falha na análise ({str(exc)}).")
        progress.progress((i + 1) / len(batches), text=f"Interpretando comentários com {model}…")

    progress.empty()
    return resultados, avisos

def synthesize_insights(
    api_key: str, base_url: str, model: str, metrics: Dict[str, Any], aspect_agg: Dict[str, Any],
    sentiment_counts: Dict[str, int], sample_comments: List[Dict[str, Any]], temporal_hint: str
) -> Dict[str, Any]:
    context = {
        "estatisticas_quantitativas": metrics,
        "sentimento_agregado": sentiment_counts,
        "aspectos_agregados": aspect_agg,
        "sinal_temporal": temporal_hint,
        "amostra_comentarios": sample_comments,
    }
    user_content = SYNTHESIS_INSTRUCTION + "\n\nDados de contexto (JSON):\n" + json.dumps(context, ensure_ascii=False)
    return _call_llm_json(api_key, base_url, model, SYSTEM_PROMPT, user_content)

def safe_json_loads(content: Optional[str]) -> Dict[str, Any]:
    if not content: raise ValueError("Resposta vazia da API.")
    content = content.strip()
    content = re.sub(r"^```(?:json)?", "", content).strip()
    content = re.sub(r"```$", "", content).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except json.JSONDecodeError: pass
    raise ValueError("A API retornou um JSON inválido.")

# ==========================================================
# 7. PROCESSAR RESPOSTA
# ==========================================================

def aggregate_batch_results(df: pd.DataFrame, comment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    sentiment_counts = {"positivo": 0, "negativo": 0, "neutro": 0, "misto": 0}
    aspect_stats: Dict[str, Dict[str, int]] = {}
    brand_mentions = {"positivo": 0, "negativo": 0, "neutro": 0}
    behavior_counts: Dict[str, int] = {}
    map_sentimento: Dict[int, str] = {}
    map_aspecto: Dict[int, str] = {}

    for res in comment_results:
        if not isinstance(res, dict): continue
        cid = res.get("id")
        sent = str(res.get("sentimento_geral", "")).lower()
        if sent in sentiment_counts: sentiment_counts[sent] += 1
        if cid is not None: map_sentimento[cid] = sent or "n/d"

        aspectos = res.get("aspectos", []) or []
        aspecto_principal = None
        for a in aspectos:
            if not isinstance(a, dict): continue
            nome = str(a.get("aspecto", "")).strip().lower()
            if not nome: continue
            asent = str(a.get("sentimento", "neutro")).lower()
            bucket = aspect_stats.setdefault(nome, {"mencoes": 0, "positivas": 0, "negativas": 0, "neutras": 0})
            bucket["mencoes"] += 1
            if asent == "positivo": bucket["positivas"] += 1
            elif asent == "negativo": bucket["negativas"] += 1
            else: bucket["neutras"] += 1
            if aspecto_principal is None: aspecto_principal = nome
        if cid is not None: map_aspecto[cid] = aspecto_principal or "n/d"

        marca = res.get("marca", {}) or {}
        if isinstance(marca, dict) and marca.get("mencionada"):
            msent = str(marca.get("sentimento", "neutro")).lower()
            if msent in brand_mentions: brand_mentions[msent] += 1

        for b in res.get("comportamento", []) or []:
            key = str(b).lower().strip()
            if key: behavior_counts[key] = behavior_counts.get(key, 0) + 1

    df["sentimento_ia"] = df.index.map(lambda i: map_sentimento.get(i, "n/d"))
    df["aspecto_principal"] = df.index.map(lambda i: map_aspecto.get(i, "n/d"))

    aspectos_ordenados = [
        {"nome": nome, "mencoes": v["mencoes"], "positivas": v["positivas"], "negativas": v["negativas"], "neutras": v["neutras"]}
        for nome, v in sorted(aspect_stats.items(), key=lambda kv: kv[1]["mencoes"], reverse=True)
    ]

    return {"sentiment_counts": sentiment_counts, "aspectos": aspectos_ordenados, "marca": brand_mentions, "comportamento": behavior_counts}

def validate_synthesis(data: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "sentimento": {"positivo": 0, "negativo": 0, "neutro": 0, "misto": 0},
        "aspectos": [], "pontos_positivos": [], "pontos_negativos": [], "oportunidades": [],
        "comportamento": {"recompra": "", "recomendacao": "", "fidelidade": "", "risco_abandono": ""},
        "marca": {"percepcao": "", "principais_pontos": [], "opiniao_publica_campanha": ""},
        "insights": [],
    }
    if not isinstance(data, dict): return defaults
    for key, default in defaults.items():
        if key not in data or data[key] is None: data[key] = default
    return data

# ==========================================================
# 8. DESIGN SYSTEM E GRÁFICOS (PLOTLY)
# ==========================================================

CHOCO_DARK = "#2A1208"
CHOCO_SOFT = "#7A3E1D"
GOLD       = "#C98A00"
INK        = "#1C140F"
POS_COLOR  = "#1B7A3D"
NEG_COLOR  = "#C0261A"
NEU_COLOR  = "#5C534C"
MIS_COLOR  = "#B26A00"

def inject_design_system() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Montserrat:wght@700;800;900&display=swap');
    :root { --choco:#241006; --choco-2:#3A1A0D; --gold:#B87E00; --cream:#FFF7E7; --ink:#171009; }
    html, body, [class*="css"] { font-family:'Manrope',sans-serif; font-size:17px; }
    .stApp { background: radial-gradient(circle at 95% 2%, rgba(245,184,0,.10), transparent 26%), #FFFCF6; color:var(--ink); }
    .stApp, .stMarkdown, p, span, li, label { color:var(--ink); }
    .block-container { max-width:1280px; padding-top:1.2rem; padding-bottom:4rem; }
    header[data-testid="stHeader"] { background:transparent; }
    #MainMenu, footer { visibility:hidden; }
    h1,h2,h3 { font-family:'Montserrat',sans-serif !important; letter-spacing:-.03em; }
    h2 { color:var(--choco) !important; }

    [data-testid="stFileUploader"] { background:#fff; border:2px dashed rgba(58,26,13,.45); border-radius:22px; padding:1rem; box-shadow:0 12px 35px rgba(63,33,19,.10); }
    [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] span { color:#241006 !important; font-weight:600; font-size:1.02rem; }
    section[data-testid="stSidebar"] { background:#2A1208; }
    section[data-testid="stSidebar"] * { color:#FFF7E7 !important; }

    div[data-testid="stMetric"] { background:white; border:1px solid rgba(58,26,13,.14); border-radius:22px; padding:1.15rem 1.2rem; box-shadow:0 12px 30px rgba(63,33,19,.10); }
    div[data-testid="stMetricLabel"] { color:#3A2A20; font-weight:800; font-size:1rem; }
    div[data-testid="stMetricValue"] { color:var(--choco); font-family:'Montserrat',sans-serif; font-size:2.4rem; font-weight:900; }

    div[data-testid="stPlotlyChart"] { background:white; border:1px solid rgba(58,26,13,.14); border-radius:24px; padding:.55rem; box-shadow:0 14px 36px rgba(63,33,19,.10); overflow:hidden; }
    [data-testid="stDataFrame"] { border-radius:18px; overflow:hidden; border:1px solid rgba(58,26,13,.16); }
    .stButton>button { border:0; border-radius:999px; min-height:3.2rem; padding:0 1.7rem; font-weight:900; font-size:1.05rem; color:#231005; background:linear-gradient(135deg,#FFCE33,#F0AF00); box-shadow:0 10px 22px rgba(245,184,0,.35); transition:.2s ease; }

    .hero { position:relative; overflow:hidden; border-radius:34px; padding:3.2rem 3rem; color:white; background:linear-gradient(120deg,#1E0D04 0%,#3A1A0D 58%,#5A2C17 100%); box-shadow:0 24px 55px rgba(42,18,8,.28); margin-bottom:1.6rem; }
    .hero-kicker { display:inline-flex; border:1px solid rgba(255,255,255,.34); padding:.5rem .9rem; border-radius:999px; text-transform:uppercase; font-size:.82rem; font-weight:900; color:#FFD84D; letter-spacing:.04em; }
    .hero h1 { margin:.8rem 0 .45rem; font-size:clamp(2.6rem,5vw,4.9rem); line-height:.98; color:white !important; }
    .hero p { color:#FBEBD2 !important; font-size:1.18rem; font-weight:500; }

    .section-head { margin:2.3rem 0 .85rem; display:flex; gap:1rem; align-items:flex-end; justify-content:space-between; }
    .section-eyebrow { color:#8A4A00; text-transform:uppercase; font-weight:900; font-size:.86rem; letter-spacing:.06em; }
    .section-title { margin:.15rem 0 0; color:#241006; font-family:'Montserrat',sans-serif; font-size:2.1rem; font-weight:900; }
    .section-copy { color:#4A382E; font-size:1.02rem; font-weight:500; max-width:38ch; text-align:right; }

    .kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:.2rem 0 1.2rem; }
    .kpi { background:#fff; border-radius:24px; padding:1.35rem 1.3rem; box-shadow:0 14px 36px rgba(63,33,19,.10); border:1px solid rgba(58,26,13,.10); }
    .kpi-label { color:#3A2A20; font-size:.95rem; text-transform:uppercase; font-weight:900; letter-spacing:.03em; }
    .kpi-value { margin:.4rem 0 .18rem; font-family:'Montserrat',sans-serif; font-size:2.7rem; font-weight:900; color:#241006; line-height:1; }
    .kpi-note { color:#5A463C; font-size:.9rem; font-weight:600; }

    .panel { background:white; border-radius:24px; padding:1.5rem; margin-bottom:1rem; box-shadow:0 14px 36px rgba(63,33,19,.10); border:1px solid rgba(58,26,13,.10); }
    .panel p, .panel b, .panel strong { color:#241006 !important; font-size:1.05rem; }

    .ai-shell { padding:2px; border-radius:30px; background:linear-gradient(135deg,#F0AF00,#FFDF68 38%,#5A2C17); margin:1rem 0; }
    .ai-inner { border-radius:28px; padding:1.7rem; background:linear-gradient(145deg,#241006,#3A1A0D); color:white; }
    .ai-badge { color:#FFD84D; font-size:.82rem; text-transform:uppercase; font-weight:900; }
    .ai-title { color:white; font-family:'Montserrat',sans-serif; font-size:2.1rem; font-weight:900; margin:.25rem 0; }
    .ai-copy { color:#FBEBD2 !important; font-size:1.1rem; }

    .insight-card { background:white; border-radius:22px; padding:1.35rem; margin-bottom:1rem; box-shadow:0 12px 28px rgba(63,33,19,.10); border:1px solid rgba(58,26,13,.10); }
    .insight-title { color:#241006; font-family:'Montserrat',sans-serif; font-weight:900; font-size:1.25rem; margin-bottom:.75rem; }
    .insight-row { color:#2E2018; font-size:1.02rem; line-height:1.6; margin:.5rem 0; }
    .tag { display:inline-block; color:#3A2200; background:#FFE08A; border-radius:999px; padding:.24rem .6rem; font-size:.78rem; font-weight:900; text-transform:uppercase; }
    .soft-card { background:#fff; border-left:6px solid var(--accent,#B87E00); border-radius:18px; padding:1.1rem 1.2rem; margin:.7rem 0; box-shadow:0 8px 20px rgba(63,33,19,.08); }
    .soft-card b { color:#241006; font-size:1.08rem; }
    .soft-card { color:#2E2018; font-size:1rem; line-height:1.55; }
    </style>
    """, unsafe_allow_html=True)

def section_header(eyebrow: str, title: str, copy: str = "") -> None:
    st.markdown(f"""<div class="section-head"><div><div class="section-eyebrow">{eyebrow}</div><div class="section-title">{title}</div></div><div class="section-copy">{copy}</div></div>""", unsafe_allow_html=True)

def style_figure(fig: go.Figure, height: int = 440) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope", color=INK, size=17),
        title=dict(font=dict(family="Montserrat", color=CHOCO_DARK, size=24), x=0.02, xanchor="left"),
        margin=dict(l=42, r=28, t=84, b=52),
        legend=dict(font=dict(size=16, color=INK), bgcolor="rgba(255,255,255,.65)"),
        uniformtext=dict(minsize=15, mode="show"),
    )
    fig.update_xaxes(
        tickfont=dict(size=16, color=INK), title_font=dict(size=17, color=INK),
        showline=True, linecolor="rgba(28,20,15,.35)", gridcolor="rgba(28,20,15,.10)"
    )
    fig.update_yaxes(
        tickfont=dict(size=16, color=INK), title_font=dict(size=17, color=INK),
        showline=True, linecolor="rgba(28,20,15,.35)", gridcolor="rgba(28,20,15,.10)"
    )
    return fig

def chart_rating_distribution(metrics: Dict[str, Any]) -> go.Figure:
    notas=list(range(1,6)); valores=[metrics["distribuicao"].get(n,0) for n in notas]
    pct=[metrics["percentual"].get(n,0) for n in notas]
    cores=["#B0241A","#C56A12","#B78A00","#3F8F3A","#166B2E"]
    rotulos=[f"<b>{v}</b><br>{p:.0f}%" for v, p in zip(valores, pct)]
    fig=go.Figure(go.Bar(
        x=[f"{n}★" for n in notas], y=valores,
        text=rotulos, textposition="outside", textfont=dict(size=18, color=INK, family="Montserrat"),
        marker=dict(color=cores, line=dict(color="rgba(28,20,15,.35)", width=1.2)),
        cliponaxis=False,
    ))
    fig.update_layout(title="Distribuição das avaliações (por nota)", yaxis_title="Nº de avaliações")
    return style_figure(fig)

def chart_time_evolution(ts: pd.DataFrame) -> go.Figure:
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["dia"], y=ts["media_nota"], mode="lines+markers", name="Média diária",
        line=dict(color=CHOCO_SOFT, width=3), marker=dict(size=8, color=CHOCO_SOFT)
    ))
    fig.add_trace(go.Scatter(
        x=ts["dia"], y=ts["media_movel_7d"], mode="lines", name="Média móvel 7 dias",
        line=dict(color=GOLD, width=5)
    ))
    fig.update_layout(
        title="Evolução da percepção (nota média ao longo do tempo)",
        yaxis_title="Nota média (1–5)", yaxis=dict(range=[1, 5.2]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return style_figure(fig,460)

def chart_sentiment_distribution(sentiment_counts: Dict[str, int]) -> go.Figure:
    labels=["Positivo","Negativo","Neutro","Misto"]
    values=[sentiment_counts.get("positivo",0),sentiment_counts.get("negativo",0),sentiment_counts.get("neutro",0),sentiment_counts.get("misto",0)]
    fig=go.Figure(go.Pie(
        labels=labels, values=values, hole=.58, sort=False,
        marker=dict(colors=[POS_COLOR,NEG_COLOR,NEU_COLOR,MIS_COLOR], line=dict(color="#FFFFFF", width=2.5)),
        texttemplate="<b>%{label}</b><br>%{value} (%{percent})",
        textposition="outside",
        textfont=dict(size=17, color=INK, family="Manrope"),
        insidetextfont=dict(size=17, color="white"),
        pull=[0.02,0.02,0.02,0.02],
    ))
    fig.update_layout(title="Distribuição de sentimento", showlegend=False)
    return style_figure(fig)

def _aspect_bar(aspectos: List[Dict[str, Any]], key: str, title: str, color: str) -> Optional[go.Figure]:
    data=sorted([a for a in aspectos if a.get(key,0)>0],key=lambda a:a[key],reverse=True)[:8]
    if not data:return None
    valores=[a[key] for a in data][::-1]
    nomes=[a["nome"].capitalize() for a in data][::-1]
    fig=go.Figure(go.Bar(
        x=valores, y=nomes, orientation="h",
        marker=dict(color=color, line=dict(color="rgba(28,20,15,.30)", width=1)),
        text=[f"<b>{v}</b>" for v in valores], textposition="outside",
        textfont=dict(size=17, color=INK, family="Montserrat"), cliponaxis=False,
    ))
    fig.update_layout(title=title, xaxis_title="Nº de menções")
    return style_figure(fig,440)

def chart_top_positive_aspects(aspectos): return _aspect_bar(aspectos,"positivas","O que mais elogiam",POS_COLOR)
def chart_top_negative_aspects(aspectos): return _aspect_bar(aspectos,"negativas","O que gera fricção",NEG_COLOR)

def chart_diverging_aspects(aspectos: List[Dict[str, Any]]) -> Optional[go.Figure]:
    data=sorted(aspectos,key=lambda a:a["mencoes"],reverse=True)[:8]
    data=[a for a in data if a["positivas"]+a["negativas"]>0]
    if not data:return None
    names=[a["nome"].capitalize() for a in data][::-1]
    neg=[a["negativas"] for a in data][::-1]
    pos=[a["positivas"] for a in data][::-1]
    fig=go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=[-v for v in neg], name="Negativas", orientation="h",
        marker=dict(color=NEG_COLOR, line=dict(color="rgba(28,20,15,.30)", width=1)),
        text=[f"<b>{v}</b>" if v else "" for v in neg], textposition="outside",
        textfont=dict(size=16, color=NEG_COLOR, family="Montserrat"),
    ))
    fig.add_trace(go.Bar(
        y=names, x=pos, name="Positivas", orientation="h",
        marker=dict(color=POS_COLOR, line=dict(color="rgba(28,20,15,.30)", width=1)),
        text=[f"<b>{v}</b>" if v else "" for v in pos], textposition="outside",
        textfont=dict(size=16, color=POS_COLOR, family="Montserrat"),
    ))
    fig.update_layout(
        title="Equilíbrio de percepção por aspecto", barmode="relative",
        xaxis_title="◄ Negativas    |    Positivas ►", cliponaxis=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return style_figure(fig,440)

def render_top_metrics(metrics: Dict[str, Any], sentiment_counts: Dict[str, int]) -> None:
    total=sum(sentiment_counts.values())
    pct_pos=round(sentiment_counts.get("positivo",0)/total*100,1) if total else metrics.get("pct_positivas",0.0)
    pct_neg=round(sentiment_counts.get("negativo",0)/total*100,1) if total else metrics.get("pct_negativas",0.0)
    media=f"{metrics.get('media'):.2f}".replace('.',',') if metrics.get('media') is not None else "—"
    st.markdown(f"""<div class="kpi-grid">
    <div class="kpi"><div class="kpi-label">Nota média</div><div class="kpi-value">{media}</div></div>
    <div class="kpi"><div class="kpi-label">Total</div><div class="kpi-value">{metrics.get('total_avaliacoes',0)}</div></div>
    <div class="kpi"><div class="kpi-label">Positivas</div><div class="kpi-value">{pct_pos:.1f}%</div></div>
    <div class="kpi"><div class="kpi-label">Negativas</div><div class="kpi-value">{pct_neg:.1f}%</div></div></div>""",unsafe_allow_html=True)

def render_insight_cards(insights: List[Dict[str, Any]]) -> None:
    cols=st.columns(2)
    for i,ins in enumerate(insights):
        with cols[i%2]:
            st.markdown(f"""<div class="insight-card"><div class="insight-title">{ins.get('titulo','Insight')}</div>
            <div class="insight-row"><span class="tag">Evidência</span> {ins.get('evidencia','—')}</div>
            <div class="insight-row"><span class="tag">Interpretação</span> {ins.get('interpretacao','—')}</div></div>""",unsafe_allow_html=True)

def _list_cards(items, kind):
    color=POS_COLOR if kind=="positive" else NEG_COLOR
    for p in items:
        st.markdown(f"""<div class="soft-card" style="--accent:{color}"><b>{p.get('aspecto','—')}</b><br>{p.get('resumo','')}</div>""",unsafe_allow_html=True)

def render_behavior_and_brand(synthesis, brand_counts):
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="panel"><div class="section-eyebrow">Comportamento</div><div class="section-title">Sinais do consumidor</div>',unsafe_allow_html=True)
        comp=synthesis.get("comportamento",{})
        for label,key in [("Recompra","recompra"),("Recomendação","recomendacao"),("Fidelidade","fidelidade")]: st.markdown(f"**{label}:** {comp.get(key,'—')}")
        st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><div class="section-eyebrow">Marca & Campanha</div><div class="section-title">Percepção de Marketing</div>',unsafe_allow_html=True)
        marca=synthesis.get("marca",{})
        st.markdown(f"**Opinião Geral da Marca:** {marca.get('percepcao','—')}")
        st.markdown(f"**Reação à Campanha/Propaganda:** {marca.get('opiniao_publica_campanha','Não houve menções suficientes sobre campanhas nos comentários analisados.')}")
        st.markdown('</div>',unsafe_allow_html=True)

# ==========================================================
# TESTE DE CONEXÃO
# ==========================================================

def test_connection(api_key: str, base_url: str, model: str) -> Tuple[bool, str]:
    try:
        client = _build_client(api_key, base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "responda apenas: ok"}],
        )
        modelo_ret = resp.model or model
        return True, f"✅ Conexão OK. Modelo que respondeu: {modelo_ret}"
    except Exception as exc:
        return False, f"❌ Falha na conexão: {exc}"

# ==========================================================
# ORQUESTRAÇÃO
# ==========================================================

def run_ai_analysis(api_key: str, base_url: str, model: str, df: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    comments = prepare_comments(df)
    if not comments: return None

    comment_results, avisos = analyze_batches(api_key, base_url, model, comments, BATCH_SIZE)

    if avisos:
        with st.expander(f"⚠️ {len(avisos)} aviso(s) durante a análise dos lotes"):
            for a in avisos:
                st.write("• " + a)

    if not comment_results:
        st.error(
            "Falha ao processar com a IA. Verifique: (1) a API Key do provedor escolhido; "
            "(2) se o nome do modelo está correto/disponível; (3) o rate limit do plano free."
        )
        return None

    agg = aggregate_batch_results(df, comment_results)
    ts = build_time_series(df)
    temporal_hint = "Sem dados temporais suficientes."
    if ts is not None and len(ts) >= 2:
        temporal_hint = f"Média inicial {ts['media_nota'].iloc[0]}, final {ts['media_nota'].iloc[-1]}."

    sample = comments[:80]

    try:
        raw_synthesis = synthesize_insights(api_key, base_url, model, metrics, {"aspectos": agg["aspectos"][:15]}, agg["sentiment_counts"], sample, temporal_hint)
        synthesis = validate_synthesis(raw_synthesis)
    except Exception as exc:
        st.warning(f"Erro na síntese executiva ({str(exc)}).")
        synthesis = validate_synthesis({})

    return {"synthesis": synthesis, "aggregation": agg, "temporal_hint": temporal_hint}

def main() -> None:
    inject_design_system()
    st.markdown("""<div class="hero"><div class="hero-kicker">Consumer analytics · IA gratuita</div><h1>Consumer Insights — Nescau</h1><p>Extração automática de pontos fortes, fracos e opinião sobre campanhas de marketing.</p></div>""",unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configurações IA")
        provider_name = st.selectbox("Provedor de IA (gratuito):", list(PROVIDERS.keys()))
        preset = PROVIDERS[provider_name]
        st.caption(preset["key_hint"])

        model = st.text_input("Modelo:", value=preset["model"],
                              help="Você pode trocar pelo ID de outro modelo do mesmo provedor.")
        user_api_key = st.text_input(f"Cole aqui sua API Key ({provider_name}):", type="password")

    base_url = preset["base_url"]
    # Nomes de env/secrets aceitos por provedor.
    env_map = {
        "Google Gemini (AI Studio)": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "Groq": ("GROQ_API_KEY",),
        "OpenRouter": ("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
    }
    api_key = user_api_key or get_api_key(env_map.get(provider_name, ()))

    # Botão de teste de conexão (não gasta os lotes).
    with st.sidebar:
        if st.button("🔌 Testar conexão", use_container_width=True):
            if not api_key:
                st.error("Cole a API Key primeiro.")
            else:
                ok, msg = test_connection(api_key, base_url, model)
                (st.success if ok else st.error)(msg)

    uploaded=st.file_uploader("Envie o CSV (data, 1-5, comentario)",type=["csv"])
    if uploaded is None: return

    df_raw,load_err=load_data(uploaded)
    if load_err: st.error(load_err); return

    ok_csv, msgs = validate_csv(df_raw)
    if not ok_csv:
        for m in msgs: st.error(m)
        return

    df, quality = clean_dataframe(df_raw)

    metrics = compute_metrics(df)
    section_header("Visão geral","O pulso da experiência")
    render_top_metrics(metrics, {})

    c1,c2=st.columns([1,1.45])
    with c1: st.plotly_chart(chart_rating_distribution(metrics),use_container_width=True)
    with c2:
        ts = build_time_series(df)
        if ts is not None: st.plotly_chart(chart_time_evolution(ts),use_container_width=True)

    st.markdown('<div class="ai-shell"><div class="ai-inner"><div class="ai-title" style="color:#FFFFFF;">AI Consumer Insights</div><p class="ai-copy">Organiza e descobre o que os clientes acharam do produto e das campanhas.</p></div></div>',unsafe_allow_html=True)

    if not api_key:
        st.error("👈 Cole a API Key do provedor escolhido no menu lateral para liberar as funções da IA.")
        return

    if not st.button(f"Gerar análise inteligente com {provider_name} 🚀",use_container_width=False): return

    with st.spinner(f"Analisando com {provider_name} ({model})..."):
        result=run_ai_analysis(api_key,base_url,model,df,metrics)
    if result is None: return

    synthesis=result["synthesis"]; agg=result["aggregation"]; sentiment_counts=agg["sentiment_counts"]; aspectos=synthesis.get("aspectos") or agg["aspectos"]

    section_header("Sentimento dos consumidores","Como a percepção se distribui")
    a,b=st.columns([.85,1.35])
    with a: st.plotly_chart(chart_sentiment_distribution(sentiment_counts),use_container_width=True)
    with b:
        fig=chart_diverging_aspects(aspectos)
        if fig: st.plotly_chart(fig,use_container_width=True)

    section_header("Drivers de percepção","O que encanta e o que pode melhorar")
    p,n=st.columns(2)
    with p:
        fig=chart_top_positive_aspects(aspectos)
        if fig: st.plotly_chart(fig,use_container_width=True)
        _list_cards(synthesis.get("pontos_positivos",[]), "positive")
    with n:
        fig=chart_top_negative_aspects(aspectos)
        if fig: st.plotly_chart(fig,use_container_width=True)
        _list_cards(synthesis.get("pontos_negativos",[]), "negative")

    section_header("Contexto estratégico","Marketing e Marca")
    render_behavior_and_brand(synthesis,agg["marca"])

    section_header("Voz do consumidor","Insights Encontrados")
    render_insight_cards(synthesis.get("insights",[]))


if __name__ == "__main__":
    main()

