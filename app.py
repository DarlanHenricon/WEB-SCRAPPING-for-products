"""
Nescau — Consumer Insights
==========================================================
Aplicativo Streamlit para análise de Consumer Insights de
avaliações do produto Nescau.

O Pandas cuida das análises quantitativas (locais).
A API da OpenAI cuida da interpretação semântica dos comentários
(sentimento por aspecto, temas, elogios, reclamações, oportunidades,
comportamento do consumidor, percepção de marca e insights executivos).

Execução:
    streamlit run app.py

Requisitos de ambiente:
    A chave da OpenAI DEVE ser fornecida via variável de ambiente
    OPENAI_API_KEY ou via Streamlit Secrets. NUNCA no código.
==========================================================
"""

from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================================
# CONFIGURAÇÕES GERAIS
# ==========================================================

APP_TITLE = "Nescau — Consumer Insights"
APP_SUBTITLE = "Como os consumidores percebem o produto?"

# Modelo da OpenAI. Pode ser sobrescrito por variável de ambiente.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Tamanho do lote de comentários enviado por chamada de API.
BATCH_SIZE = int(os.getenv("NESCAU_BATCH_SIZE", "40"))

# Limite defensivo de caracteres por comentário (evita payloads gigantes).
MAX_COMMENT_CHARS = 600

# Colunas esperadas no CSV.
COL_DATA = "data"
COL_NOTA = "1-5"
COL_COMENTARIO = "comentario"
REQUIRED_COLUMNS = [COL_DATA, COL_NOTA, COL_COMENTARIO]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================================
# 1. CARREGAR DADOS
# ==========================================================

def load_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Lê o CSV enviado tentando diferentes separadores e encodings.

    Retorna (DataFrame, None) em caso de sucesso,
    ou (None, mensagem_de_erro) em caso de falha.
    """
    if uploaded_file is None:
        return None, "Nenhum arquivo foi enviado."

    raw = uploaded_file.getvalue()
    if not raw or len(raw.strip()) == 0:
        return None, "O arquivo enviado está vazio."

    # Tentativas de leitura combinando separadores e encodings comuns.
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
                dtype=str,          # lemos tudo como texto e convertemos depois
                keep_default_na=False,
                engine="python",
            )
            # Heurística: leitura válida deve ter mais de 1 coluna reconhecida
            if df.shape[1] >= 2:
                df.columns = [str(c).strip().lower() for c in df.columns]
                return df, None
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            continue

    return None, (
        "Não foi possível interpretar o arquivo como CSV. "
        f"Verifique o separador e o encoding. Detalhe técnico: {last_error}"
    )


# ==========================================================
# 2. VALIDAR CSV
# ==========================================================

def validate_csv(df: Optional[pd.DataFrame]) -> Tuple[bool, List[str]]:
    """Valida a presença das colunas obrigatórias e a existência de dados.

    Retorna (is_valid, lista_de_mensagens).
    """
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
    """Limpa e normaliza o DataFrame validado.

    - Converte notas para inteiros de 1 a 5 (valores inválidos viram NaN).
    - Converte datas (valores inválidos viram NaT).
    - Remove comentários totalmente vazios.

    Retorna (df_limpo, relatorio_de_qualidade).
    """
    report = {
        "linhas_originais": len(df),
        "notas_invalidas": 0,
        "datas_invalidas": 0,
        "comentarios_vazios": 0,
    }

    work = df.copy()

    # --- Nota (1-5) ---
    work[COL_NOTA] = pd.to_numeric(work[COL_NOTA], errors="coerce")
    # Somente notas inteiras entre 1 e 5 são válidas.
    valid_mask = work[COL_NOTA].between(1, 5)
    report["notas_invalidas"] = int((~valid_mask).sum())
    work.loc[~valid_mask, COL_NOTA] = pd.NA
    work[COL_NOTA] = work[COL_NOTA].round().astype("Int64")

    # --- Data ---
    work[COL_DATA] = pd.to_datetime(
        work[COL_DATA], errors="coerce", dayfirst=True
    )
    report["datas_invalidas"] = int(work[COL_DATA].isna().sum())

    # --- Comentário ---
    work[COL_COMENTARIO] = work[COL_COMENTARIO].astype(str).str.strip()
    empty_mask = work[COL_COMENTARIO].isin(["", "nan", "none", "null"])
    report["comentarios_vazios"] = int(empty_mask.sum())
    work = work[~empty_mask].reset_index(drop=True)

    return work, report


# ==========================================================
# 4 / 3. CALCULAR MÉTRICAS (LOCAL, PANDAS)
# ==========================================================

def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula todas as métricas quantitativas localmente com Pandas."""
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

    # Distribuição e percentual por nota (1 a 5).
    for n in range(1, 6):
        qtd = int((notas == n).sum())
        metrics["distribuicao"][n] = qtd
        metrics["percentual"][n] = (
            round(qtd / total_com_nota * 100, 1) if total_com_nota else 0.0
        )

    # Percentual de positivas (4-5) e negativas (1-2) com base na nota.
    positivas = int((notas >= 4).sum())
    negativas = int((notas <= 2).sum())
    metrics["pct_positivas"] = (
        round(positivas / total_com_nota * 100, 1) if total_com_nota else 0.0
    )
    metrics["pct_negativas"] = (
        round(negativas / total_com_nota * 100, 1) if total_com_nota else 0.0
    )

    return metrics


def build_time_series(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Constrói a série temporal (por dia) com contagem e média das notas.

    Retorna None se não houver datas válidas suficientes.
    """
    valid = df.dropna(subset=[COL_DATA]).copy()
    if valid.empty:
        return None

    valid["dia"] = valid[COL_DATA].dt.date
    grouped = (
        valid.groupby("dia")
        .agg(
            qtd_avaliacoes=(COL_COMENTARIO, "count"),
            media_nota=(COL_NOTA, "mean"),
        )
        .reset_index()
        .sort_values("dia")
    )
    grouped["media_nota"] = grouped["media_nota"].round(2)
    # Média móvel de 7 dias para suavizar a leitura da tendência.
    grouped["media_movel_7d"] = (
        grouped["media_nota"].rolling(window=7, min_periods=1).mean().round(2)
    )
    return grouped


def build_period_series(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Agrega avaliações por período (mês) para leitura macro."""
    valid = df.dropna(subset=[COL_DATA]).copy()
    if valid.empty:
        return None

    valid["periodo"] = valid[COL_DATA].dt.to_period("M").astype(str)
    grouped = (
        valid.groupby("periodo")
        .agg(
            qtd_avaliacoes=(COL_COMENTARIO, "count"),
            media_nota=(COL_NOTA, "mean"),
        )
        .reset_index()
        .sort_values("periodo")
    )
    grouped["media_nota"] = grouped["media_nota"].round(2)
    return grouped


# ==========================================================
# 3 / 4. PREPARAR COMENTÁRIOS (LOTES)
# ==========================================================

def prepare_comments(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Prepara a lista de comentários enviados à IA.

    Cada item preserva um id local (índice), a nota e o texto truncado.
    """
    items: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        texto = str(row[COL_COMENTARIO]).strip()
        if not texto:
            continue
        if len(texto) > MAX_COMMENT_CHARS:
            texto = texto[:MAX_COMMENT_CHARS] + "…"
        nota = row[COL_NOTA]
        items.append(
            {
                "id": int(idx),
                "nota": int(nota) if pd.notna(nota) else None,
                "comentario": texto,
            }
        )
    return items


def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    """Divide uma lista em lotes de tamanho `size`."""
    return [items[i : i + size] for i in range(0, len(items), size)]


# ==========================================================
# 5. PROMPTS DA IA
# ==========================================================

SYSTEM_PROMPT = (
    "Você é um analista sênior de Consumer Insights especializado em "
    "produtos de grande consumo (FMCG). Analisa avaliações de "
    "consumidores do achocolatado Nescau com rigor metodológico. "
    "Você distingue claramente DADO (o que foi observado), "
    "INTERPRETAÇÃO (o significado possível) e RECOMENDAÇÃO (ação sugerida). "
    "Você faz análise de sentimento POR ASPECTO e nunca reduz um comentário "
    "com sentimentos mistos a um único rótulo. Você NUNCA inventa "
    "informações para preencher lacunas e prioriza temas recorrentes em "
    "vez de opiniões isoladas. Você diferencia percepção do PRODUTO da "
    "percepção da MARCA. Você não afirma causalidade em análises temporais. "
    "Responde SEMPRE e exclusivamente em JSON válido, sem texto extra."
)

# Prompt do estágio 1: classificação por lote (nível de comentário).
BATCH_INSTRUCTION = """
Analise os comentários abaixo (formato JSON: lista de objetos com id, nota e comentario).

Para CADA comentário, retorne um objeto com:
- "id": o mesmo id recebido
- "sentimento_geral": um de ["positivo","negativo","neutro","misto"]
- "intensidade": um de ["baixa","media","alta"]
- "aspectos": lista de objetos {"aspecto": "<nome curto padronizado>", "sentimento": "positivo|negativo|neutro"}
  Exemplos de aspectos: sabor, docura, cremosidade, textura, aroma, qualidade,
  ingredientes, preco, custo-beneficio, embalagem, quantidade, praticidade,
  variedade, inovacao, disponibilidade, experiencia. Crie novos aspectos se necessário.
- "marca": {"mencionada": true|false, "sentimento": "positivo|negativo|neutro|na"}
  (considere marca/empresa/confiança/reputação/propaganda; produto != marca)
- "comportamento": lista com zero ou mais de
  ["recompra","recomendacao","fidelidade","abandono","comparacao_concorrente",
   "surpresa_positiva","decepcao","expectativa_superada","expectativa_nao_atendida"]

Regras:
- Não classifique como puramente positivo/negativo comentários com sentimentos mistos.
- Padronize nomes de aspectos (minúsculo, sem acento quando possível).
- Não invente aspectos que não estão no texto.

Retorne JSON no formato EXATO:
{"resultados": [ { ... um objeto por comentário ... } ]}
"""

# Prompt do estágio 2: síntese executiva (nível agregado).
SYNTHESIS_INSTRUCTION = """
Você recebe (1) estatísticas quantitativas já calculadas e (2) uma amostra
representativa de comentários. Produza uma análise agregada de Consumer Insights.

Baseie-se EXCLUSIVAMENTE nas evidências fornecidas. Priorize temas recorrentes.
Não invente dados. Diferencie produto de marca. Não afirme causalidade temporal.

Retorne JSON no formato EXATO abaixo (preencha todos os campos):
{
  "sentimento": {"positivo": 0, "negativo": 0, "neutro": 0, "misto": 0},
  "aspectos": [
    {"nome": "Sabor", "mencoes": 0, "positivas": 0, "negativas": 0,
     "sentimento_predominante": "positivo", "resumo": ""}
  ],
  "pontos_positivos": [{"aspecto": "", "mencoes": 0, "resumo": ""}],
  "pontos_negativos": [{"aspecto": "", "mencoes": 0, "resumo": ""}],
  "oportunidades": [{"tema": "", "problema": "", "oportunidade": "", "evidencia": ""}],
  "comportamento": {"recompra": "", "recomendacao": "", "fidelidade": "", "risco_abandono": ""},
  "marca": {"percepcao": "", "principais_pontos": []},
  "insights": [{"titulo": "", "evidencia": "", "interpretacao": "", "importancia": ""}]
}

Regras adicionais:
- Gere de 3 a 6 insights executivos, específicos e baseados em evidências.
- Se não houver menções suficientes à marca, defina "marca.percepcao" como:
  "Não foram encontradas menções suficientes à marca para uma conclusão confiável."
- Cada oportunidade deve separar claramente o problema observado da oportunidade.
- Se a sugestão vier do próprio consumidor, deixe isso explícito na "evidencia".
"""


# ==========================================================
# 3 / 20. CHAVE DE API
# ==========================================================

def get_api_key() -> Optional[str]:
    """Obtém a chave da OpenAI de st.secrets ou variável de ambiente.

    Nunca há chave escrita no código.
    """
    # 1) Streamlit Secrets
    try:
        if "OPENAI_API_KEY" in st.secrets:
            key = str(st.secrets["OPENAI_API_KEY"]).strip()
            if key:
                return key
    except Exception:  # secrets pode não existir; ignoramos com segurança
        pass

    # 2) Variável de ambiente
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return key or None


# ==========================================================
# 6. CHAMAR A API (COM CACHE E LOTES)
# ==========================================================

def _call_openai_json(api_key: str, system_prompt: str, user_content: str) -> Dict[str, Any]:
    """Faz uma chamada à OpenAI forçando resposta em JSON.

    Levanta exceção em caso de erro de conexão/API. Retorna dict.
    """
    from openai import OpenAI  # import tardio: só quando há chave

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    content = response.choices[0].message.content
    return safe_json_loads(content)


@st.cache_data(show_spinner=False)
def analyze_batches(
    api_key: str, comments: List[Dict[str, Any]], batch_size: int
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Estágio 1 — classifica comentários por lote.

    Cacheado: mesma combinação (chave + comentários) não repete chamadas.
    Retorna (resultados_por_comentario, avisos).
    """
    resultados: List[Dict[str, Any]] = []
    avisos: List[str] = []

    batches = chunk_list(comments, batch_size)
    progress = st.progress(0.0, text="Interpretando comentários com IA…")

    for i, batch in enumerate(batches):
        payload = json.dumps(batch, ensure_ascii=False)
        user_content = BATCH_INSTRUCTION + "\n\nComentários:\n" + payload
        try:
            data = _call_openai_json(api_key, SYSTEM_PROMPT, user_content)
            lote_res = data.get("resultados", [])
            if isinstance(lote_res, list):
                resultados.extend(lote_res)
            else:
                avisos.append(f"Lote {i + 1}: formato inesperado, ignorado.")
        except Exception as exc:  # noqa: BLE001
            avisos.append(f"Lote {i + 1}: falha na análise ({type(exc).__name__}).")
        progress.progress((i + 1) / len(batches), text="Interpretando comentários com IA…")

    progress.empty()
    return resultados, avisos


@st.cache_data(show_spinner=False)
def synthesize_insights(
    api_key: str,
    metrics: Dict[str, Any],
    aspect_agg: Dict[str, Any],
    sentiment_counts: Dict[str, int],
    sample_comments: List[Dict[str, Any]],
    temporal_hint: str,
) -> Dict[str, Any]:
    """Estágio 2 — síntese executiva agregada."""
    context = {
        "estatisticas_quantitativas": metrics,
        "sentimento_agregado": sentiment_counts,
        "aspectos_agregados": aspect_agg,
        "sinal_temporal": temporal_hint,
        "amostra_comentarios": sample_comments,
    }
    user_content = (
        SYNTHESIS_INSTRUCTION
        + "\n\nDados de contexto (JSON):\n"
        + json.dumps(context, ensure_ascii=False)
    )
    return _call_openai_json(api_key, SYSTEM_PROMPT, user_content)


# ==========================================================
# 7. PROCESSAR RESPOSTA
# ==========================================================

def safe_json_loads(content: Optional[str]) -> Dict[str, Any]:
    """Interpreta uma string como JSON de forma tolerante.

    Tenta parse direto; se falhar, extrai o primeiro bloco {...}.
    Levanta ValueError se nada válido for encontrado.
    """
    if not content:
        raise ValueError("Resposta vazia da API.")

    content = content.strip()
    # Remove cercas de código eventuais (```json ... ```).
    content = re.sub(r"^```(?:json)?", "", content).strip()
    content = re.sub(r"```$", "", content).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError("A API retornou um JSON inválido.")


def aggregate_batch_results(
    df: pd.DataFrame, comment_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Agrega os resultados por comentário em estruturas quantitativas.

    Também anexa 'sentimento' e 'aspecto_principal' ao DataFrame (por id).
    Retorna dicionário com contagens de sentimento, aspectos, marca e comportamento.
    """
    sentiment_counts = {"positivo": 0, "negativo": 0, "neutro": 0, "misto": 0}
    aspect_stats: Dict[str, Dict[str, int]] = {}
    brand_mentions = {"positivo": 0, "negativo": 0, "neutro": 0}
    behavior_counts: Dict[str, int] = {}

    # Mapas por id para anexar ao DataFrame.
    map_sentimento: Dict[int, str] = {}
    map_aspecto: Dict[int, str] = {}

    for res in comment_results:
        if not isinstance(res, dict):
            continue
        cid = res.get("id")
        sent = str(res.get("sentimento_geral", "")).lower()
        if sent in sentiment_counts:
            sentiment_counts[sent] += 1
        if cid is not None:
            map_sentimento[cid] = sent or "n/d"

        aspectos = res.get("aspectos", []) or []
        aspecto_principal = None
        for a in aspectos:
            if not isinstance(a, dict):
                continue
            nome = str(a.get("aspecto", "")).strip().lower()
            if not nome:
                continue
            asent = str(a.get("sentimento", "neutro")).lower()
            bucket = aspect_stats.setdefault(
                nome, {"mencoes": 0, "positivas": 0, "negativas": 0, "neutras": 0}
            )
            bucket["mencoes"] += 1
            if asent == "positivo":
                bucket["positivas"] += 1
            elif asent == "negativo":
                bucket["negativas"] += 1
            else:
                bucket["neutras"] += 1
            if aspecto_principal is None:
                aspecto_principal = nome
        if cid is not None:
            map_aspecto[cid] = aspecto_principal or "n/d"

        # Marca
        marca = res.get("marca", {}) or {}
        if isinstance(marca, dict) and marca.get("mencionada"):
            msent = str(marca.get("sentimento", "neutro")).lower()
            if msent in brand_mentions:
                brand_mentions[msent] += 1

        # Comportamento
        for b in res.get("comportamento", []) or []:
            key = str(b).lower().strip()
            if key:
                behavior_counts[key] = behavior_counts.get(key, 0) + 1

    # Anexa colunas ao DataFrame (por índice/id).
    df["sentimento_ia"] = df.index.map(lambda i: map_sentimento.get(i, "n/d"))
    df["aspecto_principal"] = df.index.map(lambda i: map_aspecto.get(i, "n/d"))

    # Ordena aspectos por número de menções.
    aspectos_ordenados = [
        {
            "nome": nome,
            "mencoes": v["mencoes"],
            "positivas": v["positivas"],
            "negativas": v["negativas"],
            "neutras": v["neutras"],
        }
        for nome, v in sorted(
            aspect_stats.items(), key=lambda kv: kv[1]["mencoes"], reverse=True
        )
    ]

    return {
        "sentiment_counts": sentiment_counts,
        "aspectos": aspectos_ordenados,
        "marca": brand_mentions,
        "comportamento": behavior_counts,
    }


def validate_synthesis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Garante que o JSON de síntese tenha as chaves esperadas (com defaults)."""
    defaults = {
        "sentimento": {"positivo": 0, "negativo": 0, "neutro": 0, "misto": 0},
        "aspectos": [],
        "pontos_positivos": [],
        "pontos_negativos": [],
        "oportunidades": [],
        "comportamento": {
            "recompra": "",
            "recomendacao": "",
            "fidelidade": "",
            "risco_abandono": "",
        },
        "marca": {"percepcao": "", "principais_pontos": []},
        "insights": [],
    }
    if not isinstance(data, dict):
        return defaults
    for key, default in defaults.items():
        if key not in data or data[key] is None:
            data[key] = default
    return data


# ==========================================================
# 8. DESIGN SYSTEM E GRÁFICOS (PLOTLY)
# ==========================================================

CHOCO_DARK = "#2A1208"
CHOCO = "#4A2112"
CHOCO_SOFT = "#6B3824"
GOLD = "#F5B800"
GOLD_LIGHT = "#FFD84D"
CREAM = "#FFF7E7"
INK = "#2B211D"
POS_COLOR = "#2E8B57"
NEG_COLOR = "#D8563F"
NEU_COLOR = "#A99E96"
MIS_COLOR = "#E7A72B"


def inject_design_system() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Montserrat:wght@700;800;900&display=swap');
    :root { --choco:#2A1208; --choco-2:#4A2112; --gold:#F5B800; --cream:#FFF7E7; --ink:#2B211D; }
    html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
    .stApp { background: radial-gradient(circle at 95% 2%, rgba(245,184,0,.12), transparent 26%), #FFF9EF; color:var(--ink); }
    .block-container { max-width:1280px; padding-top:1.2rem; padding-bottom:4rem; }
    header[data-testid="stHeader"] { background:transparent; }
    #MainMenu, footer { visibility:hidden; }
    h1,h2,h3 { font-family:'Montserrat',sans-serif !important; letter-spacing:-.035em; }
    h2 { color:var(--choco) !important; }
    [data-testid="stFileUploader"] { background:#fff; border:1px dashed rgba(74,33,18,.3); border-radius:22px; padding:1rem; box-shadow:0 12px 35px rgba(63,33,19,.08); }
    div[data-testid="stMetric"] { background:white; border:1px solid rgba(74,33,18,.08); border-radius:22px; padding:1.15rem 1.2rem; box-shadow:0 12px 30px rgba(63,33,19,.08); }
    div[data-testid="stMetricLabel"] { color:#725C50; font-weight:700; }
    div[data-testid="stMetricValue"] { color:var(--choco); font-family:'Montserrat',sans-serif; font-size:2rem; }
    div[data-testid="stPlotlyChart"] { background:white; border:1px solid rgba(74,33,18,.08); border-radius:24px; padding:.45rem; box-shadow:0 14px 36px rgba(63,33,19,.08); overflow:hidden; }
    [data-testid="stDataFrame"] { border-radius:18px; overflow:hidden; border:1px solid rgba(74,33,18,.1); }
    .stButton>button { border:0; border-radius:999px; min-height:3rem; padding:0 1.5rem; font-weight:800; color:#2A1208; background:linear-gradient(135deg,#FFD84D,#F5B800); box-shadow:0 10px 22px rgba(245,184,0,.25); transition:.2s ease; }
    .stButton>button:hover { transform:translateY(-2px); box-shadow:0 14px 28px rgba(245,184,0,.32); color:#2A1208; }
    .hero { position:relative; overflow:hidden; border-radius:34px; padding:3.2rem 3rem; color:white; background:linear-gradient(120deg,#241006 0%,#4A2112 58%,#6D371F 100%); box-shadow:0 24px 55px rgba(42,18,8,.24); margin-bottom:1.6rem; }
    .hero:before { content:""; position:absolute; width:420px; height:420px; border:82px solid rgba(245,184,0,.92); border-radius:48% 52% 60% 40%; right:-250px; top:-160px; transform:rotate(-18deg); }
    .hero:after { content:""; position:absolute; width:260px; height:48px; background:#F5B800; right:-35px; bottom:38px; transform:rotate(-13deg); border-radius:70px; opacity:.9; }
    .hero-kicker { display:inline-flex; border:1px solid rgba(255,255,255,.24); padding:.45rem .8rem; border-radius:999px; text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:800; color:#FFD84D; }
    .hero h1 { position:relative; z-index:1; margin:.8rem 0 .45rem; max-width:760px; font-size:clamp(2.5rem,5vw,4.9rem); line-height:.96; color:white !important; }
    .hero p { position:relative; z-index:1; max-width:700px; margin:0; color:#F7EBDD; font-size:1.05rem; line-height:1.65; }
    .hero-meta { position:relative; z-index:1; margin-top:1.5rem; display:flex; gap:.65rem; flex-wrap:wrap; }
    .hero-chip { background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.14); padding:.5rem .8rem; border-radius:999px; font-size:.78rem; font-weight:700; }
    .section-head { margin:2.3rem 0 .85rem; display:flex; gap:1rem; align-items:flex-end; justify-content:space-between; }
    .section-eyebrow { color:#A56A00; text-transform:uppercase; letter-spacing:.14em; font-weight:800; font-size:.72rem; }
    .section-title { margin:.15rem 0 0; color:#2A1208; font-family:'Montserrat',sans-serif; font-size:1.85rem; font-weight:900; }
    .section-copy { color:#806C61; font-size:.92rem; max-width:520px; }
    .kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:.2rem 0 1.2rem; }
    .kpi { position:relative; overflow:hidden; min-height:145px; background:#fff; border:1px solid rgba(74,33,18,.08); border-radius:24px; padding:1.2rem; box-shadow:0 14px 36px rgba(63,33,19,.08); }
    .kpi:after { content:""; position:absolute; width:75px; height:75px; border-radius:50%; background:var(--accent,#F5B800); opacity:.13; right:-18px; top:-20px; }
    .kpi-label { color:#7A655A; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; font-weight:800; }
    .kpi-value { margin:.36rem 0 .18rem; font-family:'Montserrat',sans-serif; font-size:2.25rem; font-weight:900; color:#2A1208; }
    .kpi-note { color:#9A877C; font-size:.78rem; }
    .panel { background:white; border:1px solid rgba(74,33,18,.08); border-radius:24px; padding:1.35rem; box-shadow:0 14px 36px rgba(63,33,19,.08); margin-bottom:1rem; }
    .ai-shell { padding:2px; border-radius:30px; background:linear-gradient(135deg,#F5B800,#FFDF68 38%,#6B3824); box-shadow:0 20px 48px rgba(63,33,19,.14); margin:1rem 0; }
    .ai-inner { border-radius:28px; padding:1.6rem; background:linear-gradient(145deg,#2A1208,#4A2112); color:white; }
    .ai-badge { color:#FFD84D; font-size:.72rem; text-transform:uppercase; letter-spacing:.14em; font-weight:900; }
    .ai-title { color:white; font-family:'Montserrat',sans-serif; font-size:2rem; font-weight:900; margin:.25rem 0; }
    .ai-copy { color:#E9D9CE; margin:0; }
    .insight-card { height:100%; background:white; border:1px solid rgba(74,33,18,.1); border-radius:22px; padding:1.25rem; box-shadow:0 12px 28px rgba(63,33,19,.07); margin-bottom:1rem; }
    .insight-title { color:#2A1208; font-family:'Montserrat',sans-serif; font-weight:900; font-size:1.05rem; margin-bottom:.75rem; }
    .insight-row { color:#5E4A40; font-size:.9rem; line-height:1.55; margin:.45rem 0; }
    .tag { display:inline-block; color:#654900; background:#FFF1B8; border-radius:999px; padding:.2rem .55rem; margin-right:.35rem; font-size:.68rem; font-weight:900; text-transform:uppercase; letter-spacing:.06em; }
    .soft-card { background:#fff; border-left:5px solid var(--accent,#F5B800); border-radius:18px; padding:1rem 1.1rem; margin:.65rem 0; box-shadow:0 10px 24px rgba(63,33,19,.06); }
    .empty-state { text-align:center; padding:2rem; color:#6E594E; }
    .method { background:#2A1208; color:#EEDFD3; border-radius:22px; padding:1.2rem 1.35rem; font-size:.82rem; line-height:1.6; margin-top:2rem; }
    @keyframes rise { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
    .hero,.kpi,.ai-shell { animation:rise .55s ease both; }
    @media(max-width:900px){ .kpi-grid{grid-template-columns:repeat(2,1fr)} .hero{padding:2.2rem 1.5rem} }
    @media(max-width:560px){ .kpi-grid{grid-template-columns:1fr} }
    </style>
    """, unsafe_allow_html=True)


def section_header(eyebrow: str, title: str, copy: str = "") -> None:
    st.markdown(f"""<div class="section-head"><div><div class="section-eyebrow">{eyebrow}</div><div class="section-title">{title}</div></div><div class="section-copy">{copy}</div></div>""", unsafe_allow_html=True)


def style_figure(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope", color=INK, size=12),
        title=dict(font=dict(family="Montserrat", color=CHOCO_DARK, size=20), x=.04, xanchor="left"),
        margin=dict(l=35, r=24, t=74, b=45), legend=dict(orientation="h", y=-.18),
        hoverlabel=dict(bgcolor=CHOCO_DARK, font_color="white", font_family="Manrope"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(74,33,18,.12)")
    fig.update_yaxes(gridcolor="rgba(74,33,18,.08)", zeroline=False)
    return fig


def chart_rating_distribution(metrics: Dict[str, Any]) -> go.Figure:
    notas=list(range(1,6)); valores=[metrics["distribuicao"].get(n,0) for n in notas]
    fig=go.Figure(go.Bar(x=[f"{n} estrelas" for n in notas],y=valores,text=valores,textposition="outside",marker=dict(color=["#D9C5B8","#BA8C70","#9B6040","#F6C946","#F5B800"],line=dict(width=0))))
    fig.update_layout(title="Distribuição das avaliações", showlegend=False)
    fig.update_yaxes(title="Quantidade")
    return style_figure(fig)


def chart_time_evolution(ts: pd.DataFrame) -> go.Figure:
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ts["dia"],y=ts["media_nota"],mode="lines+markers",name="Média diária",line=dict(color=CHOCO_SOFT,width=2),marker=dict(size=6),opacity=.55))
    fig.add_trace(go.Scatter(x=ts["dia"],y=ts["media_movel_7d"],mode="lines",name="Média móvel, 7 dias",line=dict(color=GOLD,width=4,shape="spline")))
    fig.update_layout(title="Evolução da percepção ao longo do tempo")
    fig.update_yaxes(title="Média das notas",range=[1,5])
    return style_figure(fig,440)


def chart_sentiment_distribution(sentiment_counts: Dict[str, int]) -> go.Figure:
    labels=["Positivo","Negativo","Neutro","Misto"]
    values=[sentiment_counts.get("positivo",0),sentiment_counts.get("negativo",0),sentiment_counts.get("neutro",0),sentiment_counts.get("misto",0)]
    fig=go.Figure(go.Pie(labels=labels,values=values,hole=.62,marker=dict(colors=[POS_COLOR,NEG_COLOR,NEU_COLOR,MIS_COLOR]),textinfo="percent+label",sort=False))
    fig.update_layout(title="Sentimento predominante",showlegend=False,annotations=[dict(text="Voz do<br>consumidor",x=.5,y=.5,showarrow=False,font=dict(size=13,color=CHOCO_DARK))])
    return style_figure(fig)


def _aspect_bar(aspectos: List[Dict[str, Any]], key: str, title: str, color: str) -> Optional[go.Figure]:
    data=sorted([a for a in aspectos if a.get(key,0)>0],key=lambda a:a[key],reverse=True)[:8]
    if not data:return None
    fig=go.Figure(go.Bar(x=[a[key] for a in data][::-1],y=[a["nome"].capitalize() for a in data][::-1],orientation="h",text=[a[key] for a in data][::-1],textposition="outside",marker_color=color))
    fig.update_layout(title=title,showlegend=False)
    return style_figure(fig,430)


def chart_top_positive_aspects(aspectos): return _aspect_bar(aspectos,"positivas","O que os consumidores mais elogiam",POS_COLOR)
def chart_top_negative_aspects(aspectos): return _aspect_bar(aspectos,"negativas","O que mais gera fricção",NEG_COLOR)


def chart_diverging_aspects(aspectos: List[Dict[str, Any]]) -> Optional[go.Figure]:
    data=sorted(aspectos,key=lambda a:a["mencoes"],reverse=True)[:8]
    data=[a for a in data if a["positivas"]+a["negativas"]>0]
    if not data:return None
    names=[a["nome"].capitalize() for a in data][::-1]
    fig=go.Figure()
    fig.add_trace(go.Bar(y=names,x=[-a["negativas"] for a in data][::-1],name="Negativas",orientation="h",marker_color=NEG_COLOR))
    fig.add_trace(go.Bar(y=names,x=[a["positivas"] for a in data][::-1],name="Positivas",orientation="h",marker_color=POS_COLOR))
    fig.update_layout(title="Equilíbrio de percepção por aspecto",barmode="relative")
    fig.update_xaxes(title="Menções negativas ←  |  → Menções positivas")
    return style_figure(fig,430)


# ==========================================================
# 9. RENDERIZAR DASHBOARD (COMPONENTES)
# ==========================================================

def render_header() -> None:
    st.markdown("""<div class="hero"><div class="hero-kicker">Consumer analytics · Data science · AI</div><h1>Consumer Insights — Nescau</h1><p>Transforme avaliações em uma leitura clara das percepções, tensões e oportunidades que movem a experiência do consumidor.</p><div class="hero-meta"><span class="hero-chip">Análise quantitativa local</span><span class="hero-chip">Sentimento por aspecto</span><span class="hero-chip">Síntese executiva por IA</span></div></div>""",unsafe_allow_html=True)


def render_top_metrics(metrics: Dict[str, Any], sentiment_counts: Dict[str, int]) -> None:
    total=sum(sentiment_counts.values())
    pct_pos=round(sentiment_counts.get("positivo",0)/total*100,1) if total else metrics.get("pct_positivas",0.0)
    pct_neg=round(sentiment_counts.get("negativo",0)/total*100,1) if total else metrics.get("pct_negativas",0.0)
    media=f"{metrics.get('media'):.2f}".replace('.',',') if metrics.get('media') is not None else "—"
    st.markdown(f"""<div class="kpi-grid">
    <div class="kpi" style="--accent:#F5B800"><div class="kpi-label">Nota média</div><div class="kpi-value">{media}<span style="font-size:1rem"> / 5</span></div><div class="kpi-note">Base: avaliações com nota válida</div></div>
    <div class="kpi" style="--accent:#6B3824"><div class="kpi-label">Total de avaliações</div><div class="kpi-value">{metrics.get('total_avaliacoes',0):,}</div><div class="kpi-note">Comentários disponíveis na base</div></div>
    <div class="kpi" style="--accent:#2E8B57"><div class="kpi-label">Avaliações positivas</div><div class="kpi-value">{pct_pos:.1f}%</div><div class="kpi-note">IA quando disponível, notas 4–5 como fallback</div></div>
    <div class="kpi" style="--accent:#D8563F"><div class="kpi-label">Avaliações negativas</div><div class="kpi-value">{pct_neg:.1f}%</div><div class="kpi-note">IA quando disponível, notas 1–2 como fallback</div></div></div>""".replace(',', '.'),unsafe_allow_html=True)


def render_insight_cards(insights: List[Dict[str, Any]]) -> None:
    if not insights:
        st.info("Nenhum insight foi gerado a partir dos dados disponíveis."); return
    cols=st.columns(2)
    for i,ins in enumerate(insights):
        with cols[i%2]:
            st.markdown(f"""<div class="insight-card"><div class="insight-title">{ins.get('titulo','Insight')}</div>
            <div class="insight-row"><span class="tag">Evidência</span>{ins.get('evidencia','—')}</div>
            <div class="insight-row"><span class="tag">Interpretação</span>{ins.get('interpretacao','—')}</div>
            <div class="insight-row"><span class="tag">Importância</span>{ins.get('importancia','—')}</div></div>""",unsafe_allow_html=True)


def _list_cards(items, kind):
    if not items:
        st.info("Não há evidências recorrentes suficientes para destacar."); return
    color=POS_COLOR if kind=="positive" else NEG_COLOR
    for p in items:
        st.markdown(f"""<div class="soft-card" style="--accent:{color}"><b>{p.get('aspecto','—').capitalize()}</b><br><span style="color:#7A655A;font-size:.82rem">{p.get('mencoes',0)} menções</span><br>{p.get('resumo','')}</div>""",unsafe_allow_html=True)


def render_strengths(items): _list_cards(items,"positive")
def render_attention_points(items): _list_cards(items,"negative")


def render_opportunities(items):
    if not items:
        st.info("Nenhuma oportunidade recorrente identificada com evidência suficiente."); return
    for o in items:
        st.markdown(f"""<div class="soft-card" style="--accent:#F5B800"><div class="insight-title">{o.get('tema','—').capitalize()}</div><div class="insight-row"><span class="tag">Problema observado</span>{o.get('problema','')}</div><div class="insight-row"><span class="tag">Oportunidade</span>{o.get('oportunidade','')}</div><div class="insight-row"><span class="tag">Evidência</span>{o.get('evidencia','—')}</div></div>""",unsafe_allow_html=True)


def render_behavior_and_brand(synthesis, brand_counts):
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="panel"><div class="section-eyebrow">Comportamento</div><div class="section-title" style="font-size:1.25rem">Sinais do consumidor</div>',unsafe_allow_html=True)
        comp=synthesis.get("comportamento",{})
        for label,key in [("Recompra","recompra"),("Recomendação","recomendacao"),("Fidelidade","fidelidade"),("Risco de abandono","risco_abandono")]: st.markdown(f"**{label}:** {comp.get(key,'—')}")
        st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><div class="section-eyebrow">Marca x produto</div><div class="section-title" style="font-size:1.25rem">Percepção de marca</div>',unsafe_allow_html=True)
        marca=synthesis.get("marca",{}); st.markdown(marca.get("percepcao","—"))
        for p in marca.get("principais_pontos",[]) or []: st.markdown(f"• {p}")
        st.caption(f"Menções à marca: {sum(brand_counts.values())} · positivas {brand_counts.get('positivo',0)} · negativas {brand_counts.get('negativo',0)} · neutras {brand_counts.get('neutro',0)}")
        st.markdown('</div>',unsafe_allow_html=True)


def render_final_table(df):
    table=df.copy(); table["Data"]=table[COL_DATA].dt.strftime("%d/%m/%Y").fillna("—")
    table=table.rename(columns={COL_NOTA:"Nota",COL_COMENTARIO:"Comentário","sentimento_ia":"Sentimento","aspecto_principal":"Aspecto principal"})
    table=table[["Data","Nota","Comentário","Sentimento","Aspecto principal"]]
    f1,f2,f3=st.columns(3)
    notas=sorted([int(n) for n in table["Nota"].dropna().unique()]); sel_n=f1.multiselect("Nota",notas,default=notas)
    sents=sorted(table["Sentimento"].dropna().unique().tolist()); sel_s=f2.multiselect("Sentimento",sents,default=sents)
    asps=sorted(table["Aspecto principal"].dropna().unique().tolist()); sel_a=f3.multiselect("Aspecto",asps,default=asps)
    filtered=table[table["Nota"].isin(sel_n)&table["Sentimento"].isin(sel_s)&table["Aspecto principal"].isin(sel_a)]
    st.dataframe(filtered,use_container_width=True,hide_index=True,height=420)
    st.caption(f"{len(filtered)} de {len(table)} avaliações exibidas.")

# ==========================================================
# ORQUESTRAÇÃO DA ANÁLISE COMPLETA
# ==========================================================

def run_ai_analysis(
    api_key: str, df: pd.DataFrame, metrics: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Executa os dois estágios de IA e agrega tudo. Retorna None em falha grave."""
    comments = prepare_comments(df)
    if not comments:
        st.error("Não há comentários válidos para analisar.")
        return None

    # Estágio 1 — classificação por lote (cacheado)
    comment_results, avisos = analyze_batches(api_key, comments, BATCH_SIZE)
    for aviso in avisos:
        st.warning(aviso)

    if not comment_results:
        st.error(
            "A IA não retornou nenhum resultado válido. "
            "Verifique sua conexão, a chave da API e tente novamente."
        )
        return None

    # Agregação local dos resultados por comentário
    agg = aggregate_batch_results(df, comment_results)

    # Sinal temporal textual (sem afirmar causalidade)
    ts = build_time_series(df)
    temporal_hint = "Sem dados temporais suficientes."
    if ts is not None and len(ts) >= 2:
        primeira = ts["media_nota"].iloc[0]
        ultima = ts["media_nota"].iloc[-1]
        tendencia = "estável"
        if ultima - primeira >= 0.3:
            tendencia = "tendência de alta na média"
        elif primeira - ultima >= 0.3:
            tendencia = "tendência de queda na média"
        temporal_hint = (
            f"Período de {ts['dia'].iloc[0]} a {ts['dia'].iloc[-1]}; "
            f"média inicial {primeira}, média final {ultima} ({tendencia})."
        )

    # Amostra representativa para a síntese (limita tokens)
    sample = comments[:120]

    # Estágio 2 — síntese executiva (cacheado)
    try:
        raw_synthesis = synthesize_insights(
            api_key,
            metrics,
            {"aspectos": agg["aspectos"][:15]},
            agg["sentiment_counts"],
            sample,
            temporal_hint,
        )
        synthesis = validate_synthesis(raw_synthesis)
    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"A síntese executiva por IA falhou ({type(exc).__name__}). "
            "Os resultados quantitativos e por comentário continuam disponíveis."
        )
        synthesis = validate_synthesis({})

    return {
        "synthesis": synthesis,
        "aggregation": agg,
        "temporal_hint": temporal_hint,
    }


def render_api_key_help() -> None:
    st.error("Chave da OpenAI não encontrada.")
    st.markdown("""Configure `OPENAI_API_KEY` como variável de ambiente ou no arquivo `.streamlit/secrets.toml`. Nunca escreva a chave diretamente no código.

```toml
OPENAI_API_KEY = "sua-chave-aqui"
```""")


def main() -> None:
    inject_design_system()
    render_header()
    api_key=get_api_key()

    section_header("Comece pela base", "Carregue as avaliações", "O fluxo original de leitura, validação e limpeza do CSV foi preservado.")
    uploaded=st.file_uploader("Envie o CSV com as colunas data, 1-5 e comentario",type=["csv"],label_visibility="visible")
    if uploaded is None:
        st.markdown('<div class="empty-state"><b>Seu dashboard começa aqui.</b><br>Envie o CSV para revelar a visão geral, os sentimentos e os insights de IA.</div>',unsafe_allow_html=True)
        if api_key is None:
            with st.expander("Como configurar a chave da OpenAI"): render_api_key_help()
        return

    df_raw,load_err=load_data(uploaded)
    if load_err: st.error(load_err); return
    valid,msgs=validate_csv(df_raw)
    if not valid:
        for m in msgs: st.error(m)
        return
    df,quality=clean_dataframe(df_raw)
    if df.empty: st.error("Após a limpeza, não restaram comentários válidos para analisar."); return

    with st.expander("Qualidade dos dados",expanded=False):
        q1,q2,q3,q4=st.columns(4)
        q1.metric("Linhas originais",quality["linhas_originais"]); q2.metric("Notas inválidas",quality["notas_invalidas"])
        q3.metric("Datas inválidas",quality["datas_invalidas"]); q4.metric("Comentários vazios",quality["comentarios_vazios"])
        st.dataframe(df.head(10),use_container_width=True,hide_index=True)

    metrics=compute_metrics(df)
    section_header("Visão geral","O pulso da experiência","Indicadores centrais para leitura rápida durante a apresentação.")
    render_top_metrics(metrics,{})

    c1,c2=st.columns([1,1.45])
    with c1: st.plotly_chart(chart_rating_distribution(metrics),use_container_width=True,config={"displayModeBar":False})
    with c2:
        ts=build_time_series(df)
        if ts is not None: st.plotly_chart(chart_time_evolution(ts),use_container_width=True,config={"displayModeBar":False})
        else: st.info("Dados temporais insuficientes para exibir a evolução.")

    st.markdown('<div class="ai-shell"><div class="ai-inner"><div class="ai-badge">Inteligência aplicada</div><div class="ai-title">AI Consumer Insights</div><p class="ai-copy">A IA organiza a voz do consumidor por sentimento, aspecto, comportamento, marca e oportunidade, sem alterar as evidências de origem.</p></div></div>',unsafe_allow_html=True)
    if api_key is None:
        render_api_key_help(); return
    if not st.button("Gerar análise completa com IA",use_container_width=False):
        st.info("Os indicadores locais já estão disponíveis. Clique acima para gerar a camada semântica e executiva."); return

    with st.spinner("Analisando comentários e consolidando evidências..."):
        result=run_ai_analysis(api_key,df,metrics)
    if result is None: return
    synthesis=result["synthesis"]; agg=result["aggregation"]; sentiment_counts=agg["sentiment_counts"]; aspectos=synthesis.get("aspectos") or agg["aspectos"]

    section_header("Visão geral enriquecida","KPIs após análise semântica","Os percentuais passam a refletir a classificação da IA quando disponível.")
    render_top_metrics(metrics,sentiment_counts)

    section_header("Sentimento dos consumidores","Como a percepção se distribui","Leia o equilíbrio emocional e os aspectos que sustentam cada percepção.")
    a,b=st.columns([.85,1.35])
    with a: st.plotly_chart(chart_sentiment_distribution(sentiment_counts),use_container_width=True,config={"displayModeBar":False})
    with b:
        fig=chart_diverging_aspects(aspectos)
        if fig: st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        else: st.info("Dados insuficientes para o comparativo por aspecto.")

    section_header("Drivers de percepção","O que encanta e o que pode melhorar","Duas perspectivas complementares para orientar a conversa.")
    p,n=st.columns(2)
    with p:
        fig=chart_top_positive_aspects(aspectos)
        if fig: st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        section_header("Pontos positivos","O que os consumidores gostam")
        render_strengths(synthesis.get("pontos_positivos",[]))
    with n:
        fig=chart_top_negative_aspects(aspectos)
        if fig: st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        section_header("Pontos de atenção","O que pode melhorar")
        render_attention_points(synthesis.get("pontos_negativos",[]))

    section_header("Voz do consumidor","Principais percepções","Insights apresentados na estrutura original de evidência, interpretação e importância.")
    render_insight_cards(synthesis.get("insights",[]))

    section_header("Oportunidades","Da tensão à ação","Cada oportunidade preserva a separação entre problema observado, evidência e recomendação.")
    render_opportunities(synthesis.get("oportunidades",[]))

    section_header("Contexto estratégico","Comportamento e marca")
    render_behavior_and_brand(synthesis,agg["marca"])

    section_header("Sinal temporal","Leitura responsável da evolução")
    st.info("Observação sem afirmação de causalidade: "+result["temporal_hint"])

    section_header("Base explorável","Avaliações detalhadas","Use os filtros para navegar da síntese até a voz original do consumidor.")
    render_final_table(df)
    st.markdown('<div class="method"><b>Metodologia</b><br>DADO = observado nas avaliações · INTERPRETAÇÃO = leitura possível do padrão · RECOMENDAÇÃO = ação sugerida. Conclusões limitadas às evidências disponíveis. Análises quantitativas por Pandas; interpretação por IA via OpenAI.</div>',unsafe_allow_html=True)


if __name__ == "__main__":
    main()
