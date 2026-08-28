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
# 8. GERAR GRÁFICOS (PLOTLY)
# ==========================================================

NESCAU_GREEN = "#00693C"
NESCAU_YELLOW = "#FFCB05"
POS_COLOR = "#2E7D32"
NEG_COLOR = "#C62828"
NEU_COLOR = "#9E9E9E"
MIS_COLOR = "#F9A825"


def chart_rating_distribution(metrics: Dict[str, Any]) -> go.Figure:
    """Gráfico 1 — Distribuição das notas 1..5."""
    notas = list(range(1, 6))
    valores = [metrics["distribuicao"].get(n, 0) for n in notas]
    fig = px.bar(
        x=[f"{n} ⭐" for n in notas],
        y=valores,
        text=valores,
        labels={"x": "Nota", "y": "Quantidade"},
        color=[str(n) for n in notas],
        color_discrete_sequence=px.colors.sequential.Greens,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title="Como os consumidores avaliam o Nescau?",
        showlegend=False,
        margin=dict(t=60, b=20),
    )
    return fig


def chart_time_evolution(ts: pd.DataFrame) -> go.Figure:
    """Gráfico 2 — Evolução da média das avaliações ao longo do tempo."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts["dia"],
            y=ts["media_nota"],
            mode="lines+markers",
            name="Média diária",
            line=dict(color=NESCAU_GREEN, width=1.5),
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts["dia"],
            y=ts["media_movel_7d"],
            mode="lines",
            name="Média móvel (7 dias)",
            line=dict(color=NESCAU_YELLOW, width=3),
        )
    )
    fig.update_layout(
        title="Evolução da percepção dos consumidores",
        yaxis=dict(title="Média das notas", range=[1, 5]),
        xaxis_title="Data",
        margin=dict(t=60, b=20),
    )
    return fig


def chart_sentiment_distribution(sentiment_counts: Dict[str, int]) -> go.Figure:
    """Gráfico 3 — Distribuição dos sentimentos."""
    labels = ["Positivo", "Negativo", "Neutro", "Misto"]
    values = [
        sentiment_counts.get("positivo", 0),
        sentiment_counts.get("negativo", 0),
        sentiment_counts.get("neutro", 0),
        sentiment_counts.get("misto", 0),
    ]
    fig = px.pie(
        names=labels,
        values=values,
        color=labels,
        color_discrete_map={
            "Positivo": POS_COLOR,
            "Negativo": NEG_COLOR,
            "Neutro": NEU_COLOR,
            "Misto": MIS_COLOR,
        },
        hole=0.45,
    )
    fig.update_layout(title="Qual é o sentimento predominante?", margin=dict(t=60, b=20))
    return fig


def chart_top_positive_aspects(aspectos: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Gráfico 4 — Principais aspectos positivos."""
    data = [a for a in aspectos if a.get("positivas", 0) > 0]
    data = sorted(data, key=lambda a: a["positivas"], reverse=True)[:8]
    if not data:
        return None
    fig = px.bar(
        x=[a["positivas"] for a in data],
        y=[a["nome"].capitalize() for a in data],
        orientation="h",
        text=[a["positivas"] for a in data],
        color_discrete_sequence=[POS_COLOR],
        labels={"x": "Menções positivas", "y": "Aspecto"},
    )
    fig.update_layout(
        title="O que os consumidores mais elogiam?",
        yaxis=dict(autorange="reversed"),
        margin=dict(t=60, b=20),
    )
    return fig


def chart_top_negative_aspects(aspectos: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Gráfico 5 — Principais aspectos negativos."""
    data = [a for a in aspectos if a.get("negativas", 0) > 0]
    data = sorted(data, key=lambda a: a["negativas"], reverse=True)[:8]
    if not data:
        return None
    fig = px.bar(
        x=[a["negativas"] for a in data],
        y=[a["nome"].capitalize() for a in data],
        orientation="h",
        text=[a["negativas"] for a in data],
        color_discrete_sequence=[NEG_COLOR],
        labels={"x": "Menções negativas", "y": "Aspecto"},
    )
    fig.update_layout(
        title="O que mais incomoda os consumidores?",
        yaxis=dict(autorange="reversed"),
        margin=dict(t=60, b=20),
    )
    return fig


def chart_diverging_aspects(aspectos: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Gráfico 6 — Barras divergentes: positivo x negativo por aspecto."""
    data = sorted(aspectos, key=lambda a: a["mencoes"], reverse=True)[:8]
    data = [a for a in data if (a["positivas"] + a["negativas"]) > 0]
    if not data:
        return None

    nomes = [a["nome"].capitalize() for a in data]
    positivas = [a["positivas"] for a in data]
    negativas = [-a["negativas"] for a in data]  # negativo para divergir

    fig = go.Figure()
    fig.add_trace(
        go.Bar(y=nomes, x=positivas, name="Positivas", orientation="h", marker_color=POS_COLOR)
    )
    fig.add_trace(
        go.Bar(y=nomes, x=negativas, name="Negativas", orientation="h", marker_color=NEG_COLOR)
    )
    fig.update_layout(
        title="Percepção positiva x negativa por aspecto",
        barmode="relative",
        yaxis=dict(autorange="reversed"),
        xaxis_title="Menções (negativas ← 0 → positivas)",
        margin=dict(t=60, b=20),
    )
    return fig


# ==========================================================
# 9. RENDERIZAR DASHBOARD (COMPONENTES)
# ==========================================================

def render_header() -> None:
    st.markdown(f"# 🥛 {APP_TITLE}")
    st.markdown(f"##### {APP_SUBTITLE}")
    st.caption(
        "Ferramenta de Consumer Insights: análises quantitativas locais (Pandas) "
        "+ interpretação semântica por IA (OpenAI). "
        "Dados → Interpretação → Recomendação, sempre com base nas evidências."
    )
    st.divider()


def render_top_metrics(metrics: Dict[str, Any], sentiment_counts: Dict[str, int]) -> None:
    total_analisados = sum(sentiment_counts.values())
    pos = sentiment_counts.get("positivo", 0) + sentiment_counts.get("misto", 0) * 0
    # % positivas/negativas por sentimento (IA) quando disponível; senão por nota.
    if total_analisados > 0:
        pct_pos = round(sentiment_counts.get("positivo", 0) / total_analisados * 100, 1)
        pct_neg = round(sentiment_counts.get("negativo", 0) / total_analisados * 100, 1)
    else:
        pct_pos = metrics.get("pct_positivas", 0.0)
        pct_neg = metrics.get("pct_negativas", 0.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    media = metrics.get("media")
    c1.metric("⭐ Média da avaliação", f"{media:.2f}" if media is not None else "—")
    c2.metric("📝 Total de avaliações", metrics.get("total_avaliacoes", 0))
    c3.metric("👍 % positivas", f"{pct_pos:.1f}%")
    c4.metric("👎 % negativas", f"{pct_neg:.1f}%")
    c5.metric("💬 Comentários analisados", total_analisados)


def render_insight_cards(insights: List[Dict[str, Any]]) -> None:
    st.subheader("💡 O que os consumidores estão dizendo?")
    if not insights:
        st.info("Nenhum insight foi gerado a partir dos dados disponíveis.")
        return
    cols = st.columns(2)
    for i, ins in enumerate(insights):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{ins.get('titulo', 'Insight')}**")
                if ins.get("evidencia"):
                    st.markdown(f"🔎 **Evidência (dado):** {ins['evidencia']}")
                if ins.get("interpretacao"):
                    st.markdown(f"🧭 **Interpretação:** {ins['interpretacao']}")
                if ins.get("importancia"):
                    st.markdown(f"⭐ **Importância:** {ins['importancia']}")


def render_strengths(pontos_positivos: List[Dict[str, Any]]) -> None:
    st.subheader("❤️ Pontos fortes")
    if not pontos_positivos:
        st.info("Sem elogios recorrentes suficientes para destacar.")
        return
    for p in pontos_positivos:
        mencoes = p.get("mencoes", 0)
        st.markdown(
            f"- **{p.get('aspecto', '—').capitalize()}** "
            f"({mencoes} menções) — {p.get('resumo', '')}"
        )


def render_attention_points(pontos_negativos: List[Dict[str, Any]]) -> None:
    st.subheader("⚠️ Pontos de atenção")
    if not pontos_negativos:
        st.info("Sem reclamações recorrentes suficientes para destacar.")
        return
    for p in pontos_negativos:
        mencoes = p.get("mencoes", 0)
        st.markdown(
            f"- **{p.get('aspecto', '—').capitalize()}** "
            f"({mencoes} menções) — {p.get('resumo', '')}"
        )


def render_opportunities(oportunidades: List[Dict[str, Any]]) -> None:
    st.subheader("🚀 Oportunidades de melhoria")
    if not oportunidades:
        st.info("Nenhuma oportunidade recorrente identificada com evidência suficiente.")
        return
    for o in oportunidades:
        with st.container(border=True):
            st.markdown(f"**Tema:** {o.get('tema', '—').capitalize()}")
            st.markdown(f"🔴 **Reclamação observada (dado):** {o.get('problema', '')}")
            st.markdown(f"🟢 **Oportunidade (recomendação):** {o.get('oportunidade', '')}")
            if o.get("evidencia"):
                st.caption(f"Evidência: {o['evidencia']}")


def render_behavior_and_brand(synthesis: Dict[str, Any], brand_counts: Dict[str, int]) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧠 Comportamento do consumidor")
        comp = synthesis.get("comportamento", {})
        st.markdown(f"- **Recompra:** {comp.get('recompra', '—')}")
        st.markdown(f"- **Recomendação:** {comp.get('recomendacao', '—')}")
        st.markdown(f"- **Fidelidade:** {comp.get('fidelidade', '—')}")
        st.markdown(f"- **Risco de abandono:** {comp.get('risco_abandono', '—')}")
    with col2:
        st.subheader("🏷️ Marca x Produto")
        marca = synthesis.get("marca", {})
        st.markdown(f"**Percepção da marca:** {marca.get('percepcao', '—')}")
        pontos = marca.get("principais_pontos", []) or []
        for p in pontos:
            st.markdown(f"- {p}")
        total_marca = sum(brand_counts.values())
        st.caption(
            f"Menções à marca detectadas nos comentários: {total_marca} "
            f"(👍 {brand_counts.get('positivo', 0)} / "
            f"👎 {brand_counts.get('negativo', 0)} / "
            f"😐 {brand_counts.get('neutro', 0)})"
        )


def render_final_table(df: pd.DataFrame) -> None:
    st.subheader("📋 Avaliações detalhadas")

    table = df.copy()
    table["Data"] = table[COL_DATA].dt.strftime("%d/%m/%Y").fillna("—")
    table = table.rename(
        columns={
            COL_NOTA: "Nota",
            COL_COMENTARIO: "Comentário",
            "sentimento_ia": "Sentimento",
            "aspecto_principal": "Aspecto principal",
        }
    )
    cols_show = ["Data", "Nota", "Comentário", "Sentimento", "Aspecto principal"]
    table = table[cols_show]

    # Filtros
    f1, f2, f3 = st.columns(3)
    notas_disp = sorted([int(n) for n in table["Nota"].dropna().unique()])
    sel_notas = f1.multiselect("Filtrar por nota", notas_disp, default=notas_disp)
    sent_disp = sorted(table["Sentimento"].dropna().unique().tolist())
    sel_sent = f2.multiselect("Filtrar por sentimento", sent_disp, default=sent_disp)
    asp_disp = sorted(table["Aspecto principal"].dropna().unique().tolist())
    sel_asp = f3.multiselect("Filtrar por aspecto", asp_disp, default=asp_disp)

    filtered = table[
        table["Nota"].isin(sel_notas)
        & table["Sentimento"].isin(sel_sent)
        & table["Aspecto principal"].isin(sel_asp)
    ]
    st.dataframe(filtered, use_container_width=True, hide_index=True)
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


# ==========================================================
# FLUXO PRINCIPAL DO STREAMLIT
# ==========================================================

def render_api_key_help() -> None:
    st.error("🔑 Chave da OpenAI não encontrada.")
    st.markdown(
        """
        Para executar a análise por IA, configure sua chave da OpenAI de uma
        das formas abaixo. **Nunca** escreva a chave no código.

        **Opção A — Variável de ambiente (terminal):**
        ```bash
        export OPENAI_API_KEY="sua-chave-aqui"     # Linux / macOS
        setx OPENAI_API_KEY "sua-chave-aqui"       # Windows (reabra o terminal)
        streamlit run app.py
        ```

        **Opção B — Streamlit Secrets:**
        Crie o arquivo `.streamlit/secrets.toml` com:
        ```toml
        OPENAI_API_KEY = "sua-chave-aqui"
        ```
        > 🔒 Adicione `.streamlit/secrets.toml` ao seu `.gitignore`.
        """
    )


def main() -> None:
    render_header()

    api_key = get_api_key()

    # --- 19. Interface simples de upload ---
    uploaded = st.file_uploader(
        "📤 Envie o arquivo CSV de avaliações (colunas: data, 1-5, comentario)",
        type=["csv"],
    )

    if uploaded is None:
        st.info(
            "Envie um CSV com aproximadamente 500 avaliações. "
            "As colunas obrigatórias são **data**, **1-5** e **comentario**."
        )
        if api_key is None:
            with st.expander("⚙️ Como configurar a chave da OpenAI"):
                render_api_key_help()
        return

    # --- Carregamento e validação ---
    df_raw, load_err = load_data(uploaded)
    if load_err:
        st.error(f"❌ {load_err}")
        return

    is_valid, msgs = validate_csv(df_raw)
    if not is_valid:
        for m in msgs:
            st.error(f"❌ {m}")
        return

    df, quality = clean_dataframe(df_raw)

    if df.empty:
        st.error("❌ Após a limpeza, não restaram comentários válidos para analisar.")
        return

    # Relatório de qualidade dos dados
    with st.expander("🧹 Relatório de qualidade dos dados", expanded=False):
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Linhas originais", quality["linhas_originais"])
        q2.metric("Notas inválidas", quality["notas_invalidas"])
        q3.metric("Datas inválidas", quality["datas_invalidas"])
        q4.metric("Comentários vazios removidos", quality["comentarios_vazios"])

    # --- 4. Prévia dos dados ---
    st.subheader("👀 Prévia dos dados")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    # --- 5. Métricas básicas locais ---
    metrics = compute_metrics(df)
    st.subheader("📊 Métricas básicas (calculadas localmente)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total de avaliações", metrics["total_avaliacoes"])
    m2.metric("Média das notas", f"{metrics['media']:.2f}" if metrics["media"] else "—")
    m3.metric("Mediana", f"{metrics['mediana']:.1f}" if metrics["mediana"] else "—")
    m4.metric("% positivas (nota ≥ 4)", f"{metrics['pct_positivas']:.1f}%")

    # Gráficos quantitativos disponíveis antes da IA
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(chart_rating_distribution(metrics), use_container_width=True)
    ts = build_time_series(df)
    with g2:
        if ts is not None and len(ts) >= 2:
            st.plotly_chart(chart_time_evolution(ts), use_container_width=True)
        else:
            st.info("Sem datas válidas suficientes para o gráfico temporal.")

    st.divider()

    # --- 6/19. Botão de análise ---
    if api_key is None:
        render_api_key_help()
        return

    if st.button("🥛 Analisar Nescau", type="primary", use_container_width=True):
        st.session_state["run_analysis"] = True

    if not st.session_state.get("run_analysis"):
        st.info("Clique em **🥛 Analisar Nescau** para iniciar a análise semântica por IA.")
        return

    # --- Execução da análise por IA ---
    with st.spinner("Analisando a percepção dos consumidores..."):
        result = run_ai_analysis(api_key, df, metrics)

    if result is None:
        return

    synthesis = result["synthesis"]
    agg = result["aggregation"]
    sentiment_counts = agg["sentiment_counts"]
    aspectos = agg["aspectos"]

    st.success("✅ Análise concluída.")
    st.divider()

    # --- 15. Dashboard: métricas de topo ---
    render_top_metrics(metrics, sentiment_counts)
    st.divider()

    # --- 16. Gráficos ---
    st.subheader("📈 Visão analítica")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(chart_rating_distribution(metrics), use_container_width=True)
    with r1c2:
        if ts is not None and len(ts) >= 2:
            st.plotly_chart(chart_time_evolution(ts), use_container_width=True)
        else:
            st.info("Sem série temporal suficiente.")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(
            chart_sentiment_distribution(sentiment_counts), use_container_width=True
        )
    with r2c2:
        fig_div = chart_diverging_aspects(aspectos)
        if fig_div:
            st.plotly_chart(fig_div, use_container_width=True)
        else:
            st.info("Dados insuficientes para o comparativo por aspecto.")

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig_pos = chart_top_positive_aspects(aspectos)
        if fig_pos:
            st.plotly_chart(fig_pos, use_container_width=True)
        else:
            st.info("Sem aspectos positivos recorrentes.")
    with r3c2:
        fig_neg = chart_top_negative_aspects(aspectos)
        if fig_neg:
            st.plotly_chart(fig_neg, use_container_width=True)
        else:
            st.info("Sem aspectos negativos recorrentes.")

    st.divider()

    # --- 17. Área de insights ---
    render_insight_cards(synthesis.get("insights", []))
    st.divider()
    cstrength, cattention = st.columns(2)
    with cstrength:
        render_strengths(synthesis.get("pontos_positivos", []))
    with cattention:
        render_attention_points(synthesis.get("pontos_negativos", []))
    st.divider()
    render_opportunities(synthesis.get("oportunidades", []))
    st.divider()

    # --- 10/11. Comportamento e marca ---
    render_behavior_and_brand(synthesis, agg["marca"])
    st.divider()

    # --- 12. Nota temporal ---
    st.subheader("📅 Sinal temporal")
    st.caption(
        "Observação sobre a evolução (sem afirmação de causalidade): "
        + result["temporal_hint"]
    )
    st.divider()

    # --- 18. Tabela final ---
    render_final_table(df)

    # Rodapé metodológico
    st.divider()
    st.caption(
        "🧭 **Metodologia:** DADO = observado nas avaliações · "
        "INTERPRETAÇÃO = leitura possível do padrão · "
        "RECOMENDAÇÃO = ação sugerida. Conclusões limitadas às evidências "
        "disponíveis. Análises quantitativas por Pandas; interpretação por IA (OpenAI)."
    )


if __name__ == "__main__":
    main()
