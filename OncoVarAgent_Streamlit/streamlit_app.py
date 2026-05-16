import asyncio
import importlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR / "backend"
ANNOTATOR_DIR = APP_DIR / "oncokb-annotator"
DEFAULT_ONCOKB_ANNOTATOR_PATH = ANNOTATOR_DIR / "MafAnnotator.py"
UPLOAD_DIR = APP_DIR / "uploads"
DEPS_DIR = APP_DIR / ".deps"
DEFAULT_LLM_API_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-5.4"


st.set_page_config(
    page_title="OncoVarAgent Offline",
    page_icon="O",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    :root {
        --ova-indigo: #4f46e5;
        --ova-slate: #0f172a;
        --ova-muted: #64748b;
        --ova-border: #e2e8f0;
        --ova-bg: #f8fafc;
    }

    .stApp {
        background: var(--ova-bg);
        color: var(--ova-slate);
    }

    div[data-testid="collapsedControl"],
    button[kind="header"],
    header[data-testid="stHeader"] {
        z-index: 999;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--ova-border);
        background: #ffffff;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #475569;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 3.25rem;
        padding-bottom: 4rem;
        overflow: visible;
    }

    .ova-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        min-height: 3.25rem;
        margin-bottom: 2rem;
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
        overflow: visible;
    }

    .ova-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-weight: 800;
        font-size: 1.25rem;
        letter-spacing: 0;
    }

    .ova-logo {
        width: 2.15rem;
        height: 2.15rem;
        min-width: 2.15rem;
        border-radius: 8px;
        background: var(--ova-indigo);
        color: white;
        display: grid;
        place-items: center;
        font-weight: 800;
        line-height: 1;
        flex-shrink: 0;
        overflow: visible;
    }

    .ova-hero {
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .ova-hero h1 {
        margin: 0 0 0.65rem;
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.05;
        letter-spacing: 0;
    }

    .ova-hero p {
        margin: 0 auto;
        max-width: 760px;
        color: var(--ova-muted);
        font-size: 1.05rem;
        line-height: 1.7;
    }

    .ova-section-title {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin: 1.75rem 0 0.85rem;
        font-size: 1.05rem;
        font-weight: 800;
    }

    .ova-step {
        width: 1.95rem;
        height: 1.95rem;
        border-radius: 999px;
        display: inline-grid;
        place-items: center;
        background: #e2e8f0;
        color: #475569;
        font-size: 0.9rem;
        font-weight: 800;
    }

    .ova-step-primary {
        background: var(--ova-indigo);
        color: white;
    }

    .ova-kv-label {
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .ova-kv-value {
        color: var(--ova-slate);
        font-size: 1rem;
        font-weight: 650;
        line-height: 1.55;
        overflow-wrap: anywhere;
    }

    .ova-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border: 1px solid var(--ova-border);
        border-radius: 999px;
        padding: 0.32rem 0.7rem;
        margin: 0.2rem 0.25rem 0.2rem 0;
        font-size: 0.82rem;
        font-weight: 700;
        color: #475569;
        text-decoration: none !important;
        background: white;
    }

    .ova-chip:hover {
        border-color: var(--ova-indigo);
        color: var(--ova-indigo);
    }

    .ova-drug {
        border: 1px solid #e0e7ff;
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.55rem;
        font-weight: 650;
        line-height: 1.55;
    }

    .ova-small-muted {
        color: var(--ova-muted);
        font-size: 0.9rem;
    }

    .ova-evidence-grid {
        margin-top: 0.35rem;
    }

    .ova-footer {
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid var(--ova-border);
        text-align: center;
        color: var(--ova-muted);
        font-size: 0.88rem;
    }

    div[data-testid="stCodeBlock"] pre {
        max-height: 340px;
        overflow-y: auto;
        border-radius: 8px;
    }

    .stButton button,
    .stDownloadButton button {
        border-radius: 8px;
        font-weight: 750;
    }

    @media print {
        @page {
            margin: 1.2cm;
        }

        header[data-testid="stHeader"],
        section[data-testid="stSidebar"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        div[data-testid="stForm"],
        div[data-testid="stExpander"],
        iframe,
        button,
        .ova-topbar,
        .ova-hero,
        .ova-footer {
            display: none !important;
        }

        .stApp {
            background: white !important;
        }

        .block-container {
            max-width: none !important;
            padding: 0 !important;
        }

        div[data-testid="stVerticalBlock"] {
            gap: 0.7rem !important;
            overflow: visible !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            break-inside: auto;
            box-shadow: none !important;
            border-color: #cbd5e1 !important;
            overflow: visible !important;
        }

        div[data-testid="stMarkdownContainer"],
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
            overflow: visible !important;
            height: auto !important;
            max-height: none !important;
        }

        .ova-evidence-grid {
            display: block;
            clear: both;
            break-before: auto;
            margin-top: 1rem !important;
            padding-top: 0.2rem !important;
        }

        .ova-evidence-grid + div[data-testid="stHorizontalBlock"] {
            display: block !important;
            clear: both !important;
        }

        .ova-evidence-grid + div[data-testid="stHorizontalBlock"] > div {
            display: block !important;
            width: 100% !important;
            min-width: 100% !important;
            margin-bottom: 0.8rem !important;
        }

        .ova-evidence-grid + div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] {
            break-inside: avoid;
        }
    }
</style>
"""


def parse_list(value: Any, separator: str = ",") -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return []
    return [item.strip() for item in text.split(separator) if item.strip()]


def get_field(result: dict[str, Any], key: str, default: str = "N/A") -> str:
    value = result.get(key, default)
    if value is None or value == "":
        return default
    return str(value)


def safe_json(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def render_print_button() -> None:
    components.html(
        """
        <button
            onclick="window.parent.print()"
            style="
                width: 100%;
                height: 40px;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                background: #ffffff;
                color: #334155;
                font: 750 14px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                cursor: pointer;
            "
        >
            Export PDF
        </button>
        """,
        height=46,
    )


def tier_badge_html(tier: str) -> str:
    lower = tier.lower()
    if "1" in lower:
        styles = "background:#ecfdf5;color:#065f46;border-color:#a7f3d0;"
    elif "2" in lower:
        styles = "background:#eff6ff;color:#1e40af;border-color:#bfdbfe;"
    elif "3" in lower:
        styles = "background:#fffbeb;color:#92400e;border-color:#fde68a;"
    else:
        styles = "background:#f1f5f9;color:#334155;border-color:#e2e8f0;"
    return f'<span class="ova-chip" style="{styles}">{tier}</span>'


def render_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ova-topbar">
            <div class="ova-brand">
                <div class="ova-logo">O</div>
                <span>OncoVarAgent</span>
            </div>
            <div class="ova-small-muted">Offline Streamlit application</div>
        </div>
        <div class="ova-hero">
            <h1>Precision Oncology Evidence AI Engine</h1>
            <p>
                Uncover actionable therapeutic evidence for somatic variants lacking established guidance via an autonomous research agent.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_settings_signature(settings: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(settings.items()))


def apply_runtime_settings(settings: dict[str, str]) -> None:
    os.environ["ONCOKB_API_TOKEN"] = settings["ONCOKB_API_TOKEN"]
    os.environ["ONCOKB_ANNOTATOR_PATH"] = settings["ONCOKB_ANNOTATOR_PATH"]
    os.environ["LLM_API_TOKEN"] = settings["LLM_API_TOKEN"]
    os.environ["LLM_API_URL"] = settings["LLM_API_URL"]
    os.environ["LLM_MODEL"] = settings["LLM_MODEL"]
    os.environ["MULERUN_API"] = settings["LLM_API_TOKEN"]


def load_backend_module(settings_signature: tuple[tuple[str, str], ...]):
    if DEPS_DIR.exists() and str(DEPS_DIR) not in sys.path:
        sys.path.append(str(DEPS_DIR))

    if not BACKEND_DIR.exists():
        raise FileNotFoundError(f"Backend directory not found: {BACKEND_DIR}")

    backend_path = str(BACKEND_DIR)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    if "OncoVarAgent" in sys.modules:
        return importlib.reload(sys.modules["OncoVarAgent"])
    return importlib.import_module("OncoVarAgent")


def write_variant_input(gene: str, variant: str, cancer_type: str) -> Path:
    UPLOAD_DIR.mkdir(exist_ok=True)
    input_path = UPLOAD_DIR / f"{uuid.uuid4()}_variant_input.txt"
    input_path.write_text(
        "Hugo_Symbol\tHGVSp_Short\tCancer_Type\n"
        f"{gene}\t{variant}\t{cancer_type}\n",
        encoding="utf-8",
    )
    return input_path


async def run_local_agent_stream(
    input_path: Path,
    on_log,
    settings_signature: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    oncovar_agent = load_backend_module(settings_signature)

    async def send_react_log(log_entry: str):
        on_log(f"--- [ReAct Agent Log] ---\n{log_entry}")

    initial_state = {
        "patient_info": {
            "input_txt": str(input_path),
            "gene_col": "Hugo_Symbol",
            "protein_change_col": "HGVSp_Short",
            "cancer_type_col": "Cancer_Type",
        },
        "processed_variants_reports": [],
        "final_report": {},
    }
    run_config = {
        "recursion_limit": 20000,
        "configurable": {"react_log_callback": send_react_log},
    }

    final_state_result = None
    async for event in oncovar_agent.app.astream(initial_state, config=run_config):
        event_key, event_value = list(event.items())[0]
        if event_key == "deep_researcher":
            on_log(f"--- [Node: {event_key}] ---\nThe deep research node has been executed.")
        else:
            detail = safe_json(event_value) if event_value else "The node has been executed."
            on_log(f"--- [Node: {event_key}] ---\n{detail}")
        if event_key == "final_combiner":
            final_state_result = event_value

    if not final_state_result:
        raise RuntimeError("Workflow finished without a final_combiner result.")

    variant_reports = final_state_result.get("final_report", {}).get("variant_report", [])
    if not variant_reports:
        raise RuntimeError("Workflow finished, but no final variant report was generated.")

    return variant_reports[0]


def run_analysis(
    gene: str,
    variant: str,
    cancer_type: str,
    settings: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    logs: list[str] = ["Initializing local OncoVarAgent workflow..."]
    log_placeholder = st.empty()
    status_placeholder = st.empty()

    def refresh_logs() -> None:
        log_placeholder.code("\n".join(logs[-180:]), language="text")

    def append_log(message: str) -> None:
        logs.append(message)
        refresh_logs()

    refresh_logs()
    apply_runtime_settings(settings)
    settings_signature = build_settings_signature(settings)
    input_path = write_variant_input(gene, variant, cancer_type)

    try:
        with status_placeholder.status("Running local workflow...", expanded=True) as status:
            append_log(f"Created local input file: {input_path}")
            append_log("Loading backend module and starting LangGraph stream...")
            status.update(label="Agent is running locally...", state="running")
            result = asyncio.run(run_local_agent_stream(input_path, append_log, settings_signature))
            append_log("Analysis complete.")
            status.update(label="Analysis finished.", state="complete")
        log_placeholder.empty()
        status_placeholder.empty()
        return result, logs
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass


def render_baseline_guidance(result: dict[str, Any]) -> None:
    st.markdown(
        '<div class="ova-section-title"><span class="ova-step">1</span>Baseline Guidance (OncoKB)</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.markdown('<div class="ova-kv-label">Oncogenicity</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="ova-kv-value">{get_field(result, "oncokb_ONCOGENIC")}</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown('<div class="ova-kv-label">AMP Tier</div>', unsafe_allow_html=True)
            st.markdown(tier_badge_html(get_field(result, "oncokb_AMP_TIER")), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="ova-kv-label">Biological Effect</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="ova-kv-value">{get_field(result, "oncokb_MUTATION_EFFECT")}</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown('<div class="ova-kv-label">FDA/NCCN Standard Care</div>', unsafe_allow_html=True)
        oncokb_drugs = get_field(result, "oncokb_Drugs")
        if oncokb_drugs != "N/A":
            st.markdown(f'<div class="ova-kv-value">{oncokb_drugs}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="ova-small-muted">No standard variant-specific therapy defined.</div>',
                unsafe_allow_html=True,
            )

        citations = get_field(result, "oncokb_MUTATION_EFFECT_CITATIONS")
        st.markdown(
            '<div class="ova-kv-label" style="margin-top:1rem;">Citations (PMIDs)</div>',
            unsafe_allow_html=True,
        )
        st.caption(citations)


def render_agent_report(result: dict[str, Any]) -> None:
    st.markdown(
        '<div class="ova-section-title"><span class="ova-step ova-step-primary">2</span>'
        "AI-Augmented Therapeutic Intelligence (OncoVarAgent)</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("#### Proposed Therapeutic Strategies")
        drugs = parse_list(result.get("OncoVarAgent_Drugs"), separator=";")
        if drugs:
            for index, drug in enumerate(drugs, start=1):
                st.markdown(f'<div class="ova-drug">{index}. {drug}</div>', unsafe_allow_html=True)
        else:
            st.info("No specific novel strategies identified by the agent.")

    with st.container(border=True):
        st.markdown("#### Brief Report")
        st.markdown(get_field(result, "OncoVarAgent_Brief_Report"))

    with st.container(border=True):
        st.markdown("#### Deep Report")
        st.markdown(get_field(result, "OncoVarAgent_Deep_Report"))

    st.markdown('<div class="ova-evidence-grid"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Relevant Clinical Trials")
            trial_ids = parse_list(result.get("OncoVarAgent_Clinical_Trial_IDs"))
            if trial_ids:
                links = " ".join(
                    f'<a class="ova-chip" target="_blank" href="https://clinicaltrials.gov/study/{trial_id}">{trial_id}</a>'
                    for trial_id in trial_ids
                )
                st.markdown(links, unsafe_allow_html=True)
            else:
                st.caption("No clinical trial IDs returned.")

    with col2:
        with st.container(border=True):
            st.markdown("#### Supporting Literature")
            pmids = parse_list(result.get("OncoVarAgent_Support_Literatures"))
            if pmids:
                links = " ".join(
                    f'<a class="ova-chip" target="_blank" href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/">{pmid}</a>'
                    for pmid in pmids
                )
                st.markdown(links, unsafe_allow_html=True)
            else:
                st.caption("No PubMed IDs returned.")


def render_result(result: dict[str, Any], logs: list[str] | None = None) -> None:
    gene = get_field(result, "gene")
    protein_change = get_field(result, "protein_change")
    cancer_type = get_field(result, "cancer_type")

    st.markdown("### Analysis Result")
    header_cols = st.columns([1, 1.4, 2, 1.1])
    with header_cols[0]:
        st.markdown('<div class="ova-kv-label">Gene</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ova-kv-value" style="font-size:1.6rem;">{gene}</div>', unsafe_allow_html=True)
    with header_cols[1]:
        st.markdown('<div class="ova-kv-label">Variant</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ova-kv-value">{protein_change}</div>', unsafe_allow_html=True)
    with header_cols[2]:
        st.markdown('<div class="ova-kv-label">Indication</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ova-kv-value">{cancer_type}</div>', unsafe_allow_html=True)
    with header_cols[3]:
        render_print_button()

    render_baseline_guidance(result)
    render_agent_report(result)

    if logs:
        with st.expander("Run logs", expanded=False):
            st.code("\n".join(logs), language="text")


def render_runtime_info() -> dict[str, str]:
    with st.sidebar:
        st.markdown("### Runtime Settings")
        st.caption("These values are applied before importing and running the local agent workflow.")

        oncokb_api_token = st.text_input(
            "ONCOKB_API_TOKEN",
            value=os.getenv("ONCOKB_API_TOKEN", ""),
            type="password",
            placeholder="xxxx",
        )
        llm_api_token = st.text_input(
            "LLM_API_TOKEN",
            value=os.getenv("LLM_API_TOKEN", os.getenv("MULERUN_API", "")),
            type="password",
            placeholder="xxxx",
        )
        llm_api_url = st.text_input(
            "LLM_API_URL",
            value=os.getenv("LLM_API_URL", DEFAULT_LLM_API_URL),
        )
        llm_model = st.text_input(
            "LLM_MODEL",
            value=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        )
        return {
            "ONCOKB_API_TOKEN": oncokb_api_token.strip(),
            "LLM_API_TOKEN": llm_api_token.strip(),
            "LLM_API_URL": llm_api_url.strip() or DEFAULT_LLM_API_URL,
            "LLM_MODEL": llm_model.strip() or DEFAULT_LLM_MODEL,
            "ONCOKB_ANNOTATOR_PATH": str(DEFAULT_ONCOKB_ANNOTATOR_PATH),
        }


def main() -> None:
    render_header()
    settings = render_runtime_info()

    if "example_loaded" not in st.session_state:
        st.session_state.example_loaded = False
    if "result" not in st.session_state:
        st.session_state.result = None
    if "logs" not in st.session_state:
        st.session_state.logs = []

    with st.form("variant_form"):
        col1, col2, col3 = st.columns(3)
        gene = col1.text_input(
            "Gene",
            value="RBM10" if st.session_state.example_loaded else "",
            placeholder="RBM10",
        ).upper()
        variant = col2.text_input(
            "Variant",
            value="N446Kfs*35" if st.session_state.example_loaded else "",
            placeholder="N446Kfs*35",
        )
        cancer_type = col3.text_input(
            "Cancer Type",
            value="Non-Small Cell Lung Cancer" if st.session_state.example_loaded else "",
            placeholder="Non-Small Cell Lung Cancer",
        )

        action_col, example_col = st.columns([3, 1])
        submitted = action_col.form_submit_button("AI Research", type="primary", use_container_width=True)
        load_example = example_col.form_submit_button("Load Example", use_container_width=True)

    if load_example:
        st.session_state.example_loaded = True
        st.rerun()

    if submitted:
        if not gene.strip():
            st.warning("Gene is required.")
        elif not variant.strip():
            st.warning("Variant is required.")
        elif not cancer_type.strip():
            st.warning("Cancer type is required.")
        else:
            missing_settings = [
                key
                for key in ("ONCOKB_API_TOKEN", "LLM_API_TOKEN")
                if not settings[key]
            ]
            if missing_settings:
                st.warning(f"Missing runtime settings: {', '.join(missing_settings)}")
                return
            if not Path(settings["ONCOKB_ANNOTATOR_PATH"]).exists():
                st.warning("Bundled OncoKB annotator is missing.")
                return
            st.session_state.result = None
            st.session_state.logs = []
            try:
                result, logs = run_analysis(
                    gene=gene.strip(),
                    variant=variant.strip(),
                    cancer_type=cancer_type.strip(),
                    settings=settings,
                )
                st.session_state.result = result
                st.session_state.logs = logs
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.exception(exc)

    if st.session_state.result:
        render_result(st.session_state.result, st.session_state.logs)

    st.markdown('<div class="ova-footer">© 2026 OncoVarAgent Project</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
