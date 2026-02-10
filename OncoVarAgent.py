# --- OncoVarAgent.py ---

#qwen https://dashscope.aliyuncs.com/compatible-mode/v1

import os
import json
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Annotated, Sequence
import operator
import argparse
import pandas as pd
import subprocess
import tempfile
import re
import requests
import xml.etree.ElementTree as ET
import time

# --- LangChain & LangGraph Imports ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages # Helper for state
from langgraph.prebuilt import ToolNode # Prebuilt tool calling node


# --- Environment & Configuration ---
load_dotenv()
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("MODEL_NAME")

try:
    LLM = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0
    )

    print("Successfully connected to the API.")
except Exception as e:
    print(f"Could not initialize ChatOpenAI. Error: {e}")
    LLM = None

# --- Helper Functions & Constants ---
AA_MAP = {'A':'Ala','R':'Arg','N':'Asn','D':'Asp','C':'Cys','Q':'Gln','E':'Glu','G':'Gly','H':'His','I':'Ile','L':'Leu','K':'Lys','M':'Met','F':'Phe','P':'Pro','S':'Ser','T':'Thr','W':'Trp','Y':'Tyr','V':'Val','X':'Ter','*':'Ter'}
ONCOKB_TO_AMP_MAPPING = {"1":{"tier":"Tier I","level":"A"},"2":{"tier":"Tier I","level":"A"},"3A":{"tier":"Tier I","level":"B"},"3B":{"tier":"Tier II","level":"C"},"4":{"tier":"Tier II","level":"D"},"R1":{"tier":"Tier I","level":"A"},"R2":{"tier":"Tier II","level":"D"}}
LEVEL_PRECEDENCE = ['1','2','R1','3A','3B','R2','4']

def add_amp_tier_to_df(df: pd.DataFrame) -> pd.DataFrame:
    def process_row(row):
        drug_info_list, tier = [], None
        for key in LEVEL_PRECEDENCE:
            col = f'LEVEL_{key}'
            if col in row.index and pd.notna(row[col]):
                drugs, mapping = str(row[col]), ONCOKB_TO_AMP_MAPPING.get(key)
                if not mapping: continue
                if tier is None: tier = mapping['tier']
                status = "resistance" if 'R' in key else "sensitive"
                for drug in [d.strip() for d in drugs.split(',') if d.strip()]:
                    drug_info_list.append(f"{drug}({status}, Level {mapping['level']} Evidence)")
        if tier is None:
            oncogenic_status = row.get('ONCOGENIC')
            if oncogenic_status in ['Oncogenic', 'Likely Oncogenic']:
                tier = "Tier II"
            elif oncogenic_status in ['Likely Neutral', 'Neutral']:
                tier = "Tier IV"
            else:
                tier = "Tier III"
        drugs_str = "; ".join(drug_info_list) or "N/A"
        return pd.Series([tier, drugs_str])
    df[['AMP_TIER', 'Drugs']] = df.apply(process_row, axis=1)
    return df

# --- Tool Definitions ---
@tool
def run_oncokb_annotator(maf_filepath: str, tumor_type: str) -> str:
    """Runs the OncoKB annotator script to get foundational variant interpretations."""
    print(f"---TOOL: Running OncoKB Annotator on {maf_filepath} for {tumor_type}---")
    annotator_path, api_token = os.getenv("ONCOKB_ANNOTATOR_PATH"), os.getenv("ONCOKB_API_TOKEN")
    if not all([annotator_path, api_token]): return "ERROR: OncoKB annotator path or API token not configured."
    with tempfile.NamedTemporaryFile(mode='r+', delete=True, suffix=".txt") as outfile:
        command = ["python", annotator_path, "-i", maf_filepath, "-o", outfile.name, "-b", api_token, "-t", tumor_type, "-d"]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            if os.path.getsize(outfile.name) > 0:
                df = add_amp_tier_to_df(pd.read_csv(outfile.name, sep='\t'))
                df.to_csv('11.txt', sep='\t', index=False)
                cols = ['Hugo_Symbol', 'HGVSp_Short', 'ONCOGENIC', 'AMP_TIER', 'Drugs', 'MUTATION_EFFECT','MUTATION_EFFECT_CITATIONS','MUTATION_EFFECT_DESCRIPTION']
                return df[[c for c in cols if c in df.columns]].to_json(orient='records')
            return "WARNING: OncoKB annotator produced an empty output file."
        except Exception as e: return f"CRITICAL ERROR: OncoKB annotator failed. {e}"

@tool
def pubmed_search(query: str, max_results: int = 5) -> dict:
    """Searches PubMed for a specific query and returns structured article data."""
    print(f"---TOOL: Searching PubMed for query: '{query}'---")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    try:
        params={"db":"pubmed","term":query,"retmax":str(max_results),"retmode":"json"}; r=requests.get(f"{base_url}esearch.fcgi",params=params); time.sleep(1); r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids: return {"status": "no results found", "articles": []}
        params={"db":"pubmed","id":",".join(ids),"retmode":"xml","rettype":"abstract"}; r=requests.get(f"{base_url}efetch.fcgi",params=params); time.sleep(1); r.raise_for_status()
        root, articles = ET.fromstring(r.content), []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID", "N/A")
            title = article.findtext(".//ArticleTitle", "No Title Available")
            abstract = "\n".join(["".join(e.itertext()) for e in article.findall(".//Abstract/AbstractText")]) or "No Abstract Available"
            articles.append({"title": title, "abstract": abstract, "support_literatures": pmid})
        return {"status": "success", "articles": articles}
    except Exception as e: return {"status": "error", "message": f"PubMed search failed: {e}"}

@tool
def query_clinical_trials(
    intervention: str = None,
    condition: str = None,
    other_terms: str = None,
    max_results: int = 5,
    status: str = "Active", # Use a user-friendly default
    study_type: str = "Interventional"
) -> dict:
    """
    Queries ClinicalTrials.gov for relevant studies using structured parameters.
    Provide at least one of 'intervention', 'condition', or 'other_terms'.
    """
    print(f"---TOOL: Querying ClinicalTrials.gov for intervention='{intervention}', condition='{condition}', status='{status}'---")
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    # --- [FIX 1] --- Map user-friendly status to the correct API enums
    # 'Active' is a common concept for studies that are ongoing.
    status_mapping = {
        "active": "RECRUITING|NOT_YET_RECRUITING|ACTIVE_NOT_RECRUITING|ENROLLING_BY_INVITATION",
        "recruiting": "RECRUITING",
        "completed": "COMPLETED"
        # Add other mappings as needed
    }
    # Default to all if status is not in our map
    api_status = status_mapping.get(status.lower(), "RECRUITING|NOT_YET_RECRUITING|ACTIVE_NOT_RECRUITING|ENROLLING_BY_INVITATION|COMPLETED|TERMINATED")

    params = {
        "pageSize": max_results,
        "format": "json",
        "filter.overallStatus": api_status
    }

    # --- [FIX 2] --- Use the correct structured query parameters
    if intervention:
        params["query.intr"] = intervention
    if condition:
        params["query.cond"] = condition
    if other_terms:
        params["query.term"] = other_terms

    # --- [FIX 3] --- Use the correct parameter for study_type
    if study_type:
        params["filter.advanced"] = f"AREA[StudyType]{study_type}"

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Will raise an exception for 4xx/5xx errors
        time.sleep(0.5)
        
        data = response.json()
        studies = data.get("studies", [])
        if not studies: 
            return {"status": "no results found", "trials": []}
            
        trial_list = []
        for t in studies:
            proto = t.get("protocolSection", {})
            
            def get_nested(data, path):
                for key in path:
                    if isinstance(data, dict): data = data.get(key)
                    else: return None
                return data

        #    locations = get_nested(proto, ["contactsLocationsModule", "locations"]) or []
        #    location_str = ", ".join(
        #        [f"{loc.get('city', '')}, {loc.get('country', '')}" for loc in locations if loc.get('country')]
        #    )

            trial_info = {
                "nct_id": get_nested(proto, ["identificationModule", "nctId"]),
                "title": get_nested(proto, ["identificationModule", "briefTitle"]),
                "brief_summary": get_nested(proto, ["descriptionModule", "briefSummary"]),
                "study_type": get_nested(proto, ["designModule", "studyType"]),
                "status": get_nested(proto, ["statusModule", "overallStatus"]),
                "phase": ", ".join(get_nested(proto, ["designModule", "phases"]) or []),
                "conditions": ", ".join(get_nested(proto, ["conditionsModule", "conditions"]) or []),
                "interventions": [
                    {"type": i.get("type"), "name": i.get("name")} 
                    for i in get_nested(proto, ["armsAndInterventionsModule", "interventions"]) or []
                ],

                "eligibility_criteria_text": get_nested(proto, ["eligibilityModule", "eligibilityCriteria"]),
            }
            trial_list.append(trial_info)

        return {"status": "success", "total_found": data.get("totalCount", 0), "trials": trial_list}
    except requests.exceptions.HTTPError as e:
        # Provide a more informative error message
        return {"status": "error", "message": f"ClinicalTrials search failed with HTTP {e.response.status_code}. URL: {e.request.url}. Response: {e.response.text}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred in ClinicalTrials search: {e}"}
        
# --- Agent & Graph State Definition ---
class AgentState(TypedDict):
    patient_info: Dict[str, Any]
    variants_to_process: List[Dict[str, Any]]
    processed_variants_reports: Annotated[List[Dict[str, Any]], operator.add]
    current_variant_info: Dict[str, Any]
    pubmed_results: Dict[str, Any]
    clinical_trial_results: Dict[str, Any]
    summarized_evidence: str
    final_report: Dict[str, Any]



FINAL_SYNTHESIZER_PROMPT = ChatPromptTemplate.from_template(
"""You are an expert-level clinical genomics analyst AI. Your sole function is to process a pre-compiled evidence summary and format it into a structured JSON object. You do not make judgments; you only extract, summarize, and structure information.

**Baseline Variant Information (from OncoKB):**
{oncokb_info}

**Full Evidence Review from Research Agent (this is your primary source of truth):**
--- START OF REVIEW ---
{summarized_evidence}
--- END OF REVIEW ---

**Your Task (Follow these steps precisely):**

1.  **Directly Ingest the Full Report:** The text provided in the "Full Evidence Review" is the complete deep report. You will place this text, verbatim and unaltered, into the `OncoVarAgent_Brief_Deep_Report` field.

2.  **Create a Brief Summary:** Read the "Full Evidence Review" and create a concise, 2-3 sentence executive summary. This summary should capture the most critical therapeutic findings or implications. Place this summary in the `OncoVarAgent_Brief_Report` field.

3.  **Extract Key Entities:**
    *   From the "Full Evidence Review," identify all mentioned therapeutic agents (drugs). Format them as a single string: `"Drug1(Status, Evidence Type); Drug2(Status, Evidence Type); ..."`. Use information from the text to determine the status (e.g., sensitive, resistance) and evidence type (e.g., Phase II Trial, Preclinical Study).
    *   Identify all PubMed IDs (PMIDs) and format them as a comma-separated string: `"PMID1,PMID2,..."`.
    *   Identify all Clinical Trial IDs (NCT IDs) and format them as a comma-separated string: `"NCT1,NCT2,..."`.

**JSON Output Schema (You MUST adhere to this structure. Do not add any extra text or explanations outside the JSON object):**
```json
{{
    "OncoVarAgent_Drugs": "Drug(Status, Evidence Type);...",
    "OncoVarAgent_Support_Literatures": "PMID1,PMID2,...",
    "OncoVarAgent_Clinical_Trial_IDs": "NCT1,NCT2,...",
    "OncoVarAgent_Brief_Report": "A 2-3 sentence summary of the key findings from the deep report.",
    "OncoVarAgent_Deep_Report": "The full, original text from the 'Full Evidence Review' section."
}}
"""
)

# --- Node Definitions ---

def annotator_node(state: AgentState) -> dict:
    """Initial node to annotate the input file with OncoKB."""
    print("---NODE: Annotator---")
    info = state["patient_info"]
    try:
        input_filepath = info["input_txt"]
        df = pd.read_csv(input_filepath, sep='\t', comment='#')
        tumor_type = df[info["cancer_type_col"]].iloc[0]
        state["patient_info"]["cancer_type"] = tumor_type # Update state
        result_str = run_oncokb_annotator.invoke({"maf_filepath": input_filepath, "tumor_type": tumor_type})
        variants = json.loads(result_str)
        return {"variants_to_process": variants if isinstance(variants, list) else []}
    except Exception as e:
        print(f"Error in annotator node: {e}")
        return {"variants_to_process": []}

def get_next_variant(state: AgentState) -> dict:
    """Pops the next variant from the list to be processed."""
    print("\n---NODE: Get Next Variant---")
    variants = state.get('variants_to_process', [])
    if not variants:
        print("No more variants to process.")
        return {"current_variant_info": None}
    next_variant = variants.pop(0)
    print(f"Processing: {next_variant.get('Hugo_Symbol')} {next_variant.get('HGVSp_Short')}")
    return {"current_variant_info": next_variant, "variants_to_process": variants}

# --- [NEW] ReAct Agent for PubMed Search (Custom Implementation) ---

# 1. Define the State for the ReAct agent's internal loop
class ReActState(TypedDict):
    """The state of the ReAct agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # --- [NEW] --- Add fields to store curated results
    curated_pubmed_articles: Annotated[List[Dict[str, Any]], operator.add]
    curated_clinical_trials: Annotated[List[Dict[str, Any]], operator.add]


# 2. Define the tools and bind them to the LLM
#    This agent only needs the pubmed_search tool.
deep_research_tools = [pubmed_search,query_clinical_trials]
llm_with_tools = LLM.bind_tools(deep_research_tools)

# 3. Define the Nodes for the ReAct agent's graph
# --- 3. Define the Nodes for the ReAct agent's graph
def call_model(state: ReActState):
    """Node that invokes the LLM with the current conversation state."""
    print("  - ReAct Agent: Thinking...")
    response = llm_with_tools.invoke(state["messages"])

    # --- [NEW] Start of added logging ---
    # The 'content' is the LLM's thought process or final answer.
    if response.content:
        print(f"\n  >>> LLM Thought:\n  {response.content}\n")

    # The 'tool_calls' attribute contains the specific tool to be run.
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"  >>> LLM Action: {tool_call['name']}({tool_call['args']})")
    # --- [NEW] End of added logging ---

    # Return a list, which will be appended to the messages state
    return {"messages": [response]}

# We can use the prebuilt ToolNode, which is a convenient way to call tools
tool_node = ToolNode(deep_research_tools)

# 4. Define the Conditional Edge for the ReAct agent's graph
def should_continue(state: ReActState):
    """Conditional edge to decide whether to continue the loop or finish."""
    last_message = state["messages"][-1]
    # If the last message is not a tool call, then we're done
    if not last_message.tool_calls:
        return "end"
    # Otherwise, we continue by calling the tool
    return "continue"

# 5. Assemble and Compile the ReAct Agent Graph
react_workflow = StateGraph(ReActState)

react_workflow.add_node("agent", call_model)
react_workflow.add_node("action", tool_node)

react_workflow.set_entry_point("agent")

react_workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END},
)
react_workflow.add_edge("action", "agent")

# Compile the graph into a runnable agent
deep_researcher_agent = react_workflow.compile()


def deep_research_node(state: AgentState) -> dict:
    """
    Invokes the ReAct agent for deep research and extracts structured PubMed and ClinicalTrials results.
    """
    print("\n---NODE: Deep ReAct Researcher---")
    variant = state['current_variant_info']
    patient = state['patient_info']
    
    task = (
        f"You are an expert oncology researcher. Your mission is to uncover all relevant therapeutic evidence for the variant **{variant['Hugo_Symbol']} {variant['HGVSp_Short']}** in **{patient['cancer_type']}**."
        f"\nYou must follow a strict, function-driven workflow."

        f"\n\n**--- Baseline Information from OncoKB ---**"
        f"\n- **Known Mutation Effect:** {variant['MUTATION_EFFECT']}"
        f"\n- **Mutation Effect Description:** {variant['MUTATION_EFFECT_DESCRIPTION']}"
        f"\nUse this information as your starting point."

        f"\n\n**--- Workflow ---**"

        f"\n\n**Phase 1: Functional Characterization & Variant-Level Evidence**"
        f"\n1.  **Goal:** Confirm the variant's function (GoF/LoF) and find any direct therapeutic evidence for this specific variant."
        f"\n2.  **Action A (Specific Cancer):** Perform a `pubmed_search` for `'{variant['Hugo_Symbol']} AND {variant['HGVSp_Short']} AND ({patient['cancer_type']})'`."
        f"\n3.  **Action B (Pan-Cancer):** Perform a `pubmed_search` for `'{variant['Hugo_Symbol']} AND {variant['HGVSp_Short']} AND (tumor OR cancer)'`."
        f"\n4.  **Analysis:** After reviewing results from BOTH searches, you MUST declare a definitive conclusion in your thoughts (e.g., 'Based on the literature and OncoKB and mutation type, I confirm this is a LoF variant'). If the function remains unknown, your mission is complete, immediately stop."
        f"\n5.  **Follow-up Validation:** If any specific drugs are mentioned, immediately test them using `query_clinical_trials`."

        f"\n\n**Phase 2: Gene-Focused Search (Specific Cancer)**"
        f"\n1.  **Goal:** Find therapies targeting `{variant['Hugo_Symbol']}` within `({patient['cancer_type']})`."
        f"\n2.  **Action:** Perform a `pubmed_search` using `'{variant['Hugo_Symbol']} AND ({patient['cancer_type']}) AND (therapy OR treatment OR inhibitor)'`."
        f"\n3.  **Critical Analysis:** You MUST look for mentioned drugs and downstream pathways (e.g., 'AKT', 'PI3K', 'MEK'). These are new hypotheses."
        f"\n4.  **Follow-up Validation:** Immediately test ALL new hypotheses (drugs, drug classes, pathway inhibitors) using `query_clinical_trials`."

        f"\n\n**Phase 3: Gene-Focused Search (Pan-Cancer)**"
        f"\n1.  **Goal:** Find therapies for `{variant['Hugo_Symbol']}` with pan-cancer approval or strong evidence in other cancers."
        f"\n2.  **Action:** Perform a `pubmed_search` using `'{variant['Hugo_Symbol']} AND (cancer OR tumor) AND (therapy OR inhibitor)'`."
        f"\n3.  **Critical Analysis:** Identify drugs with pan-cancer relevance."
        f"\n4.  **Follow-up Validation:** Immediately test these pan-cancer hypotheses using `query_clinical_trials`."

        f"\n\n**Phase 4: Mechanistic Deep Dive**"
        f"\n1.  **Goal:** Uncover therapies based on the gene's biological function, both within `{patient['cancer_type']}` and across other cancers."
        f"\n2.  **Action A (Specific Cancer):** Perform a creative `pubmed_search` based on the gene's role within the patient's cancer, informed by your GoF/LoF conclusion."
        f"\n3.  **Action B (Pan-Cancer):** Broaden the mechanistic search to find evidence of the same therapeutic strategy in other cancers, linking it to the gene."
        f"\n4.  **Follow-up Validation:** Test any final hypotheses from BOTH searches with `query_clinical_trials`."

        f"\n\n**--- Tool Usage Rules ---**"
        f"\n1.  **Think Step-by-Step:** Before every tool call, you MUST output your thought process. This thought process MUST start with a brief summary of the previous action's result (e.g., 'The last search found 2 relevant articles(must list pmid or nctid)..., '), then state your reasoning for the next action."
        f"\n2.  **Single Tool Per Action:** You MUST call only one tool in a single thinking step. Do not issue multiple tool calls at once. Plan your steps sequentially."
        f"\n3.  **`pubmed_search`:** For 'OR' conditions, you MUST use parentheses: `(therapy OR treatment)`. Always set `max_results` to 20."
        f"\n4.  **`query_clinical_trials`:** You MUST use structured parameters (`intervention`, `condition`). If a specific search fails or returns no results, DO NOT give up. Your immediate next step is to broaden the search by calling the tool with only one parameter (e.g., just `intervention`).Always set `max_results` to 20."

        f"\n\n**--- Final Report Structure ---**"
        f"\nYour final thought process MUST be a mini-review with the following sections:"
        f"\n\n**1. Executive Summary:** 1-2 key sentences on the therapeutic findings."
        f"\n\n**2. Evidence Synthesis:** This is the main body of your review. Group your curated findings by therapeutic strategy or drug class, not by search order. Your synthesis MUST be driven by biological and mechanistic reasoning. For each finding, you MUST:"
        f"\n    - State the therapeutic hypothesis (e.g., 'Targeting with MEK inhibitors')."
        f"\n    - Describe the supporting evidence concisely, explaining the biological rationale (e.g., '...because this GoF variant leads to pathway hyperactivation...')."
        f"\n    - **Cite your sources in-line**, like this: (PMID: 12345678, NCT: NCT01234567).Crucially, every PMID and NCT cited in this section MUST also be present in the final 'Curated Evidence Lists' below. Do not cite any source that is not included in those final lists."
        f"\n    - Distinguish between the strength of evidence (e.g., preclinical, case report, Phase III trial, Retrospective et al)."
        f"\n\n**3. Conclusion:** Briefly summarize clinical actionability."
        
        f"\n\n**4. Curated Evidence Lists (CRITICAL INSTRUCTION):** End your report with these exact lines. 'Relevant' means a PMID or NCT **directly supports a therapeutic action** discussed in your synthesis. If no such evidence was found, these lists MUST be empty `[]`.This final list MUST contain every PMID and NCT that you cited in the 'Evidence Synthesis' section. The set of IDs in your report text and the set of IDs in this list must be absolutely identical."
        f"\n`Relevant PMIDs: [\"PMID1\", \"PMID2\"]`"
        f"\n`Relevant NCTs: [\"NCT_ID1\", \"NCT_ID2\"]`"
    )
    
    initial_react_state = {
        "messages": [("user", task)]        
        }
    final_react_state = deep_researcher_agent.invoke(initial_react_state)

    final_conclusion = final_react_state['messages'][-1].content
    print(f"\n  >>> LLM Final Conclusion:\n  {final_conclusion}\n")
    
    # --- NEW: Extract structured data directly from tool calls ---
    # Step 1: Gather ALL results found during the process, just like before.
    all_articles_found = []
    all_trials_found = []
    for message in final_react_state['messages']:
        if isinstance(message, ToolMessage):
            try:
                tool_output = json.loads(message.content)
                if message.name == 'pubmed_search' and tool_output.get('articles'):
                    all_articles_found.extend(tool_output['articles'])
                elif message.name == 'query_clinical_trials' and tool_output.get('trials'):
                    all_trials_found.extend(tool_output['trials'])
            except (json.JSONDecodeError, TypeError):
                continue

    # Step 2: Use the LLM's final conclusion to filter these comprehensive lists.
    # The regex is designed to be flexible, handling formats like ["123"], "123", [123], etc.
    final_pmids_str = re.findall(r'[\'"](\d{8,})[\'"]', final_conclusion)
    final_ncts_str = re.findall(r'[\'"](NCT\d+)[\'"]', final_conclusion)
    
    # Step 3: Create the final curated lists.
    curated_articles = [art for art in all_articles_found if art.get('support_literatures') in final_pmids_str]
    curated_trials = [trial for trial in all_trials_found if trial.get('nct_id') in final_ncts_str]

    # De-duplicate results to be safe
    if curated_articles:
        df_articles = pd.DataFrame(curated_articles).drop_duplicates(subset=['support_literatures'], keep='first')
        curated_articles = df_articles.to_dict('records')

    if curated_trials:
        df_trials = pd.DataFrame(curated_trials).drop_duplicates(subset=['nct_id'], keep='first')
        curated_trials = df_trials.to_dict('records')
    
    print(f"  - ReAct Agent curated {len(curated_articles)} relevant articles from its search.")
    print(f"  - ReAct Agent curated {len(curated_trials)} relevant trials from its search.")

    return {
        "pubmed_results": {"status": "success", "articles": curated_articles},
        "clinical_trial_results": {"status": "success", "trials": curated_trials},
        "summarized_evidence": final_conclusion
    }
    



def format_oncokb_only_node(state: AgentState) -> dict:
    """Formats a variant report using only OncoKB data when no deep search is needed."""
    print("---NODE: Format OncoKB Only (Skipped Deep Search)---")
    variant = state['current_variant_info']

    report = {
        "gene": variant.get('Hugo_Symbol'),
        "protein_change": variant.get('HGVSp_Short'),
        "cancer_type": state['patient_info'].get('cancer_type'),
        "oncokb_ONCOGENIC": variant.get('ONCOGENIC', 'N/A'),
        "oncokb_AMP_TIER": variant.get('AMP_TIER', 'N/A'),
        "oncokb_Drugs": variant.get('Drugs', 'N/A'),
        "oncokb_MUTATION_EFFECT": str(variant.get('MUTATION_EFFECT', 'N/A')),
        "oncokb_MUTATION_EFFECT_CITATIONS": str(variant.get('MUTATION_EFFECT_CITATIONS', 'N/A')),
        # Fill agent columns with placeholder values
        "OncoVarAgent_Drugs": "N/A",
        "OncoVarAgent_Support_Literatures": "N/A",
        "OncoVarAgent_Clinical_Trial_IDs": "N/A",
        "OncoVarAgent_Brief_Report": "N/A",
        "OncoVarAgent_Deep_Report": "N/A",
    }
    return {"processed_variants_reports": [report]}

def single_variant_synthesizer_node(state: AgentState) -> dict:
    """Synthesizes OncoKB data with new evidence from deep research into a structured report."""
    print("---NODE: Synthesize with Agent Findings---")
    variant = state['current_variant_info']
    
    # Base report from OncoKB data
    base_report = {
        "gene": variant.get('Hugo_Symbol'),
        "protein_change": variant.get('HGVSp_Short'),
        "cancer_type": state['patient_info'].get('cancer_type'),
        "oncokb_ONCOGENIC": variant.get('ONCOGENIC', 'N/A'),
        "oncokb_AMP_TIER": variant.get('AMP_TIER', 'N/A'),
        "oncokb_Drugs": variant.get('Drugs', 'N/A'),
        "oncokb_MUTATION_EFFECT": str(variant.get('MUTATION_EFFECT', 'N/A')),
        "oncokb_MUTATION_EFFECT_CITATIONS": str(variant.get('MUTATION_EFFECT_CITATIONS', 'N/A')),
    }
    
    chain = FINAL_SYNTHESIZER_PROMPT | LLM
    raw_output = chain.invoke({
        "gene": base_report["gene"], "variant": base_report["protein_change"],
        "cancer_type": base_report["cancer_type"],
        "oncokb_info": json.dumps(variant),
        "summarized_evidence": state.get("summarized_evidence", "No new evidence was summarized by the research agent.")
    }).content
    
    try:
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if json_match:
            agent_findings = json.loads(json_match.group(0))
            # Merge base OncoKB info with the LLM's new analysis
            final_report = {**base_report, **agent_findings}
            return {"processed_variants_reports": [final_report]}
        else:
            raise ValueError("No JSON object found in LLM output.")
    except Exception as e:
        print(f"ERROR during final synthesis: {e}. Raw output was: {raw_output}")
        error_report = {
            **base_report,
            "OncoVarAgent_Proposed_AMP_Tier": "Error", "OncoVarAgent_Drugs": "Error",
            "OncoVarAgent_Support_Literatures": "Error", "OncoVarAgent_Clinical_Trial_IDs": "Error",
            "AMP_Tier_Adjustment": "Error", "Justification": f"Failed to synthesize report. Error: {e}"
        }
        return {"processed_variants_reports": [error_report]}

def final_combiner_node(state: AgentState) -> dict:
    """Combines all individual variant reports into one final file."""
    print("\n---NODE: Final Combiner---")
    return {"final_report": {"variant_report": state.get('processed_variants_reports', [])}}

# --- Graph Assembly (No changes here) ---
workflow = StateGraph(AgentState)

# In --- Graph Assembly --- section

def route_after_variant_get(state: AgentState) -> str:
    """
    Routes the workflow based on the current variant's properties.
    This is the main filter to decide if a deep search is necessary.
    """
    current_variant = state.get("current_variant_info")
    if not current_variant:
        return "end_loop"

    # Priority 1: OncoKB already provides therapeutic options.
    drugs = current_variant.get('Drugs', 'N/A')
    if drugs and drugs != 'N/A':
        print(f"  - ROUTING: Skipping deep search for {current_variant.get('HGVSp_Short')}. Reason: OncoKB provided drugs.")
        return "skip_deep_search"

    # Priority 2: The variant is classified as likely benign.
    oncogenicity = current_variant.get('ONCOGENIC', 'N/A')
    if oncogenicity in ['Likely Neutral', 'Neutral']:
        print(f"  - ROUTING: Skipping deep search for {current_variant.get('HGVSp_Short')}. Reason: Variant is likely benign.")
        return "skip_deep_search"

    # Default: The variant is actionable (Oncogenic, Likely Oncogenic, or Unknown) and has no direct drug info.
    print(f"  - ROUTING: Proceeding to deep search for {current_variant.get('HGVSp_Short')}. Reason: Actionable oncogenicity ('{oncogenicity}') with no drugs listed.")
    return "perform_deep_search"

workflow.add_node("annotator", annotator_node)
workflow.add_node("get_next_variant", get_next_variant)
workflow.add_node("deep_researcher", deep_research_node)
workflow.add_node("format_oncokb_only", format_oncokb_only_node)

workflow.add_node("single_variant_synthesizer", single_variant_synthesizer_node)
workflow.add_node("final_combiner", final_combiner_node)

workflow.set_entry_point("annotator")
workflow.add_edge("annotator", "get_next_variant")
workflow.add_conditional_edges(
    "get_next_variant", route_after_variant_get,
    {"end_loop": "final_combiner", "skip_deep_search": "format_oncokb_only", "perform_deep_search": "deep_researcher"}
)

workflow.add_edge("format_oncokb_only", "get_next_variant")

workflow.add_edge("deep_researcher", "single_variant_synthesizer")
workflow.add_edge("single_variant_synthesizer", "get_next_variant")
workflow.add_edge("final_combiner", END)

app = workflow.compile()



# --- Main Execution Block (Final, Most Compatible Version) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OncoVarAgent: An agent for interpreting cancer genomic variants.")
    parser.add_argument("--input-txt", required=True, type=str, help="Path to a tab-separated file for analysis.")
    parser.add_argument("--gene-col", type=str, default="Hugo_Symbol", help="Column name for gene symbol.")
    parser.add_argument("--protein-change-col", type=str, default="HGVSp_Short", help="Column name for HGVSp.")
    parser.add_argument("--cancer-type-col", type=str, default="Cancer_Type", help="Column name for cancer type.")
    parser.add_argument("--output", type=str, default="variant_interpretation_report.xlsx", help="Output Excel file name.")
    args = parser.parse_args()

    if not LLM:
        print("Exiting: One or more LLMs not initialized.")
    else:
        initial_state = {
            "patient_info": vars(args),
            "processed_variants_reports": [],
            "final_report": {}
        }
        final_state_result = None
        print("\n" + "="*30 + " Starting OncoVarAgent Workflow " + "="*30)

        # Iterate through the stream to print logs.
        # We will capture the final state from the '__end__' event, which is guaranteed to be last.
        for event in app.stream(initial_state, {"recursion_limit": 20000}): # Increased recursion limit as in your example
            event_key = list(event.keys())[0]
            event_value = event[event_key]
            
            # Print all events for full visibility
            if event_key != "__end__":
                print(f"\n--- Event from Node: '{event_key}' ---")
                if event_value:
                    print(json.dumps(event_value, indent=2, ensure_ascii=False))
                else:
                    print("Node ran, but returned no new patch to the state.")
                print("-" * 80)
            
            # Capture the specific output from the final_combiner node
            if event_key == "final_combiner":
                final_state_result = event_value

        print("\n" + "="*30 + " OncoVarAgent Workflow Finished " + "="*30)
        
        # Now, final_state_result should hold the complete state from the __end__ event.
        if final_state_result and final_state_result.get('final_report', {}).get('variant_report'):
            try:
                # Ensure the report is not empty before creating a DataFrame
                report_data = final_state_result['final_report']['variant_report']
                if report_data:
                    df = pd.DataFrame(report_data)
                
                    # Define the complete and final column order
                    desired_columns = [
                        'gene', 'protein_change', 'cancer_type',
                        # OncoKB Columns
                        'oncokb_ONCOGENIC', 'oncokb_AMP_TIER', 'oncokb_Drugs', 'oncokb_MUTATION_EFFECT','oncokb_MUTATION_EFFECT_CITATIONS',
                        # OncoVarAgent Columns
                        'OncoVarAgent_Drugs', 
                        'OncoVarAgent_Support_Literatures', 'OncoVarAgent_Clinical_Trial_IDs',
                        'OncoVarAgent_Brief_Report','OncoVarAgent_Deep_Report'
                    ]
                
                    # Reorder columns, keeping only those that exist in the DataFrame
                    df = df[[col for col in desired_columns if col in df.columns]]
                
                    df.to_excel(args.output, index=False, engine='openpyxl')
                    print(f"\n--- ✅ Final report successfully saved to '{args.output}' ---")
                else:
                    print("\n--- ⚠️ WORKFLOW FINISHED, BUT THE FINAL REPORT WAS EMPTY. ---")
            except Exception as e:
                print(f"\n--- ❌ CRITICAL ERROR saving report to Excel: {e} ---")
        else:
            print("\n--- ⚠️ WORKFLOW DID NOT PRODUCE A FINAL REPORT ---")
            if final_state_result:
                print("Final state was captured, but the expected report data was not found.")
                print("Final state content:", json.dumps(final_state_result, indent=2))