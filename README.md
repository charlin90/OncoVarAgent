# 🧬 OncoVarAgent: An Autonomous Research Agent for Uncovering Actionable Therapeutic Evidence for Somatic Cancer Variants Lacking Established Guidance.

OncoVarAgent is an autonomous AI agent built with Large Language Models (LLMs) and LangGraph, designed for automated, in-depth therapeutic evidence discovery of somatic cancer variants lacking established guidance.

While traditional annotation tools like the OncoKB Annotator provide an excellent baseline for known biomarkers, researchers often face a manual, time-consuming process of literature review and clinical trial searches for variants without standard-of-care drugs. OncoVarAgent aims to automate this complex research process.

[![License: MIT](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-390/)

---

## ✨ Core Features

-   **Automated Workflow**: End-to-end automation, from an input MAF/TSV file to a comprehensive Excel report.
-   **Intelligent Triage**: The agent intelligently decides whether to perform a deep dive. Variants with clear therapeutic options or benign classifications are fast-tracked, saving time and computational resources.
-   **Multi-Tool Research Agent**: For variants requiring investigation, the agent acts like a domain expert, autonomously planning and executing research steps using tools like `pubmed_search` and `query_clinical_trials`.
-   **Evidence Synthesis & Reporting**: The agent synthesizes information from all sources (OncoKB, literature, trials) to generate a structured, well-cited analytical report.
-   **Highly Configurable**: Easily configure your LLM endpoints, model names, and API keys via a `.env` file.

## ⚙️ Technical Architecture

OncoVarAgent's core is a state machine built with **LangGraph**. The workflow proceeds as follows:

1.  **Initialization (Annotator Node)**: The workflow starts by annotating the input file with the OncoKB Annotator to get baseline information for all variants.
2.  **Looping & Triage (Routing Logic)**:
    -   The system processes one variant at a time from the annotated list.
    -   **Decision Point**: A crucial routing step decides the path based on OncoKB results:
        -   If the variant has known drug associations or is classified as `(Likely) Neutral`, it **skips** the deep dive and proceeds directly to a simple formatting step.
        -   Otherwise, it proceeds to the full research node.
3.  **Deep Research (Deep Research Node)**:
    -   A **ReAct (Reasoning and Acting)** style agent is invoked.
    -   This agent receives a complex prompt outlining a multi-phase research strategy, guiding it to investigate the variant's function, cancer-specific therapies, and pan-cancer evidence.
    -   It autonomously calls tools, analyzes their outputs, and plans subsequent actions until it has gathered sufficient evidence.
    -   Finally, it produces a comprehensive summary of its findings, citing all evidence.
4.  **Report Synthesis (Synthesizer Node)**:
    -   This node combines the baseline OncoKB data with the research agent's summary.
    -   It uses a LLM to format all the information into a strict JSON schema, creating the final report for a single variant.
5.  **Finalization (Final Combiner Node)**:
    -   Once all variants are processed, this node consolidates the individual reports into a final dataset ready for export to an Excel file.

## 🔧 Installation & Configuration

Follow these steps to set up your environment.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/OncoVarAgent.git
cd OncoVarAgent
```

### 2. Create a Python Virtual Environment

```bash
conda create -n oncovaragent python=3.10
conda activate oncovaragent
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the OncoKB Annotator

OncoVarAgent relies on the official `MafAnnotator.py` script from OncoKB.
-   Go to the [oncokb-annotator GitHub repository](https://github.com/oncokb/oncokb-annotator).
-   Download or clone the repository to your local machine.
-   Make a note of the **absolute path** to the `MafAnnotator.py` script.

### 5. Configure Environment Variables

This is the most critical step.

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
2.  Open the newly created `.env` file and fill in your credentials and paths:

    ```dotenv
    # --- LLM Provider Configuration ---
    # Your LLM provider's base URL. For OpenAI, this is "https://api.openai.com/v1"
    LLM_BASE_URL="https://api.mulerun.com/v1"

    # Your LLM API key
    LLM_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    # --- LLM Model Selection ---
    # The model for the creative ReAct agent (needs strong reasoning and tool use).
    # Examples: "gpt-5"
    LLM_MODEL_NAME="gpt-5"

    # --- OncoKB Annotator Configuration ---
    # Your OncoKB API Token, obtained from https://www.oncokb.org/apiAccess
    ONCOKB_API_TOKEN="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

    # The absolute path to the MafAnnotator.py script you downloaded in Step 4.
    # Example for Linux/macOS: "/home/user/tools/oncokb-annotator/MafAnnotator.py"
    # Example for Windows: "C:\\Users\\user\\tools\\oncokb-annotator\\MafAnnotator.py"
    ONCOKB_ANNOTATOR_PATH="/path/to/your/oncokb-annotator/MafAnnotator.py"
    ```

## ▶️ How to Use

### 1. Prepare Your Input File

Create a tab-separated values (TSV) file (e.g., `my_variants.txt`). The file must contain columns for the gene symbol, protein change, and cancer type. Cancer type must be same.

**Example `my_variants.txt`:**
```tsv
Hugo_Symbol	HGVSp_Short	Cancer_Type
EGFR	p.L858R	Non-Small Cell Lung Cancer
TP53	p.R175H	Non-Small Cell Lung Cancer
```
*Note: You can specify different column names via command-line arguments.*

### 2. Run the Script

From your terminal, with the virtual environment activated, run the agent:

```bash
python OncoVarAgent.py \
  --input-txt my_variants.txt \
  --output variant_interpretation_report.xlsx
```

The agent will begin its process, and you will see detailed logs in your terminal, including node execution, the agent's thought process, and tool calls.

### Command-Line Arguments

-   `--input-txt` (Required): Path to your input TSV file.
-   `--output` (Optional): Name for the output Excel file. Defaults to `variant_interpretation_report.xlsx`.
-   `--gene-col` (Optional): Column name for the gene symbol in your input file. Defaults to `Hugo_Symbol`.
-   `--protein-change-col` (Optional): Column name for the HGVSp notation. Defaults to `HGVSp_Short`.
-   `--cancer-type-col` (Optional): Column name for the cancer type. Defaults to `Cancer_Type`.

## 📄 Output Interpretation

The script generates an Excel file with the following columns, providing a comprehensive view of each variant.

| Column Name                        | Description                                                                     |
| ---------------------------------- | ------------------------------------------------------------------------------- |
| `gene`                             | Gene symbol.                                                                    |
| `protein_change`                   | Protein change in HGVSp format.                                                 |
| `cancer_type`                      | The cancer type context for the interpretation.                                 |
| **--- OncoKB Baseline Data ---**   |                                                                                 |
| `oncokb_ONCOGENIC`                 | OncoKB's oncogenicity classification.                                           |
| `oncokb_AMP_TIER`                  | The AMP/ASCO/CAP tier mapped from the OncoKB level of evidence.                   |
| `oncokb_Drugs`                     | Drugs (sensitivity/resistance) associated with the variant in OncoKB.           |
| `oncokb_MUTATION_EFFECT`           | The functional effect of the mutation (e.g., Gain-of-function).                 |
| `oncokb_MUTATION_EFFECT_CITATIONS` | Supporting PubMed IDs for the mutation effect.                                  |
| **--- OncoVarAgent Deep Analysis ---** |                                                                                 |
| `OncoVarAgent_Drugs`               | Potential drugs identified by the agent, formatted as `Drug(Status, Evidence)`. |
| `OncoVarAgent_Support_Literatures` | A comma-separated list of PubMed IDs (PMIDs) supporting the agent's analysis.   |
| `OncoVarAgent_Clinical_Trial_IDs`  | A comma-separated list of relevant Clinical Trial IDs (NCT IDs).                |
| `OncoVarAgent_Brief_Report`        | A 2-3 sentence executive summary of the agent's key findings.                   |
| `OncoVarAgent_Deep_Report`         | The full, detailed analysis from the agent, including its reasoning and evidence. |

## ⚖️ License

This project is licensed under the GPLv3 License. See the `LICENSE` file for details.
