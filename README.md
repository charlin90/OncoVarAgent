# 🧬 OncoVarAgent: AI Agent for Somatic Variant Interpretation

OncoVarAgent 是一个基于大型语言模型（LLM）和 LangGraph 构建的自主 AI 代理，旨在自动化和深化癌症体细胞突变的临床意义解读。

传统的变异注释工具（如 OncoKB Annotator）为已知的生物标志物提供了极好的基线信息。然而，对于那些没有明确分级或药物关联的“潜在可操作”变异，研究人员通常需要手动查阅大量文献和临床试验数据。OncoVarAgent 的目标就是自动化这一耗时且复杂的研究过程。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

---

## ✨ 核心特性

-   **自动化工作流**: 从输入一个包含变异列表的 MAF/TSV 文件开始，到输出一个包含深度解读的 Excel 报告，全程自动化。
-   **智能分流**: 代理会首先利用 OncoKB 的结果进行智能判断。对于已有明确治疗方案或被认为是良性的变异，代理会跳过深度研究，从而节省计算资源和时间。
-   **多工具代理研究**: 对于需要深度研究的变异，代理会像一个领域专家一样，自主规划并执行一系列研究步骤：
    -   **PubMed**: 搜索与变异功能、治疗相关的最新文献。
    -   **ClinicalTrials.gov**: 查询与特定药物、基因或癌种相关的临床试验。
-   **证据综合与报告生成**: 代理能够综合从多个来源（OncoKB、文献、临床试验）获得的信息，生成一个结构化的、包含证据引用的深度分析报告。
-   **高度可配置**: 通过 `.env` 文件，用户可以轻松配置自己的 LLM API、模型名称和 OncoKB 凭证。

## ⚙️ 技术架构

OncoVarAgent 的核心是一个使用 **LangGraph** 构建的状态机（StateGraph）。工作流程如下：

1.  **初始化 (Annotator Node)**: 使用 OncoKB Annotator 对输入的 MAF 文件进行初步注释，获取基线信息。
2.  **循环与分流 (Routing Logic)**:
    -   系统从待处理列表中取出一个变异。
    -   **决策点**: 根据 OncoKB 的结果判断：
        -   如果变异已有明确的药物信息 (`Drugs` 字段非空) 或被分类为良性/可能良性，则**跳过**深度研究，直接进入格式化步骤。
        -   否则，进入深度研究流程。
3.  **深度研究 (Deep Research Node)**:
    -   一个 **ReAct (Reasoning and Acting)** 风格的代理被激活。
    -   该代理接收到一个复杂的任务指令，要求它分阶段、有逻辑地研究该变异。
    -   代理会自主决定调用 `pubmed_search` 和 `query_clinical_trials` 工具，分析返回结果，并根据分析规划下一步行动。
    -   最终，代理会输出一份综合性的研究摘要。
4.  **报告合成 (Synthesizer Node)**:
    -   此节点将 OncoKB 的基线数据与深度研究代理生成的摘要结合起来。
    -   调用一个“事实型”LLM，根据严格的 JSON Schema 将所有信息格式化为最终的单变异报告。
5.  **循环结束与合并 (Final Combiner Node)**:
    -   当所有变异都处理完毕后，此节点将所有单个报告合并，并准备最终输出。

## 🔧 安装与配置

在运行脚本之前，请确保你已完成以下环境配置。

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/OncoVarAgent.git
cd OncoVarAgent
```

### 2. 创建 Python 虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. 安装依赖

首先，请在项目根目录创建一个 `requirements.txt` 文件。
*如果你已经安装了所有需要的包，可以运行 `pip freeze > requirements.txt` 来生成此文件。*

`requirements.txt` 文件应包含以下内容：
```
langchain
langgraph
langchain-openai
pandas
python-dotenv
requests
openpyxl
```

然后运行安装命令：
```bash
pip install -r requirements.txt
```

### 4. 下载 OncoKB Annotator

OncoVarAgent 依赖于 OncoKB 官方提供的 `MafAnnotator.py` 脚本。
-   前往 [oncokb-annotator GitHub 仓库](https://github.com/oncokb/oncokb-annotator)。
-   下载该仓库，并记下 `MafAnnotator.py` 脚本在你本地文件系统中的**完整路径**。

### 5. 配置环境变量

这是最关键的一步。

1.  将 `.env.example` 文件复制为 `.env` 文件：
    ```bash
    cp .env.example .env
    ```
2.  打开 `.env` 文件，并填入你自己的凭证和路径：

    ```dotenv
    # --- LLM Provider Configuration ---
    # 你的大模型API服务地址
    LLM_BASE_URL="https://api.mulerun.com/v1"

    # 你的大模型API密钥
    LLM_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"

    # --- LLM Model Selection ---
    # 用于ReAct Agent深度研究的模型 (推荐使用能力强的模型, 如 GPT-4, Claude 3 Opus)
    LLM_CREATIVE_MODEL_NAME="gpt-4-turbo"

    # 用于最终报告合成的模型 (推荐使用遵循指令能力强的模型)
    LLM_FACTUAL_MODEL_NAME="gpt-4-turbo"

    # --- OncoKB Annotator Configuration ---
    # 从 https://www.oncokb.org/apiAccess 获取你的 OncoKB API Token
    ONCOKB_API_TOKEN="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

    # 你在步骤4中下载的 MafAnnotator.py 脚本的完整路径
    # 例如: "/home/user/tools/oncokb-annotator/MafAnnotator.py"
    ONCOKB_ANNOTATOR_PATH="/path/to/your/oncokb-annotator/MafAnnotator.py"
    ```

## ▶️ 如何使用

1.  **准备输入文件**:
    创建一个 TSV (Tab-Separated Values) 文件，例如 `my_variants.txt`。文件必须包含基因符号、蛋白质改变和癌症类型的列。默认列名如下：

    ```tsv
    Hugo_Symbol	HGVSp_Short	Cancer_Type
    BRAF	p.V600E	Melanoma
    EGFR	p.L858R	Non-Small Cell Lung Cancer
    TP53	p.R175H	Ovarian Cancer
    ```
    *你可以通过命令行参数指定不同的列名。*

2.  **运行脚本**:
    打开终端，激活虚拟环境，然后运行以下命令：
    ```bash
    python OncoVarAgent.py \
      --input-txt my_variants.txt \
      --output variant_interpretation_report.xlsx
    ```
    代理将会开始执行，你会在终端看到详细的运行日志，包括每个节点的执行情况、代理的思考过程和工具调用。

### 命令行参数

-   `--input-txt`: (必需) 输入的 TSV 文件路径。
-   `--output`: (可选) 输出的 Excel 报告文件名。默认为 `variant_interpretation_report.xlsx`。
-   `--gene-col`: (可选) 输入文件中基因符号的列名。默认为 `Hugo_Symbol`。
-   `--protein-change-col`: (可选) 输入文件中蛋白质改变的列名。默认为 `HGVSp_Short`。
-   `--cancer-type-col`: (可选) 输入文件中癌症类型的列名。默认为 `Cancer_Type`。


## 📄 输出解读

脚本执行完毕后，会生成一个名为 `variant_interpretation_report.xlsx` 的 Excel 文件。文件包含以下列：

| 列名                               | 描述                                                                    |
| ---------------------------------- | ----------------------------------------------------------------------- |
| `gene`                             | 基因符号。                                                              |
| `protein_change`                   | HGVSp 格式的蛋白质改变。                                                |
| `cancer_type`                      | 癌症类型。                                                              |
| **--- OncoKB 基线数据 ---**        |                                                                         |
| `oncokb_ONCOGENIC`                 | OncoKB 对变异致癌性的分类。                                             |
| `oncokb_AMP_TIER`                  | 根据 OncoKB 证据等级映射的 AMP/ASCO/CAP 分级。                            |
| `oncokb_Drugs`                     | OncoKB 中与该变异相关的药物（敏感性/耐药性）。                           |
| `oncokb_MUTATION_EFFECT`           | OncoKB 对突变功能的描述 (例如, Gain-of-function)。                       |
| `oncokb_MUTATION_EFFECT_CITATIONS` | 支持突变功能描述的 PubMed ID。                                          |
| **--- OncoVarAgent 深度分析 ---**  |                                                                         |
| `OncoVarAgent_Drugs`               | 代理从文献和临床试验中发现的潜在药物，格式为 `Drug(Status, Evidence)`。    |
| `OncoVarAgent_Support_Literatures` | 支持其分析结论的 PubMed ID 列表 (PMID)。                                 |
| `OncoVarAgent_Clinical_Trial_IDs`  | 相关的临床试验 ID 列表 (NCT ID)。                                         |
| `OncoVarAgent_Brief_Report`        | 代理生成的 2-3 句话核心摘要。                                           |
| `OncoVarAgent_Deep_Report`         | 代理完整的、包含内在逻辑和证据引用的详细分析报告。                        |

## ⚖️ 许可证

本项目采用 [GPL License](./LICENSE) 开源。
