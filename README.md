# AOE: Arranged and Organized Extraction Benchmark

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/tianyumyum/AOE/tree/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for the paper: **"Beyond Isolated Dots: Benchmarking Structured Table Construction as Deep Knowledge Extraction"**.

## Abstract

Arranged and Organized Extraction (AOE) is a comprehensive benchmark designed to evaluate structured table construction as a form of deep knowledge extraction. Unlike traditional benchmarks that focus on isolated information pieces, AOE emphasizes the holistic understanding and reconstruction of tabular structures from diverse textual sources across legal, financial, and academic domains.

## Project Structure

```
AOE/
├── config/
│   ├── paths.yaml                          # Path configurations for data and outputs
│   ├── prompt_templates_eval.yaml          # Evaluation prompt templates
│   ├── prompt_templates_one_piece_csv.yaml # Generation prompt templates
│   └── task_config.yaml                    # Task-specific configurations
├── config_loader.py                        # Configuration management module
├── README.md                               # This file
├── requirements.txt                        # Python dependencies
├── scripts/
│   ├── run_eval.sh                         # Evaluation pipeline script
│   └── run_generate.sh                     # Generation pipeline script
└── src/
    ├── eval/
    │   ├── eval_parallel.py                # Parallel evaluation execution
    │   ├── eval_stat_detail.py             # Detailed statistical analysis
    │   └── eval_stat.py                    # Statistical evaluation summary
    ├── generate/
    │   └── generate_parallel.py            # Parallel generation execution
    ├── llm/
    │   ├── call_llm.py                     # LLM interface and API calls
    │   └── __init__.py
    └── utils.py                            # Data processing utilities
```


## Setup and Installation


### 1. Clone Repository
```bash
git clone https://github.com/tianyumyum/AOE.git
cd AOE
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Create data directory and download from Hugging Face:

```bash
mkdir data

# Download table data (main benchmark file)
wget https://huggingface.co/datasets/tianyumyum/AOE/resolve/main/table_data/all_AOE_tables.jsonl -P data/

# Download source documents archive
wget https://huggingface.co/datasets/tianyumyum/AOE/resolve/main/documents.tar -P data/

# Extract documents
tar -xvf data/documents.tar -C data/
```

Final data structure:
```
data/
├── all_AOE_tables.jsonl    # Main benchmark dataset
├── documents.tar           # Compressed source documents
├── law_docx/              # Legal domain documents
├── academic/              # Academic domain documents
└── financial/             # Financial domain documents
```

### 4. Configure Paths
Edit `config/paths.yaml` and set your project root path:

```yaml
# Essential: Set this to your AOE project directory
base_path: "/path/to/your/AOE"

paths:
  root_key: "{base_path}/outputs"
  data:
    data_input: "{base_path}/data/all_AOE_tables.jsonl"
    law_docx: "{base_path}/data/law_docx"
    academic: "{base_path}/data/academic"
    financial: "{base_path}/data/financial"
  logs: "{base_path}/logs"

# API Configuration (optional)
api_key: ""
base_url: ""
eval_model_name: "your-evaluation-model"
```

## Running Experiments

### Generation Pipeline
Execute the generation script to produce model predictions:

```bash
bash scripts/run_generate.sh
```

**Key Parameters:**
- `--model-name`: Target model identifier (e.g., "gpt-4", "claude-3")
- `--prompt_setting`: Prompt configuration (`all`, `no_teach`, `no_cot`)
  - `all`: Complete prompt with teaching examples and chain-of-thought
  - `no_teach`: No demonstration examples
  - `no_cot`: No chain-of-thought reasoning
- `--others`: Additional processing modes
  - `""` (empty): Standard processing
  - `rag`: Enable RAG-enhanced processing
  - `with_schema`: Include schema information
- `--domains`: Target domains (`legal`, `financial`, `academic`)
- `--max_workers`: Parallel processing threads (default: 4)

**Example configurations:**
```bash
# Baseline evaluation with all prompt components
"${MODEL_NAME};;"

# RAG-enhanced processing with full prompts
"${MODEL_NAME};rag;all"

# Schema-aware processing without cot examples
"claude-3;;no_cot"
```

### Evaluation Pipeline
Evaluate generated predictions against ground truth:

```bash
bash scripts/run_eval.sh
```

**Evaluation Parameters:**
- `--eval_model_name`: Model used for semantic evaluation
- `--input_dir`: Directory containing prediction files
- `--output_dir`: Directory for evaluation results

### Statistical Analysis
Generate comprehensive evaluation statistics:

```bash
python src/eval/eval_stat.py --results_dir outputs/ --output_file evaluation_summary.csv
```

## RAG Configuration Details

For RAG-enhanced experiments (`--others rag`), the system uses:

**Framework:** LangChain v0.3 with the following configuration:
- **Text Splitting:** Documents (4-10 per table) are processed using `RecursiveCharacterTextSplitter`
  - `chunk_size=512`: Optimal balance between context and granularity
  - `overlap=100`: Ensures context continuity across chunks
- **Vectorization:** BGE-M3 model converts text blocks to dense vector representations
- **Storage:** Chroma database provides efficient indexing and persistent caching
- **Retrieval:** Each table query retrieves the most semantically relevant information blocks from indexed documents

This configuration ensures comprehensive information coverage while maintaining computational efficiency through persistent vector caching.

## Evaluation Metrics

AOE employs a comprehensive multi-faceted automated evaluation pipeline that assesses generated tables from basic parsability to fine-grained cell-level accuracy. The evaluation consists of three primary metrics:

### 1. CSV Parsability
A binary indicator measuring whether the model's output can be successfully parsed into a structured format (e.g., a CSV file readable by pandas DataFrame). This **Pass Rate** reflects the model's basic ability to follow formatting instructions and instruction-following capabilities.

### 2. Overall Quality (LLM-Assessed)
Large Language Models serve as evaluators to assess overall table quality, assigning percentage scores (0-100) based on four key dimensions:

- **Intent Understanding**: Extent to which the table fulfills task objectives
- **Schema Construction**: Logical coherence, relevance, and comprehensiveness of selected columns
- **Content Accuracy & Completeness**: Overall correctness and coverage of information from source documents
- **Format Compliance**: Adherence to structural expectations beyond basic parsability

### 3. Cell F1-Score
Quantitative measurement of extracted content accuracy through a multi-step pipeline:

**(a) Column Alignment**: Establishes comparison basis by aligning predicted table columns with ground-truth columns using predefined `target_value_columns`.

**(b) Row Matching**: LLM-assisted approach that first attempts exact match, then uses semantic comparison to resolve discrepancies (abbreviations, aliases, formatting differences).

**(c) Cell Value Evaluation**: LLM-based expert evaluation with scoring criteria:
- **Semantic Equivalence (1.0)**: Full score for semantically identical content despite formatting differences
- **Factual Inconsistency (0.0)**: Zero score for clearly incorrect information  
- **List Content Overlap (0.0-1.0)**: Proportional score based on correctly identified items
- **Partial Information Match (0.1-0.9)**: Partial scores for partially correct information
- **Null Value Handling**: Consistent handling of empty or N/A values

## Evaluation Example

Here's a concrete example demonstrating our evaluation pipeline:

### Ground Truth Table
```csv
文档名称,报表所属期,资产总额(元),资产总额增长率(%),营业收入(元),营业收入增长率(%),所有者权益(元),加权平均净资产收益率(%)
美的集团2020年年度报告摘要,2020,"360,382,603,000",,"284,221,249,000",,"117,516,260,000",24.95
美的集团2021年年度报告摘要,2021,"387,946,104,000",7.65,"341,233,208,000",20.06,"124,868,124,000",24.09
美的集团2022年年度报告摘要,2022,"422,555,267,000",8.92,"343,917,531,000",0.79,"142,935,236,000",22.21
美的集团2023年年度报告摘要,2023,"486,038,184,000",15.02,"372,037,280,000",8.18,"162,878,825,000",22.23
```

### Predicted Table  
```csv
文档名称,报表所属期,资产总额(元),资产总额增长率(%),营业收入(元),营业收入增长率(%),所有者权益(元)
美的集团2020年年度报告摘要.md,2020年,"360,382,603,000",19.35,"284,221,249,000",2.16,"117,516,260,000"
美的集团2021年年度报告摘要.md,2021年,"387,946,104,000",7.65,"341,233,208,000",20.06,"124,868,124,000"
美的集团2022年年度报告摘要.md,2022年,"422,555,267,000",8.92,"343,917,531,000",0.79,"142,935,236,000"
美的集团2023年年度报告摘要.md,2023年,"486,038,184,000",15.02,"372,037,280,000",8.18,"162,878,825,000"
```

### Evaluation Results
```json
{
    "if_csv_parsed": {
        "direct": 1,
        "mapped": 1
    },
    "one_piece": {
        "score1": 85,  // Intent Understanding
        "score2": 80,  // Schema Construction  
        "score3": 75,  // Content Accuracy & Completeness
        "score4": 80   // Format Compliance
    },
    "cell_scores": {
        "column_recall": 1.0,
        "row_precision": 1.0,
        "row_recall": 1.0,
        "row_f1": 1.0,
        "cell_precision": 0.643,
        "cell_recall": 0.643,
        "cell_f1": 0.643,
        "total_compared_cells_in_matched": 28,
        "average_cell_score_in_matched": 0.643
    }
}
```

This example shows perfect structural alignment (CSV parsability: 1.0, row matching: 1.0) but moderate cell-level accuracy (0.643), indicating some content discrepancies while maintaining correct table structure.

## Model Configuration Examples

The benchmark supports various model configurations. Example setups:

```yaml
# config/model_configs.yaml
models:
    model_name: "Llama-3-8B-Instruct"
    api_key: "${API_KEY}"
    base_url: ""
    max_tokens: 4096  # Set to model's maximum context window  
    temperature: 0.1  # Uses model's default temperature setting
```

**Note**: `max_tokens` is configured to each model's maximum allowable context window as specified in official documentation. Temperature settings use each model's default values to ensure consistent behavior across different model providers.
