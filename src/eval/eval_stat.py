# import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import argparse
from tqdm import tqdm
from config_loader import PATHS
# from collections import defaultdict
# from src.utils import load_data
# import csv

from config_loader import PATHS, get_evaluation_filepath



import os
import csv
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Set, Tuple

def load_data(path, filetype="jsonl"):
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        if filetype == "jsonl":
            return [json.loads(line.strip()) for line in f if line.strip()]
        elif filetype == "json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {filetype}")

def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")

def parse_scores(item: dict, task_name: str, all_metric_names: Set[str]) -> Tuple[Dict[str, float], List[str]]:
    domain_scores = defaultdict(float)
    metrics_added = []

    if not item or "scores" not in item or not isinstance(item["scores"], dict):
        return domain_scores, metrics_added

    scores = item["scores"]

    if "one_piece" in scores and isinstance(scores["one_piece"], dict):
        for metric, value in scores["one_piece"].items():
            if isinstance(value, (int, float)):
                metric_name = f"one_piece_{metric}".replace(" ", "_").replace(":", "")
                domain_scores[metric_name] += value
                all_metric_names.add(metric_name)
                metrics_added.append(metric_name)


    if "cell_scores" in scores and isinstance(scores["cell_scores"], dict):
        for metric, value in scores["cell_scores"].items():
            if isinstance(value, (int, float)):
                metric_name = f"cell_scores_{metric}"
                domain_scores[metric_name] += value
                all_metric_names.add(metric_name)
                metrics_added.append(metric_name)

    return domain_scores, metrics_added

def process_model(model_name: str, domains: List[str], all_metric_names: Set[str]) -> Tuple[List[dict], List[dict]]:
    domain_results = []
    task_results = []

    model_name_safe = safe_model_name(model_name)
    # data_evaluated_path = get_evaluation_filepath(
    #             current_prompt_setting, current_others, current_model_name, f"eval_details_{domain_name}.jsonl"
    #         )
    
    base_path = PATHS['output_evaluation']
    
    for domain in domains:
        path = os.path.join(base_path, f"{model_name_safe}/{domain}_evaluated.jsonl")
        data = load_data(path, "jsonl")
        if not data:
            continue

        domain_score_sum = defaultdict(float)
        domain_count = 0
        task_score_sum = defaultdict(lambda: defaultdict(float))
        task_count = defaultdict(int)

        for item in data:
            for key, content in item.items():
                task_id = key.split("_")[-1]
                task_name = f"{domain}_{task_id}"

                score_dict, added_metrics = parse_scores(content, task_name, all_metric_names)
                if not score_dict:
                    continue

                domain_count += 1
                task_count[task_name] += 1

                for metric in added_metrics:
                    domain_score_sum[metric] += score_dict[metric]
                    task_score_sum[task_name][metric] += score_dict[metric]

        if domain_count > 0:
            avg_row = {"model": model_name, "domain": domain}
            for metric in all_metric_names:
                score = domain_score_sum.get(metric, 0.0)
                avg_row[metric] = round((score / domain_count) * (100 if any(k in metric for k in ["f1", "precision", "recall"]) else 1), 2)
            domain_results.append(avg_row)

        for task_name, scores in task_score_sum.items():
            count = task_count[task_name]
            avg_row = {"model": model_name, "task_name": task_name}
            for metric in all_metric_names:
                score = scores.get(metric, 0.0)
                avg_row[metric] = round((score / count) * (100 if any(k in metric for k in ["f1", "precision", "recall"]) else 1), 2)
            task_results.append(avg_row)

    return domain_results, task_results

def write_csv(output_path: str, rows: List[dict], fieldnames: List[str]):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def evaluate_all_models(model_names: List[str], domains: List[str]):
    all_domain_results = []
    all_task_results = []
    all_metrics: Set[str] = set()

    for model in model_names:
        domain_rows, task_rows = process_model(model, domains, all_metrics)
        all_domain_results.extend(domain_rows)
        all_task_results.extend(task_rows)

    domain_fields = ["model", "domain"] + sorted(all_metrics)
    task_fields = ["model", "task_name"] + sorted(all_metrics)
    
    output_dir = PATHS['output_evaluation']
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, "all_models_domain_average_scores.csv"), all_domain_results, domain_fields)
    write_csv(os.path.join(output_dir, "all_models_task_average_scores.csv"), all_task_results, task_fields)
    
    model_level_scores = defaultdict(lambda: defaultdict(float))
    model_task_counts = defaultdict(int)

    for row in all_task_results:
        model = row["model"]
        model_task_counts[model] += 1
        for metric in all_metrics:
            model_level_scores[model][metric] += row.get(metric, 0.0)

    all_model_results = []
    for model, scores in model_level_scores.items():
        avg_row = {"model": model}
        for metric in all_metrics:
            score = scores.get(metric, 0.0)
            avg_row[metric] = round(score / model_task_counts[model], 2)
        all_model_results.append(avg_row)

    model_fields = ["model"] + sorted(all_metrics)
    write_csv(os.path.join(output_dir, "all_models_overall_average_scores.csv"), all_model_results, model_fields)


def main():
    parser = argparse.ArgumentParser(description="Summarize model evaluation scores")
    parser.add_argument("--models", nargs="+", required=True, help="List of model names")
    parser.add_argument("--domains", nargs="+", required=True, help="List of domain names (e.g., text2text qa s2t)")

    args = parser.parse_args()

    evaluate_all_models(
        model_names=args.models,
        domains=args.domains,
    )

if __name__ == "__main__":
    main()

"""
Simple Model Evaluation Summary Script

A streamlined script for aggregating and summarizing model evaluation results across domains.
This script provides essential functionality for comparing model performance with clean,
readable output suitable for research and development workflows.

Features:
- Clean aggregation of evaluation scores across models and domains
- Support for both one-piece and cell-level evaluation metrics
- Automatic generation of summary CSV reports
- Simple command-line interface for easy integration
"""

import os
import sys
import csv
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config_loader import PATHS, get_evaluation_filepath


def load_data(path: str, filetype: str = "jsonl") -> List[Dict]:
    """
    Load data from JSON or JSONL files with error handling.
    
    Args:
        path: Path to the data file
        filetype: Type of file to load ('jsonl' or 'json')
        
    Returns:
        List of loaded data items, empty list if file not found or invalid
    """
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            if filetype == "jsonl":
                return [json.loads(line.strip()) for line in f if line.strip()]
            elif filetype == "json":
                return json.load(f)
            else:
                raise ValueError(f"Unsupported file type: {filetype}")
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading {path}: {e}")
        return []


def safe_model_name(model_name: str) -> str:
    """Convert model name to filesystem-safe format."""
    return model_name.replace("/", "_")


def parse_scores(item: Dict, task_name: str, all_metric_names: Set[str]) -> Tuple[Dict[str, float], List[str]]:
    """
    Parse evaluation scores from item data.
    
    Args:
        item: Item data containing evaluation scores
        task_name: Name of the current task (for debugging)
        all_metric_names: Set to track all encountered metric names
        
    Returns:
        Tuple of (parsed_scores_dict, list_of_added_metrics)
    """
    domain_scores = defaultdict(float)
    metrics_added = []

    # Validate input structure
    if not item or "scores" not in item or not isinstance(item["scores"], dict):
        return domain_scores, metrics_added

    scores_data = item["scores"]
    
    # Find the actual scores under predict_* keys
    actual_scores = {}
    for key_name in scores_data.keys():
        if key_name.startswith("predict_"):
            content = scores_data.get(key_name)
            if isinstance(content, dict):
                actual_scores = content
                break
    
    if not actual_scores:
        return domain_scores, metrics_added

    # Parse one-piece evaluation scores
    if "one_piece" in actual_scores and isinstance(actual_scores["one_piece"], dict):
        one_piece_data = actual_scores["one_piece"]
        for metric, value in one_piece_data.items():
            if isinstance(value, (int, float)) and value is not None:
                # Clean metric name for CSV compatibility
                metric_name = f"one_piece_{metric}".replace(" ", "_").replace(":", "")
                domain_scores[metric_name] += float(value)
                all_metric_names.add(metric_name)
                metrics_added.append(metric_name)

    # Parse cell-level evaluation scores
    if "cell_scores" in actual_scores and isinstance(actual_scores["cell_scores"], dict):
        cell_scores_data = actual_scores["cell_scores"]
        for metric, value in cell_scores_data.items():
            if isinstance(value, (int, float)) and value is not None:
                metric_name = f"cell_scores_{metric}"
                domain_scores[metric_name] += float(value)
                all_metric_names.add(metric_name)
                metrics_added.append(metric_name)

    return domain_scores, metrics_added


def process_model(
    model_name: str, 
    domains: List[str], 
    all_metric_names: Set[str],
    prompt_setting: str = "all",
    others: str = ""
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process evaluation results for a single model across specified domains.
    
    Args:
        model_name: Name of the model to process
        domains: List of domains to process
        all_metric_names: Set to track all encountered metrics
        prompt_setting: Prompt setting identifier
        others: Additional run-specific identifiers
        
    Returns:
        Tuple of (domain_results, task_results)
    """
    domain_results = []
    task_results = []

    for domain in tqdm(domains, desc=f"Processing {model_name}", leave=False):
        # Try to load data using the evaluation filepath helper
        try:
            eval_path = get_evaluation_filepath(
                prompt_setting=prompt_setting,
                others=others,
                model_name=model_name,
                eval_filename=f"eval_details_{domain}.jsonl"
            )
        except:
            # Fallback to direct path construction if helper fails
            model_name_safe = safe_model_name(model_name)
            base_path = PATHS.get('output_evaluation', './output/evaluation')
            eval_path = os.path.join(base_path, f"{model_name_safe}/{domain}_evaluated.jsonl")
        
        data = load_data(eval_path, "jsonl")
        if not data:
            print(f"No data found for {model_name} - {domain}")
            continue

        # Initialize aggregation containers
        domain_score_sum = defaultdict(float)
        domain_count = 0
        task_score_sum = defaultdict(lambda: defaultdict(float))
        task_count = defaultdict(int)

        # Process each item in the domain data
        for item_wrapper in data:
            # Handle both old and new data formats
            if 'record_id' in item_wrapper:
                # New format: single item with record_id
                task_key = item_wrapper['record_id']
                content = item_wrapper
                items_to_process = [(task_key, content)]
            else:
                # Old format: wrapper dict with task_key: content
                items_to_process = list(item_wrapper.items())
            
            for task_key, content in items_to_process:
                if not isinstance(content, dict):
                    continue
                
                # Extract task information
                task_parts = task_key.split("_")
                # Handle new format with language suffix
                if len(task_parts) >= 4 and task_parts[-1] in ['en', 'zh']:
                    task_id = task_parts[-2]
                else:
                    task_id = task_parts[-1]
                
                task_name = f"{domain}_{task_id}"

                # Parse scores for this item
                score_dict, added_metrics = parse_scores(content, task_name, all_metric_names)
                if not score_dict:
                    continue

                # Aggregate scores
                domain_count += 1
                task_count[task_name] += 1

                for metric in added_metrics:
                    score_value = score_dict[metric]
                    domain_score_sum[metric] += score_value
                    task_score_sum[task_name][metric] += score_value

        # Generate domain-level averages
        if domain_count > 0:
            avg_row = {"model": model_name, "domain": domain}
            for metric in all_metric_names:
                score = domain_score_sum.get(metric, 0.0)
                # Apply scaling for percentage metrics
                scale_factor = 100 if any(k in metric for k in ["f1", "precision", "recall"]) and not metric.startswith("one_piece") else 1
                avg_row[metric] = round((score / domain_count) * scale_factor, 2)
            domain_results.append(avg_row)

        # Generate task-level averages
        for task_name, scores in task_score_sum.items():
            count = task_count[task_name]
            if count == 0:
                continue
                
            avg_row = {"model": model_name, "task_name": task_name}
            for metric in all_metric_names:
                score = scores.get(metric, 0.0)
                scale_factor = 100 if any(k in metric for k in ["f1", "precision", "recall"]) and not metric.startswith("one_piece") else 1
                avg_row[metric] = round((score / count) * scale_factor, 2)
            task_results.append(avg_row)

    return domain_results, task_results


def write_csv(output_path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    """
    Write data to CSV file with proper error handling.
    
    Args:
        output_path: Path where to save the CSV file
        rows: List of dictionaries containing the data
        fieldnames: List of field names for the CSV header
    """
    if not rows:
        print(f"Warning: No data to write to {output_path}")
        return
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        
        print(f"✅ Successfully wrote {len(rows)} rows to {output_path}")
        
    except IOError as e:
        print(f"❌ Error writing to {output_path}: {e}")
    except Exception as e:
        print(f"❌ Unexpected error writing CSV {output_path}: {e}")


def evaluate_all_models(
    model_names: List[str], 
    domains: List[str],
    prompt_setting: str = "all",
    others: str = "",
    output_dir: Optional[str] = None
) -> None:
    """
    Evaluate all models and generate comprehensive summary reports.
    
    Args:
        model_names: List of model names to evaluate
        domains: List of domains to process
        prompt_setting: Prompt setting identifier
        others: Additional run-specific identifiers  
        output_dir: Custom output directory (optional)
    """
    print(f"🚀 Starting evaluation summary for {len(model_names)} models across {len(domains)} domains")
    
    # Initialize result containers
    all_domain_results = []
    all_task_results = []
    all_metrics: Set[str] = set()

    # Process each model
    for model in tqdm(model_names, desc="Processing models"):
        domain_rows, task_rows = process_model(
            model, domains, all_metrics, prompt_setting, others
        )
        all_domain_results.extend(domain_rows)
        all_task_results.extend(task_rows)

    if not all_metrics:
        print("❌ No metrics found in any of the processed data")
        return

    # Prepare output directory
    if output_dir is None:
        base_dir = PATHS.get('output_evaluation', './output/evaluation')
        if others:
            output_dir = os.path.join(base_dir, f"summary_{prompt_setting}_{others}")
        else:
            output_dir = os.path.join(base_dir, f"summary_{prompt_setting}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Sort metrics for consistent output
    sorted_metrics = sorted(all_metrics)
    print(f"📊 Found {len(sorted_metrics)} metrics: {sorted_metrics[:3]}{'...' if len(sorted_metrics) > 3 else ''}")

    # Define field names for different report types
    domain_fields = ["model", "domain"] + sorted_metrics
    task_fields = ["model", "task_name"] + sorted_metrics

    # Write domain and task level summaries
    write_csv(
        os.path.join(output_dir, "all_models_domain_average_scores.csv"), 
        all_domain_results, 
        domain_fields
    )
    write_csv(
        os.path.join(output_dir, "all_models_task_average_scores.csv"), 
        all_task_results, 
        task_fields
    )

    # Generate overall model performance summary
    print("📈 Generating overall model performance summary...")
    model_level_scores = defaultdict(lambda: defaultdict(float))
    model_task_counts = defaultdict(int)

    for row in all_task_results:
        model = row["model"]
        model_task_counts[model] += 1
        for metric in all_metrics:
            model_level_scores[model][metric] += row.get(metric, 0.0)

    # Calculate model-level averages
    all_model_results = []
    for model, scores in model_level_scores.items():
        task_count = model_task_counts[model]
        if task_count == 0:
            continue
            
        avg_row = {"model": model}
        for metric in all_metrics:
            score = scores.get(metric, 0.0)
            avg_row[metric] = round(score / task_count, 2)
        all_model_results.append(avg_row)

    # Write overall model summary
    model_fields = ["model"] + sorted_metrics
    write_csv(
        os.path.join(output_dir, "all_models_overall_average_scores.csv"), 
        all_model_results, 
        model_fields
    )

    # Print summary statistics
    print("\n📋 Summary Statistics:")
    print(f"   • Total models processed: {len(model_names)}")
    print(f"   • Total domains processed: {len(domains)}")
    print(f"   • Total metrics found: {len(sorted_metrics)}")
    print(f"   • Total domain results: {len(all_domain_results)}")
    print(f"   • Total task results: {len(all_task_results)}")
    print(f"\n✅ Evaluation summary completed successfully!")


def main():
    """Main entry point for the evaluation summary script."""
    parser = argparse.ArgumentParser(
        description="Summarize model evaluation scores across domains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_summary.py --models gpt-4 claude-3 --domains academic legal financial
  python simple_summary.py --models model1 model2 --domains academic --prompt_setting all --others rag
        """
    )
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        required=True, 
        help="List of model names to evaluate"
    )
    parser.add_argument(
        "--domains", 
        nargs="+", 
        required=True, 
        help="List of domain names (e.g., academic, legal, financial)"
    )
    parser.add_argument(
        "--prompt_setting", 
        default="all", 
        help="Prompt setting identifier (default: all)"
    )
    parser.add_argument(
        "--others", 
        default="", 
        help="Additional run-specific identifiers (default: empty)"
    )
    parser.add_argument(
        "--output_dir", 
        help="Custom output directory (optional)"
    )

    args = parser.parse_args()
    
    print("🎯 Model Evaluation Summary")
    print(f"Models: {args.models}")
    print(f"Domains: {args.domains}")
    print(f"Prompt Setting: {args.prompt_setting}")
    print(f"Others: {args.others if args.others else 'None'}")
    print("-" * 50)

    try:
        evaluate_all_models(
            model_names=args.models,
            domains=args.domains,
            prompt_setting=args.prompt_setting,
            others=args.others,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())