"""
Model Evaluation Summary Script

This script aggregates and summarizes evaluation results from multiple models across different domains,
providing comprehensive analysis including fixed-level and dynamic-level performance breakdowns.

Features:
- Aggregates scores across domains and tasks
- Analyzes performance by document count, average length, and answer count levels
- Supports both fixed thresholds and dynamic quantile-based analysis
- Generates comprehensive CSV reports for performance comparison
"""

import os
import csv
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any
import sys
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config_loader import (
    PATHS, 
    task_config, 
    get_predict_result_filepath, 
    get_evaluation_filepath, 
    get_intermediate_results_filepath, 
    get_run_specific_base_dir
)


def get_doc_count_level_fixed(count: int) -> str:
    """Categorize document count into fixed levels."""
    if count <= 5:
        return "0-5"
    elif 6 <= count <= 10:
        return "6-10"
    else:
        return ">10"


def get_average_length_level_fixed(avg_len: float) -> str:
    """Categorize average document length into fixed levels."""
    if avg_len < 10000:
        return "<10000"
    elif 10000 <= avg_len <= 20000:
        return "10000-20000"
    else:
        return ">20000"


def get_answers_count_level_fixed(count: int) -> str:
    """Categorize answer count into fixed levels."""
    if count <= 0:
        return "0"
    elif count <= 3:
        return "1-3"
    elif count <= 7:
        return "4-7"
    else:
        return ">7"


def get_level_dynamic(value: float, thresholds: List[float], labels: List[str]) -> str:
    """
    Assign a label based on dynamic thresholds (e.g., quantiles).
    
    Args:
        value: The value to categorize
        thresholds: Sorted list of threshold values
        labels: List of labels for each bin
        
    Returns:
        Appropriate label for the value
    """
    if not thresholds or not labels or len(thresholds) != len(labels) - 1:
        return "unknown_dynamic_level"
        
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return labels[i]
    return labels[-1]


def load_data(path: str, filetype: str = "jsonl") -> List[Dict]:
    """Load data from JSON or JSONL files."""
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
        item: Item data containing scores
        task_name: Name of the current task
        all_metric_names: Set to track all encountered metrics
        
    Returns:
        Tuple of (parsed_scores_dict, list_of_metrics_added)
    """
    domain_scores = defaultdict(float)
    metrics_added = []
    
    if "scores" not in item or not isinstance(item["scores"], dict):
        return domain_scores, metrics_added

    scores_data = item["scores"]
    scores = {}

    # Find the actual scores dictionary under predict_* keys
    for key_name in scores_data.keys():
        if key_name.startswith("predict_"):
            actual_scores_content = scores_data.get(key_name)
            if isinstance(actual_scores_content, dict):
                scores = actual_scores_content
                break

    # Parse CSV parsing metrics
    if "if_csv_parsed" in scores and isinstance(scores.get("if_csv_parsed"), dict):
        csv_parsed = scores["if_csv_parsed"]
        for metric, value in csv_parsed.items():
            if isinstance(value, (int, float)):
                metric_name = f"if_csv_{metric}"
                domain_scores[metric_name] += float(value)
                all_metric_names.add(metric_name)
                metrics_added.append(metric_name)
    
    # Parse one-piece evaluation scores
    if "one_piece" in scores and isinstance(scores.get("one_piece"), dict):
        one_piece = scores["one_piece"]
        score_weights = {"score1": 2, "score2": 3, "score3": 3, "score4": 2}
        total_weight = sum(score_weights.values())
        weighted_sum = 0
        
        for score_key_op, weight in score_weights.items():
            value = one_piece.get(score_key_op, 0.0)
            if value is None:
                value = 0.0
            metric_name = f"one_piece_{score_key_op}"
            domain_scores[metric_name] += float(value)
            all_metric_names.add(metric_name)
            metrics_added.append(metric_name)
            weighted_sum += float(value) * weight
        
        if total_weight > 0:
            overall_score = round(weighted_sum / total_weight, 2)
            metric_name = "one_piece_overall_total_score"
            domain_scores[metric_name] = overall_score
            all_metric_names.add(metric_name)
            metrics_added.append(metric_name)
    
    # Parse cell-level evaluation scores
    if "cell_scores" in scores and isinstance(scores.get("cell_scores"), dict):
        cell_scores_data = scores["cell_scores"]
        for metric, value in cell_scores_data.items():
            if isinstance(value, (int, float)):
                metric_name = f"cell_scores_{metric}"
                domain_scores[metric_name] += float(value)
                all_metric_names.add(metric_name)
                metrics_added.append(metric_name)
    
    return domain_scores, metrics_added


def process_model(
    model_name: str, 
    prompt_setting: str, 
    others: str, 
    domains: List[str], 
    all_metric_names: Set[str]
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[Dict[str, Any]]]:
    """
    Process evaluation results for a single model across all domains.
    
    Returns:
        Tuple containing:
        - domain_results: Aggregated scores by domain
        - task_results: Aggregated scores by task
        - domain_doc_count_level_results: Scores by document count levels
        - domain_average_length_level_results: Scores by average length levels
        - domain_answers_count_level_results: Scores by answer count levels
        - detailed_item_data: Raw item data for further analysis
    """
    domain_results = []
    task_results = []
    domain_doc_count_level_results = []
    domain_average_length_level_results = []
    domain_answers_count_level_results = []
    detailed_item_data_for_model = []

    for domain in tqdm(domains, desc=f"Processing domains for {model_name}", leave=False, unit="domain"):
        eval_path = get_evaluation_filepath(
            prompt_setting=prompt_setting,
            others=others,
            model_name=model_name,
            eval_filename=f"eval_details_{domain}.jsonl"
        )
        data = load_data(eval_path, "jsonl")
        
        if not data:
            print(f"No data found for {model_name} - {domain}")
            continue
        
        print(f"Processing {len(data)} items for {model_name} - {domain}")
        
        # Initialize aggregation containers
        domain_score_sum = defaultdict(float)
        domain_item_count = 0
        task_score_sum = defaultdict(lambda: defaultdict(float))
        task_item_count = defaultdict(int)

        # Fixed-level aggregation containers
        domain_score_sum_doc_count_fixed = defaultdict(lambda: defaultdict(float))
        domain_tally_doc_count_fixed = defaultdict(int)
        domain_score_sum_avg_len_fixed = defaultdict(lambda: defaultdict(float))
        domain_tally_avg_len_fixed = defaultdict(int)
        domain_score_sum_answers_fixed = defaultdict(lambda: defaultdict(float))
        domain_tally_answers_fixed = defaultdict(int)

        for item_wrapper in tqdm(data, desc=f"Items in {domain}", leave=False, unit="item"):
            # Handle both old and new data formats
            if 'record_id' in item_wrapper:
                # New format
                task_key_in_item = item_wrapper['record_id']
                content = item_wrapper
            else:
                # Old format
                if not isinstance(item_wrapper, dict) or len(item_wrapper) != 1:
                    continue
                task_key_in_item = list(item_wrapper.keys())[0]
                content = item_wrapper[task_key_in_item]
            
            if not isinstance(content, dict):
                continue

            # Extract task information
            task_id_parts = task_key_in_item.split("_")
            # Handle new format with language suffix
            if len(task_id_parts) >= 4 and task_id_parts[-1] in ['en', 'zh']:
                task_id = task_id_parts[-2]
            else:
                task_id = task_id_parts[-1]
            
            task_name = f"{domain}_{task_id}"

            # Extract metadata
            doc_lengths_dict = content.get('doc_length', {})
            raw_doc_count = len(doc_lengths_dict) if isinstance(doc_lengths_dict, dict) else 0
            
            raw_average_length = content.get('average_length', 0.0)
            try:
                raw_average_length = float(raw_average_length)
            except (ValueError, TypeError):
                raw_average_length = 0.0

            answers_list = content.get('answers', [])
            raw_answers_count = len(answers_list) if isinstance(answers_list, list) else 0

            # Determine fixed levels
            doc_count_level_fixed = get_doc_count_level_fixed(raw_doc_count)
            average_length_level_fixed = get_average_length_level_fixed(raw_average_length)
            answers_count_level_fixed = get_answers_count_level_fixed(raw_answers_count)
            
            # Parse scores
            item_scores, added_metrics = parse_scores(content, task_name, all_metric_names)

            if not item_scores:
                continue
            
            # Aggregate scores
            domain_item_count += 1
            task_item_count[task_name] += 1
            
            for metric_name in added_metrics:
                score_value = item_scores.get(metric_name, 0.0)
                domain_score_sum[metric_name] += score_value
                task_score_sum[task_name][metric_name] += score_value

            # Accumulate for fixed-level domain reports
            domain_tally_doc_count_fixed[doc_count_level_fixed] += 1
            domain_tally_avg_len_fixed[average_length_level_fixed] += 1
            domain_tally_answers_fixed[answers_count_level_fixed] += 1
            
            for metric_name in added_metrics:
                score_value = item_scores.get(metric_name, 0.0)
                domain_score_sum_doc_count_fixed[doc_count_level_fixed][metric_name] += score_value
                domain_score_sum_avg_len_fixed[average_length_level_fixed][metric_name] += score_value
                domain_score_sum_answers_fixed[answers_count_level_fixed][metric_name] += score_value

            # Store detailed data for dynamic analysis
            detailed_item_data_for_model.append({
                "model": model_name,
                "domain": domain,
                "task_name": task_name,
                "task_key_in_item": task_key_in_item,
                "raw_doc_count": raw_doc_count,
                "raw_average_length": raw_average_length,
                "raw_answers_count": raw_answers_count,
                "scores": dict(item_scores),
                "metrics_present": list(added_metrics)
            })

        # Generate aggregated results
        if domain_item_count > 0:
            avg_row = {"model": model_name, "domain": domain}
            for metric in all_metric_names:
                score = domain_score_sum.get(metric, 0.0)
                scale_factor = 100 if (any(k in metric for k in ["f1", "precision", "recall"]) and not metric.startswith("one_piece_score")) or metric.startswith("if_csv_") else 1
                avg_row[metric] = round((score / domain_item_count) * scale_factor, 2)
            domain_results.append(avg_row)

        # Generate fixed-level results
        for level_name, count_val in domain_tally_doc_count_fixed.items():
            if count_val == 0:
                continue
            avg_row = {"model": model_name, "domain": domain, "doc_count_level": level_name}
            for metric in all_metric_names:
                total = domain_score_sum_doc_count_fixed[level_name].get(metric, 0.0)
                scale_factor = 100 if (any(k in metric for k in ["f1", "precision", "recall"]) and not metric.startswith("one_piece_score")) or metric.startswith("if_csv_") else 1
                avg_row[metric] = round((total / count_val) * scale_factor, 2)
            domain_doc_count_level_results.append(avg_row)

        for level_name, count_val in domain_tally_avg_len_fixed.items():
            if count_val == 0:
                continue
            avg_row = {"model": model_name, "domain": domain, "average_length_level": level_name}
            for metric in all_metric_names:
                total = domain_score_sum_avg_len_fixed[level_name].get(metric, 0.0)
                scale_factor = 100 if (any(k in metric for k in ["f1", "precision", "recall"]) and not metric.startswith("one_piece_score")) or metric.startswith("if_csv_") else 1
                avg_row[metric] = round((total / count_val) * scale_factor, 2)
            domain_average_length_level_results.append(avg_row)

        for level_name, count_val in domain_tally_answers_fixed.items():
            if count_val == 0:
                continue
            avg_row = {"model": model_name, "domain": domain, "answers_count_level": level_name}
            for metric in all_metric_names:
                total = domain_score_sum_answers_fixed[level_name].get(metric, 0.0)
                scale_factor = 100 if (any(k in metric for k in ["f1", "precision", "recall"]) and not metric.startswith("one_piece_score")) or metric.startswith("if_csv_") else 1
                avg_row[metric] = round((total / count_val) * scale_factor, 2)
            domain_answers_count_level_results.append(avg_row)
            
        # Generate task-level results
        for task_name_key, scores_sum_dict in task_score_sum.items():
            count_val = task_item_count[task_name_key]
            if count_val == 0:
                continue
            avg_row = {"model": model_name, "task_name": task_name_key}
            for metric in all_metric_names:
                score = scores_sum_dict.get(metric, 0.0)
                scale_factor = 100 if (any(k in metric for k in ["f1", "precision", "recall"]) and not metric.startswith("one_piece_score")) or metric.startswith("if_csv_") else 1
                avg_row[metric] = round((score / count_val) * scale_factor, 2)
            task_results.append(avg_row)
            
    return (
        domain_results, task_results, 
        domain_doc_count_level_results, domain_average_length_level_results, domain_answers_count_level_results,
        detailed_item_data_for_model
    )


def write_csv(output_path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    """Write data to CSV file with proper error handling."""
    if not rows:
        print(f"Warning: No data rows to write for {output_path}")
        if fieldnames:
            try:
                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
            except Exception as e:
                print(f"Error writing header for {output_path}: {e}")
        return

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Successfully wrote {len(rows)} rows to {output_path}")
    except IOError as e:
        print(f"Error writing to file {output_path}: {e}")
    except Exception as e:
        print(f"Unexpected error writing CSV {output_path}: {e}")


def analyze_dataset_distribution(all_detailed_items: List[Dict[str, Any]], output_dir: str) -> Dict[str, List[Any]]:
    """
    Analyze dataset distribution across different dimensions.
    
    Returns:
        Dictionary containing raw value collections for dynamic analysis
    """
    print("\n--- Dataset Distribution Analysis (Fixed Levels) ---")
    if not all_detailed_items:
        print("No detailed item data to analyze for distribution.")
        return {}

    # Use unique items to avoid double-counting across models
    seen_item_keys = set()
    unique_items_for_stats = []
    for item in all_detailed_items:
        item_unique_id = (item['domain'], item['task_key_in_item'])
        if item_unique_id not in seen_item_keys:
            seen_item_keys.add(item_unique_id)
            unique_items_for_stats.append(item)
    
    print(f"Analyzing distribution based on {len(unique_items_for_stats)} unique items.")

    # Calculate distributions
    dist_doc_count = defaultdict(int)
    dist_avg_len = defaultdict(int)
    dist_ans_count = defaultdict(int)

    raw_doc_counts_list = []
    raw_avg_lengths_list = []
    raw_answers_counts_list = []

    for item in unique_items_for_stats:
        dist_doc_count[get_doc_count_level_fixed(item['raw_doc_count'])] += 1
        dist_avg_len[get_average_length_level_fixed(item['raw_average_length'])] += 1
        dist_ans_count[get_answers_count_level_fixed(item['raw_answers_count'])] += 1
        
        raw_doc_counts_list.append(item['raw_doc_count'])
        raw_avg_lengths_list.append(item['raw_average_length'])
        raw_answers_counts_list.append(item['raw_answers_count'])

    # Print distribution summaries
    print("Document Count Distribution (Fixed):")
    for level, count in sorted(dist_doc_count.items()): 
        print(f"  {level}: {count}")
    
    print("Average Length Distribution (Fixed):")
    for level, count in sorted(dist_avg_len.items()): 
        print(f"  {level}: {count}")

    print("Answers Count Distribution (Fixed):")
    for level, count in sorted(dist_ans_count.items()): 
        print(f"  {level}: {count}")

    # Save distribution summary
    distribution_summary = {
        "doc_count_distribution_fixed": dict(sorted(dist_doc_count.items())),
        "average_length_distribution_fixed": dict(sorted(dist_avg_len.items())),
        "answers_count_distribution_fixed": dict(sorted(dist_ans_count.items())),
        "total_unique_items_analyzed": len(unique_items_for_stats)
    }
    
    dist_path = os.path.join(output_dir, "dataset_fixed_level_distribution_summary.json")
    try:
        with open(dist_path, "w", encoding="utf-8") as f:
            json.dump(distribution_summary, f, indent=2)
        print(f"Dataset distribution summary saved to {dist_path}")
    except Exception as e:
        print(f"Error saving distribution summary: {e}")

    return {
        "doc_counts": raw_doc_counts_list,
        "avg_lengths": raw_avg_lengths_list,
        "answers_counts": raw_answers_counts_list
    }


def aggregate_scores_by_dynamic_levels(
    all_detailed_items: List[Dict[str, Any]], 
    raw_value_collections: Dict[str, List[Any]],
    all_metrics: Set[str],
    output_dir: str,
    num_quantiles: int = 3
):
    """
    Aggregate scores by dynamic levels based on quantiles.
    
    Args:
        all_detailed_items: List of detailed item data
        raw_value_collections: Raw values for calculating quantiles
        all_metrics: Set of all metrics to include
        output_dir: Directory to save results
        num_quantiles: Number of quantiles to use for dynamic leveling
    """
    if not all_detailed_items:
        print("No detailed data for dynamic level aggregation.")
        return
    if num_quantiles < 2:
        print("Number of quantiles must be at least 2 for dynamic leveling.")
        return

    print(f"\n--- Aggregating Scores by Dynamic Levels ({num_quantiles}-quantiles) ---")
    
    percentiles_to_calc = [100 * i / num_quantiles for i in range(1, num_quantiles)]

    # Calculate dynamic thresholds for each dimension
    thresholds_doc_count = []
    labels_doc_count = []
    if raw_value_collections["doc_counts"]:
        dc_values = np.array(raw_value_collections["doc_counts"])
        if len(np.unique(dc_values)) == 1:
            thresholds_doc_count = [dc_values[0]] * (num_quantiles - 1)
        else:
            thresholds_doc_count = [np.percentile(dc_values, p) for p in percentiles_to_calc]
        
        labels_doc_count.append(f"<= {thresholds_doc_count[0]:.2f}")
        for i in range(len(thresholds_doc_count) - 1):
            labels_doc_count.append(f"{thresholds_doc_count[i]:.2f} < D <= {thresholds_doc_count[i+1]:.2f}")
        labels_doc_count.append(f"> {thresholds_doc_count[-1]:.2f}")
        print(f"Doc Count Dynamic Thresholds: {thresholds_doc_count}")

    # Similar calculations for average length and answer count
    thresholds_avg_len = []
    labels_avg_len = []
    if raw_value_collections["avg_lengths"]:
        al_values = np.array(raw_value_collections["avg_lengths"])
        if len(np.unique(al_values)) == 1:
            thresholds_avg_len = [al_values[0]] * (num_quantiles - 1)
        else:
            thresholds_avg_len = [np.percentile(al_values, p) for p in percentiles_to_calc]

        labels_avg_len.append(f"<= {thresholds_avg_len[0]:.2f}")
        for i in range(len(thresholds_avg_len) - 1):
            labels_avg_len.append(f"{thresholds_avg_len[i]:.2f} < AL <= {thresholds_avg_len[i+1]:.2f}")
        labels_avg_len.append(f"> {thresholds_avg_len[-1]:.2f}")
        print(f"Avg Length Dynamic Thresholds: {thresholds_avg_len}")

    thresholds_ans_count = []
    labels_ans_count = []
    if raw_value_collections["answers_counts"]:
        ac_values = np.array(raw_value_collections["answers_counts"])
        if len(np.unique(ac_values)) == 1:
            thresholds_ans_count = [ac_values[0]] * (num_quantiles - 1)
        elif len(np.unique(ac_values)) < num_quantiles:
            unique_sorted_ac = sorted(list(np.unique(ac_values)))
            thresholds_ans_count = unique_sorted_ac[:num_quantiles - 1]
            while len(thresholds_ans_count) < num_quantiles - 1:
                thresholds_ans_count.append(thresholds_ans_count[-1])
        else:
            thresholds_ans_count = [np.percentile(ac_values, p) for p in percentiles_to_calc]

        if thresholds_ans_count:
            labels_ans_count.append(f"<= {thresholds_ans_count[0]:.2f}")
            for i in range(len(thresholds_ans_count) - 1):
                labels_ans_count.append(f"{thresholds_ans_count[i]:.2f} < AC <= {thresholds_ans_count[i+1]:.2f}")
            labels_ans_count.append(f"> {thresholds_ans_count[-1]:.2f}")
            print(f"Ans Count Dynamic Thresholds: {thresholds_ans_count}")
        else:
            labels_ans_count = [f"Quantile_{i+1}" for i in range(num_quantiles)]

    # Aggregate scores by dynamic levels
    agg_data = {
        "doc_count_dynamic": defaultdict(lambda: {"scores": defaultdict(float), "count": 0}),
        "avg_len_dynamic": defaultdict(lambda: {"scores": defaultdict(float), "count": 0}),
        "ans_count_dynamic": defaultdict(lambda: {"scores": defaultdict(float), "count": 0}),
    }

    for item in all_detailed_items:
        model = item['model']
        domain = item['domain']
        scores = item['scores']

        # Aggregate by document count dynamic levels
        if thresholds_doc_count and labels_doc_count:
            level_dc = get_level_dynamic(item['raw_doc_count'], thresholds_doc_count, labels_doc_count)
            key_dc = (model, domain, level_dc)
            agg_data["doc_count_dynamic"][key_dc]["count"] += 1
            for metric, value in scores.items():
                agg_data["doc_count_dynamic"][key_dc]["scores"][metric] += value
        
        # Aggregate by average length dynamic levels
        if thresholds_avg_len and labels_avg_len:
            level_al = get_level_dynamic(item['raw_average_length'], thresholds_avg_len, labels_avg_len)
            key_al = (model, domain, level_al)
            agg_data["avg_len_dynamic"][key_al]["count"] += 1
            for metric, value in scores.items():
                agg_data["avg_len_dynamic"][key_al]["scores"][metric] += value

        # Aggregate by answer count dynamic levels
        if thresholds_ans_count and labels_ans_count:
            level_ac = get_level_dynamic(item['raw_answers_count'], thresholds_ans_count, labels_ans_count)
            key_ac = (model, domain, level_ac)
            agg_data["ans_count_dynamic"][key_ac]["count"] += 1
            for metric, value in scores.items():
                agg_data["ans_count_dynamic"][key_ac]["scores"][metric] += value
    
    # Generate CSV outputs for dynamic levels
    sorted_metrics_list = sorted(list(all_metrics))

    # Document count dynamic CSV
    rows_dc_dyn = []
    for (model, domain, level), data_dict in agg_data["doc_count_dynamic"].items():
        if data_dict["count"] > 0:
            row = {
                "model": model, 
                "domain": domain, 
                "doc_count_dynamic_level": level, 
                "item_count_in_level": data_dict["count"]
            }
            for metric in sorted_metrics_list:
                avg_score = data_dict["scores"].get(metric, 0.0) / data_dict["count"]
                scale_factor = 100 if any(k in metric for k in ["f1", "precision", "recall"]) and not metric.startswith("one_piece_score") else 1
                row[metric] = round(avg_score * scale_factor, 2)
            rows_dc_dyn.append(row)
    
    fields_dc_dyn = ["model", "domain", "doc_count_dynamic_level", "item_count_in_level"] + sorted_metrics_list
    write_csv(os.path.join(output_dir, "all_models_domain_doc_count_dynamic_level.csv"), rows_dc_dyn, fields_dc_dyn)
    
    # Average length dynamic CSV
    rows_al_dyn = []
    for (model, domain, level), data_dict in agg_data["avg_len_dynamic"].items():
        if data_dict["count"] > 0:
            row = {
                "model": model, 
                "domain": domain, 
                "average_length_dynamic_level": level, 
                "item_count_in_level": data_dict["count"]
            }
            for metric in sorted_metrics_list:
                avg_score = data_dict["scores"].get(metric, 0.0) / data_dict["count"]
                scale_factor = 100 if any(k in metric for k in ["f1", "precision", "recall"]) and not metric.startswith("one_piece_score") else 1
                row[metric] = round(avg_score * scale_factor, 2)
            rows_al_dyn.append(row)
    
    fields_al_dyn = ["model", "domain", "average_length_dynamic_level", "item_count_in_level"] + sorted_metrics_list
    write_csv(os.path.join(output_dir, "all_models_domain_average_length_dynamic_level.csv"), rows_al_dyn, fields_al_dyn)

    # Answer count dynamic CSV
    rows_ac_dyn = []
    for (model, domain, level), data_dict in agg_data["ans_count_dynamic"].items():
        if data_dict["count"] > 0:
            row = {
                "model": model, 
                "domain": domain, 
                "answers_count_dynamic_level": level, 
                "item_count_in_level": data_dict["count"]
            }
            for metric in sorted_metrics_list:
                avg_score = data_dict["scores"].get(metric, 0.0) / data_dict["count"]
                scale_factor = 100 if any(k in metric for k in ["f1", "precision", "recall"]) and not metric.startswith("one_piece_score") else 1
                row[metric] = round(avg_score * scale_factor, 2)
            rows_ac_dyn.append(row)
    
    fields_ac_dyn = ["model", "domain", "answers_count_dynamic_level", "item_count_in_level"] + sorted_metrics_list
    write_csv(os.path.join(output_dir, "all_models_domain_answers_count_dynamic_level.csv"), rows_ac_dyn, fields_ac_dyn)

    print("Dynamic level aggregation CSVs written successfully.")


def evaluate_all_models(model_names: List[str], prompt_setting: str, domains: List[str], others: str):
    """
    Main function to evaluate all models and generate comprehensive summary reports.
    
    Args:
        model_names: List of model names to evaluate
        prompt_setting: Prompt setting identifier
        domains: List of domains to process
        others: Additional run-specific identifiers
    """
    print(f"Starting evaluation for {len(model_names)} models across {len(domains)} domains")
    
    # Initialize result containers
    all_domain_results = []
    all_task_results = []
    all_doc_count_level_results_fixed = []
    all_average_length_level_results_fixed = []
    all_answers_count_level_results_fixed = []
    all_detailed_item_data = []
    all_metrics = set()

    # Set up output directory
    base_output_dir = os.path.join(PATHS["root_key"], prompt_setting, others, "evaluation_summary")
    summary_csv_output_dir = os.path.join(base_output_dir, "_meta_evaluation_summary")
    os.makedirs(summary_csv_output_dir, exist_ok=True)
    print(f"Summary reports will be saved to: {summary_csv_output_dir}")

    # Process each model
    for model_name in tqdm(model_names, desc="Processing models", unit="model"):
        print(f"\nProcessing model: {model_name}")
        
        domain_rows, task_rows, doc_count_rows, avg_len_rows, answers_count_rows, detailed_items = process_model(
            model_name, prompt_setting, others, domains, all_metrics
        )
        
        # Aggregate results
        all_domain_results.extend(domain_rows)
        all_task_results.extend(task_rows)
        all_doc_count_level_results_fixed.extend(doc_count_rows)
        all_average_length_level_results_fixed.extend(avg_len_rows)
        all_answers_count_level_results_fixed.extend(answers_count_rows)
        all_detailed_item_data.extend(detailed_items)

    # Analyze dataset distribution
    print("\nAnalyzing dataset distribution...")
    raw_value_collections = analyze_dataset_distribution(all_detailed_item_data, summary_csv_output_dir)

    # Generate dynamic level analysis
    if raw_value_collections and all_detailed_item_data:
        print("Generating dynamic level analysis...")
        aggregate_scores_by_dynamic_levels(
            all_detailed_item_data, 
            raw_value_collections, 
            all_metrics, 
            summary_csv_output_dir,
            num_quantiles=3
        )
    else:
        print("Skipping dynamic level aggregation due to insufficient data.")

    # Prepare for CSV output
    sorted_metrics_list = sorted(list(all_metrics))
    print(f"Found {len(sorted_metrics_list)} unique metrics: {sorted_metrics_list[:5]}...")

    # Define field names for different report types
    domain_fields = ["model", "domain"] + sorted_metrics_list
    task_fields = ["model", "task_name"] + sorted_metrics_list
    doc_count_fields_fixed = ["model", "domain", "doc_count_level"] + sorted_metrics_list
    avg_len_fields_fixed = ["model", "domain", "average_length_level"] + sorted_metrics_list
    answers_count_fields_fixed = ["model", "domain", "answers_count_level"] + sorted_metrics_list

    # Write main summary CSV files
    print("Writing summary CSV files...")
    write_csv(
        os.path.join(summary_csv_output_dir, "all_models_domain_average_scores.csv"), 
        all_domain_results, 
        domain_fields
    )
    write_csv(
        os.path.join(summary_csv_output_dir, "all_models_task_average_scores.csv"), 
        all_task_results, 
        task_fields
    )
    
    # Write fixed-level analysis CSV files
    write_csv(
        os.path.join(summary_csv_output_dir, "all_models_domain_doc_count_fixed_level.csv"), 
        all_doc_count_level_results_fixed, 
        doc_count_fields_fixed
    )
    write_csv(
        os.path.join(summary_csv_output_dir, "all_models_domain_average_length_fixed_level.csv"), 
        all_average_length_level_results_fixed, 
        avg_len_fields_fixed
    )
    write_csv(
        os.path.join(summary_csv_output_dir, "all_models_domain_answers_count_fixed_level.csv"), 
        all_answers_count_level_results_fixed, 
        answers_count_fields_fixed
    )
    
    # Generate overall model performance summary
    print("Generating overall model performance summary...")
    model_level_scores = defaultdict(lambda: defaultdict(float))
    model_task_counts = defaultdict(int)
    
    for row in all_task_results:
        model = row["model"]
        model_task_counts[model] += 1 
        for metric in sorted_metrics_list:
            model_level_scores[model][metric] += row.get(metric, 0.0)
            
    all_model_overall_results = []
    for model, scores_sum_dict in model_level_scores.items():
        num_tasks_for_model = model_task_counts[model]
        if num_tasks_for_model == 0:
            continue

        avg_row = {"model": model}
        for metric in sorted_metrics_list:
            avg_row[metric] = round(scores_sum_dict.get(metric, 0.0) / num_tasks_for_model, 2)
        all_model_overall_results.append(avg_row)
        
    model_fields = ["model"] + sorted_metrics_list
    write_csv(
        os.path.join(summary_csv_output_dir, "all_models_overall_average_scores.csv"), 
        all_model_overall_results, 
        model_fields
    )
    
    print(f"\n✅ Evaluation summary complete!")
    print(f"📊 Processed {len(all_detailed_item_data)} total items across {len(model_names)} models")
    print(f"📁 All summary reports saved to: {summary_csv_output_dir}")
    print(f"📈 Generated {len(sorted_metrics_list)} different metrics")


def main():
    """Main entry point for the evaluation summary script."""
    parser = argparse.ArgumentParser(
        description="Summarize model evaluation scores with comprehensive analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python summary_evaluation.py --models model1 model2 --domains academic legal financial --prompt_setting all
  python summary_evaluation.py --models gpt-4 claude-3 --domains academic --others rag --prompt_setting no_cot
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
        '--prompt_setting', 
        default="all", 
        help='Prompt setting identifier (e.g., all, no_cot, no_example)'
    )
    parser.add_argument(
        '--others', 
        default="", 
        help='Additional run-specific identifiers (e.g., rag, fine_tuned)'
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting Model Evaluation Summary")
    print(f"Models: {args.models}")
    print(f"Domains: {args.domains}")
    print(f"Prompt Setting: {args.prompt_setting}")
    print(f"Others: {args.others}")
    print("-" * 50)
    
    try:
        evaluate_all_models(
            model_names=args.models,
            prompt_setting=args.prompt_setting,
            domains=args.domains,
            others=args.others
        )
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())