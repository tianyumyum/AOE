import os
import csv
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
import sys
import time
import re
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config_loader import PATHS, prompt_templates_eval, task_config, get_predict_result_filepath, get_evaluation_filepath, get_intermediate_results_filepath
from src.llm.call_llm import call_llm_eval, call_llm_eval_fractional, call_llm_generate
from src.utils import (
    load_data,
    read_df_from_llm_output,
    save_dataframe_to_csv,
    save_gold,
    is_valid_csv,
    save_predict_txt,
    try_load_csv,
    save_data_jsonl,
    save_data_json_add,
    save_output_txt,
    save_data_jsonl_add,
    save_output_json,
    get_domain_folder_task,
    extract_law_names,
    extract_long_json_list_from_str
)

TOP_K = 50
CHUNK_SIZE = 512

think_tag_mapping = {
    "QwQ-32B": "think",
    "Qwen2.5-72B-Instruct-GPTQ-Int4": "think",
    "Default": None,
}

try:
    from thefuzz import process, fuzz
    THEFUZZ_AVAILABLE = True
except ImportError:
    THEFUZZ_AVAILABLE = False
    print("Warning: thefuzz library not found. Fuzzy matching will not be available.")

from tqdm import tqdm


def read_csv_mapped_with_llm(predict_txt, gold_header_string, current_base_url, current_api_key):
    instruction = prompt_templates_eval["Details"]["mapping_predict_tables"]["Instruction"]
    output = prompt_templates_eval["Details"]["mapping_predict_tables"]["Output"].format(
        predict_txt=predict_txt, gold_header_string=gold_header_string
    )
    example = prompt_templates_eval["Details"]["mapping_predict_tables"]['Examples']
    input_prompt = f"{instruction}\n\n{example}\n\n{output}"
    mapped_csv_txt = call_llm_eval(input_prompt, max_tokens=3000, base_url=current_base_url, api_key=current_api_key)
    print(f"Mapping base_url: {current_base_url}")
    mapped_df, error_info = read_df_from_llm_output(mapped_csv_txt)
    if isinstance(mapped_df, pd.DataFrame):
        return mapped_csv_txt, mapped_df, error_info
    else:
        return mapped_csv_txt, None, error_info


def clean_numeric_value(value):
    if pd.isna(value): 
        return np.nan
    if isinstance(value, (int, float)): 
        return float(value)
    if isinstance(value, str):
        value = value.replace(',', '')
        if '%' in value:
            try: 
                return float(value.replace('%', '')) / 100.0
            except ValueError: 
                pass
        try: 
            return float(value)
        except ValueError:
            if value == '-': 
                return np.nan
            if re.search(r'[^-+\d\.eE]', value): 
                return value
            elif value.count('.') > 1 or value.count('-') > 1 or value.count('+') > 1:
                if 'e' not in value.lower() and 'E' not in value.lower(): 
                    return value
                try: 
                    return float(value)
                except ValueError: 
                    return value
            else: 
                return np.nan
    return value


def clean_key_value(value, remove_suffix: Optional[str] = None):
    if pd.isna(value): 
        return ""
    value = str(value).strip()
    if remove_suffix and value.endswith(remove_suffix):
        value = value[:-len(remove_suffix)]
    return value


def _build_key_representation(row, key_columns) -> Any:
    if len(key_columns) == 1: 
        return str(row[key_columns[0]])
    else: 
        return tuple(str(row[col]) for col in key_columns)


def convert_numpy_types(obj):
    if isinstance(obj, np.integer): 
        return int(obj)
    elif isinstance(obj, np.floating): 
        return None if np.isnan(obj) else float(obj)
    elif isinstance(obj, np.bool_): 
        return bool(obj)
    elif isinstance(obj, np.ndarray): 
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict): 
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list): 
        return [convert_numpy_types(item) for item in obj]
    else: 
        return obj


def _match_rows_llm(gold_df_clean: pd.DataFrame, mapped_df_clean: pd.DataFrame, key_columns: List[str], fuzzy_row_match_threshold: int = 50, current_base_url: str = '', current_api_key: str = '') -> Tuple[List[Tuple[Any, Any]], Dict[str, Any]]:
    matched_pairs = []
    matched_gold_indices = set()
    stats = {"calls": 0, "failures": 0, "by_exact": 0, "by_fuzzy": 0, "by_llm": 0}
    gold_key_map = {}
    gold_key_list_for_scorer = []
    unique_string_reprs = set()
    
    for idx, row in gold_df_clean.iterrows():
        key_repr = _build_key_representation(row, key_columns)
        if key_repr not in gold_key_map:
            gold_key_map[key_repr] = idx
            key_repr_str = str(key_repr)
            if key_repr_str not in unique_string_reprs:
                gold_key_list_for_scorer.append(key_repr_str)
                unique_string_reprs.add(key_repr_str)
    
    gold_key_list_str = "\n".join([f"- {k}" for k in gold_key_map.keys()])

    for mapped_idx, mapped_row in mapped_df_clean.iterrows():
        mapped_key_repr = _build_key_representation(mapped_row, key_columns)
        matched_gold_idx = None
        potential_gold_idx_exact = gold_key_map.get(mapped_key_repr)
        
        if potential_gold_idx_exact is not None and potential_gold_idx_exact not in matched_gold_indices:
            matched_gold_idx = potential_gold_idx_exact
            stats["by_exact"] += 1
        
        if matched_gold_idx is None and fuzzy_row_match_threshold > 0 and THEFUZZ_AVAILABLE and gold_key_list_for_scorer:
            try:
                mapped_key_str = str(mapped_key_repr)
                best_match_choice_str, score = process.extractOne(mapped_key_str, gold_key_list_for_scorer, scorer=fuzz.token_sort_ratio)
                if score >= fuzzy_row_match_threshold:
                    potential_g_indices_fuzzy = [g_idx for k_repr_g, g_idx in gold_key_map.items() if str(k_repr_g) == best_match_choice_str]
                    for p_idx in potential_g_indices_fuzzy:
                        if p_idx not in matched_gold_indices:
                            matched_gold_idx = p_idx
                            stats["by_fuzzy"] += 1
                            break
            except Exception as e: 
                if THEFUZZ_AVAILABLE: 
                    print(f"Warning: Fuzzy match error for mapped_idx {mapped_idx}: {e}")

        if matched_gold_idx is None:  # LLM attempt
            stats["calls"] += 1
            instruction = prompt_templates_eval["Row_evaluation"]["Instruction"]
            output_format = prompt_templates_eval["Row_evaluation"]["Output"].format(
                key_col_names=", ".join(key_columns), 
                mapped_key_value=str(mapped_key_repr), 
                gold_key_list_str=gold_key_list_str
            )
            example = prompt_templates_eval["Row_evaluation"]['Examples']
            query = f"{instruction}\n\n{example}\n\n{output_format}"
            llm_response_raw = call_llm_eval(query, max_tokens=200, base_url=current_base_url, api_key=current_api_key)
            
            match = re.search(r"<output>(.*?)</output>", llm_response_raw, re.IGNORECASE | re.DOTALL)
            llm_response = match.group(1).strip() if match else None
            
            if llm_response and llm_response.strip().lower() != 'none':
                returned_key_str = llm_response.strip()
                for key_repr_g, gold_idx_g in gold_key_map.items():
                    if str(key_repr_g) == returned_key_str and gold_idx_g not in matched_gold_indices:
                        matched_gold_idx = gold_idx_g
                        stats["by_llm"] += 1
                        break
            elif llm_response is None: 
                stats["failures"] += 1
        
        if matched_gold_idx is not None and matched_gold_idx not in matched_gold_indices:
            matched_pairs.append((mapped_idx, matched_gold_idx))
            matched_gold_indices.add(matched_gold_idx)
    
    return matched_pairs, stats


def _compare_matched_cells(gold_df_clean: pd.DataFrame, mapped_df_clean: pd.DataFrame, matched_pairs: List[Tuple[Any, Any]], target_value_columns: List[str], numeric_columns: List[str], numeric_tolerance: float, non_numeric_match_method: str, fuzzy_threshold: int, llm_failure_score: float, current_base_url: str, current_api_key: str) -> Tuple[float, int, Dict[str, Any]]:
    tp_score = 0.0
    total_compared_cells = 0
    stats = {"calls": 0, "failures": 0}
    relevant_numeric_cols = [col for col in numeric_columns if col in target_value_columns]

    for mapped_idx, gold_idx in matched_pairs:
        for col in target_value_columns:
            total_compared_cells += 1
            gold_val = gold_df_clean.loc[gold_idx, col]
            mapped_val = mapped_df_clean.loc[mapped_idx, col]
            cell_score = 0.0
            is_numeric_col = col in relevant_numeric_cols
            
            if is_numeric_col:
                if pd.isna(gold_val) and pd.isna(mapped_val): 
                    cell_score = 1.0
                elif not pd.isna(gold_val) and not pd.isna(mapped_val):
                    if np.isclose(gold_val, mapped_val, atol=numeric_tolerance, equal_nan=False): 
                        cell_score = 1.0
            else:
                gold_str, mapped_str = str(gold_val), str(mapped_val)
                if gold_str == '' and mapped_str == '': 
                    cell_score = 1.0
                elif gold_str == '' or mapped_str == '': 
                    cell_score = 0.0
                elif non_numeric_match_method == 'exact':
                    if gold_str == mapped_str: 
                        cell_score = 1.0
                elif non_numeric_match_method == 'fuzzy' and THEFUZZ_AVAILABLE:
                    similarity = fuzz.token_sort_ratio(gold_str, mapped_str)
                    if similarity >= fuzzy_threshold: 
                        cell_score = 1.0
                elif non_numeric_match_method == 'llm':
                    if gold_str == mapped_str: 
                        cell_score = 1.0
                    else:
                        instruction = prompt_templates_eval["Cell_evaluation"]["Instruction"]
                        output_format = prompt_templates_eval["Cell_evaluation"]["Output"].format(
                            column_name=col, gold_text=gold_str, predicted_text=mapped_str
                        )
                        example = prompt_templates_eval["Cell_evaluation"]['Examples']
                        query = f"{instruction}\n\n{example}\n\n{output_format}"
                        stats["calls"] += 1
                        parsed_score = call_llm_eval_fractional(query, base_url=current_base_url, api_key=current_api_key)
                        
                        if parsed_score is not None: 
                            cell_score = parsed_score
                        else: 
                            stats["failures"] += 1
                            cell_score = llm_failure_score
            tp_score += cell_score
    
    return tp_score, total_compared_cells, stats


def compare_table_data(gold_df: pd.DataFrame, mapped_df: pd.DataFrame, key_columns: List[str], target_value_columns: List[str], numeric_columns: Optional[List[str]] = None, numeric_tolerance: float = 1e-6, row_match_method: str = 'llm_assisted', key_suffix_to_remove: Optional[str] = None, non_numeric_match_method: str = 'exact', fuzzy_threshold: int = 90, llm_failure_score: float = 0.0, current_base_url: str = '', current_api_key: str = '') -> Dict[str, Any]:
    if gold_df is None: 
        return convert_numpy_types({"error": "Gold DataFrame is None"})
    if not key_columns: 
        raise ValueError("Key columns must be specified.")
    
    original_target_value_columns = list(target_value_columns)
    
    # Ensure target_value_columns are present in gold_df
    valid_target_value_columns = [col for col in target_value_columns if col in gold_df.columns]
    if len(valid_target_value_columns) != len(target_value_columns):
        missing_in_gold = set(target_value_columns) - set(valid_target_value_columns)
        print(f"Warning: Following target_value_columns not in gold_df and will be ignored: {missing_in_gold}")
    target_value_columns = valid_target_value_columns
    
    if not target_value_columns: 
        print("Warning: No valid target_value_columns remaining after checking gold_df. Cell metrics will be zero.")
        return convert_numpy_types({
            "error": "No valid target value columns to compare.",
            "num_gold_rows": len(gold_df) if gold_df is not None else 0,
            "num_original_mapped_rows": len(mapped_df) if mapped_df is not None else 0,
        })

    if not set(key_columns).issubset(gold_df.columns):
        raise ValueError(f"Gold DF missing key columns: {set(key_columns) - set(gold_df.columns)}")

    gold_df_processed = gold_df.copy()
    num_gold_rows = len(gold_df_processed)
    selected_gold_cols = set(key_columns) | set(target_value_columns)
    gold_df_processed = gold_df_processed[[col for col in selected_gold_cols if col in gold_df_processed.columns]]

    original_num_mapped_rows = 0
    mapped_df_processed = pd.DataFrame()
    if mapped_df is not None and not mapped_df.empty:
        mapped_df_processed = mapped_df.copy()
        original_num_mapped_rows = len(mapped_df_processed)
        for col in target_value_columns:
            if col not in mapped_df_processed.columns:
                mapped_df_processed[col] = "" 
        if not set(key_columns).issubset(mapped_df_processed.columns):
            missing_keys_mapped = set(key_columns) - set(mapped_df_processed.columns)
            print(f"Warning: Mapped DF missing key columns: {missing_keys_mapped}. Adding them as empty strings for row matching.")
            for k_col in missing_keys_mapped: 
                mapped_df_processed[k_col] = ""
        
        selected_mapped_cols = set(key_columns) | set(target_value_columns)
        mapped_df_processed = mapped_df_processed[[col for col in selected_mapped_cols if col in mapped_df_processed.columns]]

    numeric_cols_to_process = numeric_columns if numeric_columns else []
    relevant_numeric_cols = [col for col in numeric_cols_to_process if col in target_value_columns]

    for col in key_columns:
        if col in gold_df_processed:
            gold_df_processed[col] = gold_df_processed[col].apply(lambda x: clean_key_value(x)).astype(str)
        if col in mapped_df_processed:
            mapped_df_processed[col] = mapped_df_processed[col].apply(lambda x: clean_key_value(x, remove_suffix=key_suffix_to_remove)).astype(str)

    for col in target_value_columns:
        is_numeric = col in relevant_numeric_cols
        if col in gold_df_processed:
            try: 
                gold_df_processed[col] = pd.to_numeric(gold_df_processed[col].apply(clean_numeric_value), errors='coerce') if is_numeric else gold_df_processed[col].fillna('').astype(str).str.strip()
            except Exception as e: 
                print(f"Error cleaning gold col {col}: {e}")
        if col in mapped_df_processed:
            try: 
                mapped_df_processed[col] = pd.to_numeric(mapped_df_processed[col].apply(clean_numeric_value), errors='coerce') if is_numeric else mapped_df_processed[col].fillna('').astype(str).str.strip()
            except Exception as e: 
                print(f"Error cleaning mapped col {col}: {e}")
    
    matched_pairs = []
    row_match_stats = {"calls": 0, "failures": 0, "by_exact": 0, "by_fuzzy": 0, "by_llm": 0}

    if mapped_df_processed.empty:
        print("Warning: Mapped DataFrame is empty. No row matching or cell comparison will be performed.")
    elif row_match_method == 'llm_assisted':
        matched_pairs, row_match_stats = _match_rows_llm(gold_df_processed, mapped_df_processed, key_columns, fuzzy_threshold, current_base_url, current_api_key)
    elif row_match_method == 'exact':
        try:
            if not all(k in mapped_df_processed.columns for k in key_columns):
                print("Warning: Not all key columns present in mapped_df for exact merge. Results may be inaccurate.")
                merged = pd.DataFrame(columns=['index_x', 'index_y'] + key_columns)
            else:
                merged = pd.merge(gold_df_processed[key_columns].reset_index(), mapped_df_processed[key_columns].reset_index(), on=key_columns, how='inner')
            matched_pairs = list(zip(merged['index_y'], merged['index_x']))
        except Exception as e: 
            print(f"Exact merge failed: {e}")
            matched_pairs = []
        row_match_stats["by_exact"] = len(matched_pairs)
    else: 
        raise ValueError(f"Unsupported row_match_method: {row_match_method}")

    num_matched_rows = len(matched_pairs)
    matched_gold_indices = {pair[1] for pair in matched_pairs}
    num_missing_rows = num_gold_rows - len(matched_gold_indices)
    num_extra_rows = original_num_mapped_rows - num_matched_rows

    row_precision = num_matched_rows / original_num_mapped_rows if original_num_mapped_rows > 0 else 0.0
    row_recall = num_matched_rows / num_gold_rows if num_gold_rows > 0 else 0.0
    row_f1 = 2*(row_precision * row_recall) / (row_precision + row_recall) if (row_precision + row_recall) > 0 else 0.0

    tp_score_sum, total_compared_cells, cell_eval_stats = 0.0, 0, {"calls": 0, "failures": 0}
    if num_matched_rows > 0 and not mapped_df_processed.empty:
        tp_score_sum, total_compared_cells, cell_eval_stats = _compare_matched_cells(gold_df_processed, mapped_df_processed, matched_pairs, target_value_columns, numeric_cols_to_process, numeric_tolerance, non_numeric_match_method, fuzzy_threshold, llm_failure_score, current_base_url, current_api_key)
    
    num_initial_target_cols = len(original_target_value_columns)
    num_valid_target_cols_in_gold = len(target_value_columns)

    total_gold_target_cells = num_gold_rows * num_valid_target_cols_in_gold
    total_mapped_target_cells = original_num_mapped_rows * num_valid_target_cols_in_gold

    cell_recall = tp_score_sum / total_gold_target_cells if total_gold_target_cells > 0 else 0.0
    cell_precision = tp_score_sum / total_mapped_target_cells if total_mapped_target_cells > 0 else 0.0
    
    cell_f1 = 2 * (cell_precision * cell_recall) / (cell_precision + cell_recall) if (cell_precision + cell_recall) > 0 else 0.0
    average_cell_score_in_matched = tp_score_sum / total_compared_cells if total_compared_cells > 0 else 0.0
    
    scores = {
        "num_initial_target_cols": num_initial_target_cols,
        "num_valid_target_cols_in_gold": num_valid_target_cols_in_gold,
        "column_recall": num_valid_target_cols_in_gold / num_initial_target_cols if num_initial_target_cols > 0 else 0.0,
        "num_gold_rows": num_gold_rows, 
        "num_original_mapped_rows": original_num_mapped_rows,
        "num_matched_rows": num_matched_rows, 
        "num_missing_rows_gold_only": num_missing_rows,
        "num_extra_rows_mapped_only": num_extra_rows,
        "row_precision": row_precision, 
        "row_recall": row_recall, 
        "row_f1": row_f1,
        "tp_score_sum": tp_score_sum, 
        "total_gold_target_cells": total_gold_target_cells,
        "total_mapped_target_cells_corresp_gold_schema": total_mapped_target_cells,
        "cell_precision": cell_precision, 
        "cell_recall": cell_recall, 
        "cell_f1": cell_f1,
        "total_compared_cells_in_matched": total_compared_cells,
        "average_cell_score_in_matched": average_cell_score_in_matched,
        "key_columns_used": key_columns, 
        "row_match_method_used": row_match_method,
        "row_match_stats": row_match_stats, 
        "cell_comparison_method": non_numeric_match_method,
        "cell_eval_stats": cell_eval_stats,
    }
    if non_numeric_match_method == 'fuzzy': 
        scores["fuzzy_cell_threshold"] = fuzzy_threshold
    return convert_numpy_types(scores)


def is_valid_score_dict(d):
    return isinstance(d, dict) and "error" not in d and all(v is not None for v in d.values())


def get_overall_evaluation_score(query, gold_txt, predict_txt, current_base_url, current_api_key):
    instruction = prompt_templates_eval["Overall_eval"]["Instruction"]
    output_format = prompt_templates_eval["Overall_eval"]["Output"].format(
        query=query, gold_csv_txt=gold_txt, predict_csv_txt=predict_txt
    )
    example = prompt_templates_eval["Overall_eval"]['Examples'] 
    full_query = f"{instruction}\n\n{example}\n\n{output_format}"
    llm_response = call_llm_eval(full_query, base_url=current_base_url, api_key=current_api_key, max_tokens=500)
    
    result_json = {
        "score1": None, "score2": None,
        "score3": None, "score4": None
    }
    
    if llm_response:
        try:
            score1_match = re.search(r"<score1>(\d+)</score1>", llm_response)
            score2_match = re.search(r"<score2>(\d+)</score2>", llm_response)
            score3_match = re.search(r"<score3>(\d+)</score3>", llm_response)
            score4_match = re.search(r"<score4>(\d+)</score4>", llm_response)
            result_json["score1"] = int(score1_match.group(1)) if score1_match else None
            result_json["score2"] = int(score2_match.group(1)) if score2_match else None
            result_json["score3"] = int(score3_match.group(1)) if score3_match else None
            result_json["score4"] = int(score4_match.group(1)) if score4_match else None
        except Exception as e:
            print(f"LLM Score RE parsing failed: {e}. Response: {llm_response}")
            result_json["error"] = f"RE parsing failed: {e}"
    else:   
        print(f"Failed to parse overall score from empty/None LLM response.")  
        result_json["error"] = "Empty/None LLM response for overall score."
    
    return result_json


def process_item_concurrently(
    item_tuple: Tuple[str, Dict],
    current_model_name: str,
    current_prompt_setting: str,
    current_others: str,
    current_think_tag: Optional[str],
    global_task_config: Dict,
    current_api_key: str,
    current_base_url: str,
) -> Tuple[str, Dict, Optional[List[Dict]]]:
    """
    Processes a single item, evaluating multiple prediction sources if available.
    Returns: (item_key, updated_item_value, list_of_bad_case_infos or None)
    """
    item_key, item_value_original = item_tuple
    item_value = item_value_original.copy()

    overall_scores_for_item = item_value.get("scores", {})
    bad_cases_for_item = []

    # Determine which answer types to process
    answer_source_keys = []
    
    if "rag" in current_others:
        for key, value in item_value.items():
            if "option2" in key:
                answer_source_keys.append(key)
    else:
        for key, value in item_value.items():
            if "predict_answer" in key:
                answer_source_keys.append(key)

    # Gold Standard (common for all answer types)
    gold_answer_csv_path = get_intermediate_results_filepath(current_prompt_setting, current_others, current_model_name, item_key, "gold_answer.csv")
    gold_answer_txt_path = get_intermediate_results_filepath(current_prompt_setting, current_others, current_model_name, item_key, "gold_answer.txt")
    _, _, df_gold = save_gold(item_value, gold_answer_txt_path, gold_answer_csv_path)
    query = item_value.get("query", "")

    # Loop through each answer type
    for answer_type_key in answer_source_keys:
        print(f"[{item_key}] Processing answer type: {answer_type_key}")
        current_type_scores = overall_scores_for_item.get(answer_type_key, {})
        local_bad_case_info = None

        # Generate paths specific to this answer_type
        predict_answer_txt_path_typed = get_intermediate_results_filepath(current_prompt_setting, current_others, current_model_name, item_key, f"{answer_type_key}_predict_answer.txt")
        predict_answer_csv_path_typed = get_intermediate_results_filepath(current_prompt_setting, current_others, current_model_name, item_key, f"{answer_type_key}_predict_direct.csv")
        predict_answer_raw_txt_path_typed = get_intermediate_results_filepath(current_prompt_setting, current_others, current_model_name, item_key, f"{answer_type_key}_predict_raw_answer.txt")
        mapped_answer_csv_path_typed = get_intermediate_results_filepath(current_prompt_setting, current_others, current_model_name, item_key, f"{answer_type_key}_mapped_answer.csv")
        scores_json_path_typed = get_intermediate_results_filepath(current_prompt_setting, current_others, current_model_name, item_key, f"{answer_type_key}_scores.json")

        # Save predicted for current answer_type
        predict_txt_content, if_csv_parsed = save_predict_txt(
            item_value,
            answer_type_key,
            current_think_tag,
            predict_answer_txt_path_typed,
            predict_answer_raw_txt_path_typed,
            predict_answer_csv_path_typed
        )
        current_type_scores["if_csv_parsed"] = current_type_scores.get("if_csv_parsed", {})
        
        if if_csv_parsed == "csv_yes":
            current_type_scores["if_csv_parsed"]["direct"] = 1
        elif if_csv_parsed == "csv_no": 
            current_type_scores["if_csv_parsed"]["direct"] = 0
            
        should_calculate_one_piece = True
        should_calculate_cell = True

        if not predict_txt_content:
            should_calculate_one_piece = False
            should_calculate_cell = False
            current_type_scores["one_piece"] = {"error": "Empty prediction text."}
            current_type_scores["cell_scores"] = {"error": "Empty prediction text, cannot perform cell evaluation."}
        else:
            # Check existing scores from file for this answer_type
            if os.path.exists(scores_json_path_typed):
                try:
                    with open(scores_json_path_typed, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    if isinstance(existing_data, dict):
                        if is_valid_score_dict(existing_data.get("one_piece")):
                            should_calculate_one_piece = False
                            current_type_scores["one_piece"] = existing_data["one_piece"]
                        
                        cell_scores = existing_data.get("cell_scores")
                        if isinstance(cell_scores, dict) and is_valid_score_dict(cell_scores):
                            if any(isinstance(v, (int, float)) for v in cell_scores.values()):
                                should_calculate_cell = False
                                current_type_scores["cell_scores"] = existing_data["cell_scores"]
                except (json.JSONDecodeError, IOError, Exception) as e:
                    print(f"Warning: Error reading existing scores for {item_key}/{answer_type_key}: {e}. Will recalculate.")

        # One-piece overall score for current answer_type
        if should_calculate_one_piece:
            print(f"[{item_key}/{answer_type_key}] Calculating one-piece score...")
            gold_txt_for_eval = df_gold.to_csv(index=False, sep=",") if not df_gold.empty else ""

            if gold_txt_for_eval and predict_txt_content:
                scores_one_piece = get_overall_evaluation_score(query, gold_txt_for_eval, predict_txt_content, current_base_url, current_api_key)
                current_type_scores["one_piece"] = scores_one_piece
                print(f"One-piece scores: {scores_one_piece}")
            else:
                err_msg = "Missing "
                if not gold_txt_for_eval: 
                    err_msg += "gold "
                if not predict_txt_content: 
                    err_msg += "prediction "
                err_msg += "content for one-piece eval."
                current_type_scores["one_piece"] = {"error": err_msg}
                print(f"[{item_key}/{answer_type_key}] Skipping one-piece: {err_msg}")
            
            save_output_json({"one_piece": current_type_scores.get("one_piece")}, scores_json_path_typed)

        # Cell-level evaluation for current answer_type
        if should_calculate_cell:
            print(f"[{item_key}/{answer_type_key}] Calculating cell-level scores...")
            if df_gold.empty:
                current_type_scores["cell_scores"] = {"error": "Gold answer CSV not found or empty, cannot get header."}
                print(f"[{item_key}/{answer_type_key}] Skipping cell-eval: gold CSV missing or empty.")
            else:
                gold_header_string = ','.join(df_gold.columns.tolist())
                MAX_RETRIES = 3
                mapped_df = try_load_csv(mapped_answer_csv_path_typed)

                retry_attempt = 0
                mapped_csv_txt_for_bad_case = "N/A"
                error_info_for_bad_case = "Mapping not attempted or predict_txt was None."

                if predict_txt_content:
                    while (mapped_df is None or mapped_df.empty) and retry_attempt < MAX_RETRIES:
                        if retry_attempt > 0: 
                            print(f"[{item_key}/{answer_type_key}] Retrying LLM mapping (Attempt {retry_attempt + 1}/{MAX_RETRIES})")
                        
                        mapped_csv_txt, temp_mapped_df, error_info = read_csv_mapped_with_llm(predict_txt_content, gold_header_string, current_base_url, current_api_key)
                        mapped_csv_txt_for_bad_case = mapped_csv_txt
                        error_info_for_bad_case = error_info

                        if isinstance(temp_mapped_df, pd.DataFrame) and not temp_mapped_df.empty:
                            mapped_df = temp_mapped_df
                            save_dataframe_to_csv(mapped_df, mapped_answer_csv_path_typed, index=False, encoding='utf-8-sig')
                            print(f"[{item_key}/{answer_type_key}] LLM mapping successful.")
                            break
                        else:
                            print(f"[{item_key}/{answer_type_key}] LLM mapping failed: {error_info}")
                            if retry_attempt < MAX_RETRIES - 1: 
                                time.sleep(1)
                        retry_attempt += 1

                if mapped_df is None or mapped_df.empty:
                    local_bad_case_info = {
                        "key": item_key,
                        "answer_type": answer_type_key,
                        "mapped_csv_txt": mapped_csv_txt_for_bad_case,
                        "mapped_error": error_info_for_bad_case
                    }
                    bad_cases_for_item.append(local_bad_case_info)
                    current_type_scores["cell_scores"] = {"error": f"Final mapping failed: {local_bad_case_info['mapped_error']}"}
                    current_type_scores["if_csv_parsed"]["mapped"] = 0
                        
                else:
                    current_type_scores["if_csv_parsed"]["mapped"] = 1
                    if df_gold is None or df_gold.empty:
                        current_type_scores["cell_scores"] = {"error": "Gold DataFrame (df_gold) is None or empty."}
                        print(f"[{item_key}/{answer_type_key}] Skipping cell comparison: df_gold is None or empty.")
                    else:
                        # Extract domain and task info from item_key
                        parts = item_key.split("_")
                        current_domain_part = parts[0]
                        
                        # Handle new format with language suffix
                        if len(parts) >= 4 and parts[-1] in ['en', 'zh']:  # New format
                            task_id_part = parts[-2]
                        else:  # Old format
                            task_id_part = parts[-1]
                            
                        task_name_key = f"{current_domain_part}_{task_id_part}"
                        lang = item_value.get("language", "cn")
                        task_specific_config = global_task_config.get(lang, {}).get(task_name_key)

                        if not task_specific_config:
                            current_type_scores["cell_scores"] = {"error": f"Task config not found for lang='{lang}', task='{task_name_key}'"}
                        else:
                            key_cols = task_specific_config.get("key_cols", [])
                            target_val_cols = task_specific_config.get("target_value_columns", [])
                            numeric_cols = task_specific_config.get("numeric_columns", [])

                            if not key_cols or not target_val_cols:
                                current_type_scores["cell_scores"] = {"error": f"Missing key_cols or target_value_columns in task_config for {task_name_key}"}
                            else:
                                try:
                                    cell_scores_dict = compare_table_data(
                                        df_gold.copy(), mapped_df.copy(), key_cols, target_val_cols, numeric_cols,
                                        non_numeric_match_method='llm', llm_failure_score=0.0, 
                                        current_base_url=current_base_url, current_api_key=current_api_key
                                    )
                                    print(cell_scores_dict)
                                    current_type_scores["cell_scores"] = cell_scores_dict
                                except Exception as e:
                                    print(f"Error during compare_table_data for {item_key}/{answer_type_key}: {e}")
                                    current_type_scores["cell_scores"] = {"error": f"compare_table_data failed: {str(e)}"}

        # Store scores for this answer_type
        overall_scores_for_item[answer_type_key] = current_type_scores
        save_output_json(current_type_scores, scores_json_path_typed)

    item_value["scores"] = overall_scores_for_item
    
    final_bad_case_info = bad_cases_for_item if bad_cases_for_item else None
    return item_key, item_value, final_bad_case_info


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions with concurrency.")
    parser.add_argument('--model-name', default="google/gemini-2.5-flash-preview", type=str, help='Model name')
    parser.add_argument('--domains', nargs='+', default=["academic", "financial", "legal"], help='Domains to process')
    parser.add_argument('--prompt_setting', default="all", help='Prompt setting')
    parser.add_argument('--others', default="", help='Other settings')
    parser.add_argument('--max_workers', type=int, default=1, help='Max worker threads')
    parser.add_argument('--api_key', default="xxxx", help='API key for the LLM service.')
    parser.add_argument('--base_url', default="http://124.16.138.150:2735/v1", help='API base URL.')
    args = parser.parse_args()

    current_model_name = args.model_name
    current_model_name_safe = current_model_name.replace("/", "_")
    current_prompt_setting = args.prompt_setting
    current_api_key = args.api_key
    current_base_url = args.base_url

    if "rag" in args.others:
        path_others = "rag"
        current_others = args.others
    else:
        path_others = args.others
        current_others = args.others
    
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    think_tag = think_tag_mapping.get(current_model_name_safe, "think")

    global_task_config = task_config

    for domain_name in args.domains:
        print(f"\n--- Processing Domain: {domain_name} for Model: {current_model_name} ---")
        
        predict_result_path = get_predict_result_filepath(
            current_prompt_setting, path_others, current_model_name, domain_name
        )
        datas_for_domain = load_data(predict_result_path, "jsonl")
        if not datas_for_domain:
            print(f"No data found for domain {domain_name} at {predict_result_path}. Skipping.")
            continue

        all_updated_items_for_domain = []
        all_bad_cases_for_domain = []

        # Prepare items for processing - handle both old and new data formats
        items_to_process = []
        for item_dict in datas_for_domain:
            if item_dict and isinstance(item_dict, dict):
                # Handle both old format {key: value} and new format with record_id
                if 'record_id' in item_dict:
                    # New format
                    key = item_dict['record_id']
                    value = item_dict
                    items_to_process.append((key, value))
                elif len(item_dict) == 1:
                    # Old format
                    for key, value in item_dict.items():
                        items_to_process.append((key, value))

        if not items_to_process:
            print(f"No valid items extracted from data for domain {domain_name}. Skipping processing loop.")
            continue

        # Use functools.partial to fix arguments for the worker function
        worker_fn = partial(process_item_concurrently,
                          current_model_name=current_model_name,
                          current_prompt_setting=current_prompt_setting,
                          current_others=current_others,
                          current_think_tag=think_tag,
                          global_task_config=global_task_config,
                          current_api_key=current_api_key,
                          current_base_url=current_base_url)
        
        num_threads = args.max_workers if args.max_workers else min(32, (os.cpu_count() or 1) + 4)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_fn, item_tuple) for item_tuple in items_to_process]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating items in {domain_name}"):
                try:
                    processed_key, processed_value, bad_case = future.result()
                    
                    # Handle output format based on input format
                    if 'record_id' in processed_value:
                        # New format - save as single dict
                        all_updated_items_for_domain.append(processed_value)
                    else:
                        # Old format - save as {key: value}
                        all_updated_items_for_domain.append({processed_key: processed_value})
                    
                    if bad_case:
                        all_bad_cases_for_domain.extend(bad_case)
                except Exception as e:
                    print(f"Error processing an item in domain {domain_name}: {e}")
                    import traceback
                    traceback.print_exc()

        # Save aggregated results for the domain
        if all_updated_items_for_domain:
            data_evaluated_path = get_evaluation_filepath(
                current_prompt_setting, current_others, current_model_name, f"eval_details_{domain_name}.jsonl"
            )
            save_data_jsonl(data_evaluated_path, all_updated_items_for_domain)
            print(f"Saved evaluation details for domain {domain_name} to {data_evaluated_path}")

        if all_bad_cases_for_domain:
            output_predict_name = f'output_predict_result_{current_prompt_setting}'
            if PATHS.get(output_predict_name):
                bad_case_base_dir = os.path.join(PATHS[output_predict_name], f'{current_others}_{current_model_name_safe}')
                os.makedirs(bad_case_base_dir, exist_ok=True)
                mapped_bad_case_path = os.path.join(
                    bad_case_base_dir,
                    f'{domain_name}_mapped_bad_cases_get_df_{current_timestamp}.jsonl'
                )

                save_data_jsonl(mapped_bad_case_path, all_bad_cases_for_domain)
                print(f"Saved bad mapping cases for domain {domain_name} to {mapped_bad_case_path}")
            else:
                print(f"Warning: Base path key '{output_predict_name}' not found in PATHS. Cannot save bad cases for {domain_name}.")

    print("\nEvaluation complete for all specified domains.")


if __name__ == "__main__":
    main()