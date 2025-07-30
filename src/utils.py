"""
Data processing utilities for evaluation tasks.

This module provides functions for handling CSV, JSON, and text data
in machine learning evaluation workflows.
"""

import json
import os
import glob
import re
import csv
import io
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import OrderedDict

import pandas as pd
import jsonlines
from docx import Document


# CSV Processing Functions
def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    """Load CSV file safely, return None if failed."""
    try:
        df = pd.read_csv(path)
        return df if isinstance(df, pd.DataFrame) and not df.empty else None
    except Exception:
        return None


def validate_csv_structure(csv_text: str) -> None:
    """Validate CSV structure integrity."""
    reader = csv.reader(io.StringIO(csv_text))
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("Empty CSV content, missing header row")
    
    expected_cols = len(header)
    for row_num, row in enumerate(reader, 2):
        if len(row) != expected_cols:
            raise ValueError(f"Row {row_num}: expected {expected_cols} columns, got {len(row)}")


def extract_csv_from_text(text: str, think_tag: str = "think") -> Tuple[str, str]:
    """Extract CSV content from text with optional think tags."""
    # Remove think tags if present
    think_pattern = rf"<{think_tag}>(.*?)</{think_tag}>"
    think_match = re.search(think_pattern, text, re.DOTALL)
    content = think_match.group(1) if think_match else text
    
    # Extract CSV from code blocks
    csv_matches = re.findall(r"```csv(.*?)```", content, re.DOTALL)
    if csv_matches:
        return csv_matches[-1].strip(), "csv_found"
    
    return content.strip(), "csv_not_found"


def read_dataframe_from_llm_output(raw_text: str) -> Tuple[Union[pd.DataFrame, str], str]:
    """Parse DataFrame from LLM output containing CSV."""
    csv_content, status = extract_csv_from_text(raw_text)
    
    if status == "csv_not_found":
        return raw_text, "No CSV content found"
    
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        return df, "success"
    except Exception as e:
        return csv_content, f"CSV parsing error: {e}"


# JSON Processing Functions
def extract_json_array_from_text(text: str) -> Optional[List[Dict]]:
    """Extract JSON array from text with multiple fallback strategies."""
    text = text.strip()
    
    # Strategy 1: JSON code blocks
    json_match = re.search(r'```json\s*(\[.*?\])\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Plain JSON arrays
    array_match = re.search(r'(\[.*?\])', text, re.DOTALL)
    if array_match:
        try:
            # Handle both single and double quotes
            json_str = array_match.group(1).replace("'", '"')
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return None


def extract_complex_json_from_text(text: str) -> Union[List[Dict], int, Exception]:
    """Extract complex JSON structures with preprocessing."""
    try:
        # Preprocessing: clean problematic characters
        text = re.sub(r'_(?=《)', '', text)  # Remove underscores before legal texts
        text = re.sub(r'(?<=》)_', '', text)  # Remove underscores after legal texts
        
        # Find outermost brackets
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            return 2  # No brackets found
            
        json_content = text[start_idx:end_idx + 1]
        
        # Fix common JSON formatting issues
        json_content = re.sub(r'"}(\s*)"', '"},\n\\1"', json_content)  # Missing commas
        json_content = re.sub(r',(\s*)]', r'\1]', json_content)  # Trailing commas
        
        return json.loads(json_content)
        
    except json.JSONDecodeError as e:
        return e
    except Exception:
        return 3  # General error


# File I/O Functions
def save_dataframe_to_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
    """Save DataFrame to CSV with directory creation."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig', **kwargs)


def save_text_file(content: str, path: str) -> None:
    """Save text content to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def save_json_file(data: Any, path: str, indent: int = 4) -> None:
    """Save data as JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def save_jsonl_file(data: List[Dict], path: str, append: bool = False) -> None:
    """Save data as JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = 'a' if append else 'w'
    
    with open(path, mode, encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def append_to_jsonl(item: Dict, path: str) -> None:
    """Append single item to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')


# Data Loading Functions
def load_data_file(path: str, file_type: str) -> Union[str, List, Dict]:
    """Load data from various file formats."""
    if file_type == "txt":
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif file_type == "json":
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif file_type == "jsonl":
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    elif file_type == "docx":
        doc = Document(path)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs)
    
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def load_filtered_jsonl(path: str, domain: Optional[str] = None, 
                       task_ids: Optional[List[str]] = None) -> List[Dict]:
    """Load JSONL with optional domain and task filtering."""
    data = load_data_file(path, "jsonl")
    
    if not domain:
        return data
    
    filtered_data = []
    for item in data:
        for key in item.keys():
            parts = key.split("_")
            if len(parts) >= 3:
                item_domain = parts[0]
                item_task_id = parts[-1]
                
                if item_domain == domain:
                    if not task_ids or item_task_id in task_ids:
                        filtered_data.append(item)
                        break
    
    return filtered_data


# Validation Functions
def is_valid_csv_file(path: str) -> bool:
    """Check if CSV file exists and is valid."""
    if not os.path.isfile(path):
        return False
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Try to read at least one row
            return True
    except Exception:
        return False


# Data Merging Functions
def merge_json_to_jsonl(root_dir: str, output_file: str) -> None:
    """Merge JSON files from subdirectories into a single JSONL file."""
    all_data = []
    
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            json_files = glob.glob(os.path.join(folder_path, "*.json"))
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.append(data)
    
    # Sort by key pattern (domain_X_Y)
    def sort_key(item: Dict) -> Tuple[int, int]:
        key = list(item.keys())[0]
        parts = key.split("_")
        return int(parts[1]), int(parts[2])
    
    all_data.sort(key=sort_key)
    
    save_jsonl_file(all_data, output_file)
    print(f"Merged {len(all_data)} items to {output_file}")


def parse_key_components(key: str) -> Tuple[str, str, str]:
    """Parse domain, folder, and task from key string."""
    parts = key.split("_")
    return parts[0], parts[-2], parts[-1]


# Legal Text Processing (Domain-specific)
def extract_law_names(answers: List[Dict]) -> List[str]:
    """Extract law names from legal case data."""
    law_names = set()
    
    for case in answers:
        references = case.get("关联索引", [])
        for reference in references:
            matches = re.findall(r"《(.*?)》", reference)
            for match in matches:
                law_name = match.strip()
                if _is_valid_law_name(law_name):
                    law_names.add(law_name)
    
    return list(law_names)


def _is_valid_law_name(name: str) -> bool:
    """Validate extracted law name."""
    return (
        "法" in name and
        not any(char in name for char in ["\\", "/", ":", "(", ")"]) and
        not name.strip().isdigit()
    )


# Prediction Processing Functions
def save_prediction_results(value: Dict, answer_type_key: str, think_tag: str, 
                          txt_path: str, raw_path: str, csv_path: str) -> Tuple[str, str]:
    """Save prediction results in multiple formats."""
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    
    raw_content = value[answer_type_key]
    if isinstance(raw_content, dict):
        raw_content = raw_content.get("result", "")
    
    if not isinstance(raw_content, str):
        save_text_file("null", raw_path)
        return None, "invalid_input"
    
    save_text_file(raw_content, raw_path)
    
    csv_content, status = extract_csv_from_text(raw_content, think_tag)
    save_text_file(csv_content, txt_path)
    
    if status == "csv_found":
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            save_dataframe_to_csv(df, csv_path)
            return csv_content, "csv_valid"
        except Exception:
            return csv_content, "csv_invalid"
    
    return csv_content, "csv_not_found"


def save_gold_standard(value: Dict, txt_path: str, csv_path: str) -> Tuple[str, str, pd.DataFrame]:
    """Save gold standard answers in txt and csv formats."""
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Save as CSV
    answers = value['answers']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if answers:
            writer = csv.DictWriter(f, fieldnames=answers[0].keys())
            writer.writeheader()
            for item in answers:
                row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
                      for k, v in item.items()}
                writer.writerow(row)
    
    # Load and save as txt
    df_gold = pd.read_csv(csv_path)
    gold_csv_text = df_gold.to_csv(index=False)
    save_text_file(gold_csv_text, txt_path)
    
    return txt_path, csv_path, df_gold