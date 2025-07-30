import os
import sys
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
import concurrent.futures
import threading
from typing import List, Any, Dict, Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
import shutil

from langchain.chains import LLMChain
from src.llm.call_llm import call_llm_generate
from src.utils import (
    load_data,
    save_data_json_add,
    get_domain_folder_task,
    extract_law_names,
)
from config_loader import PATHS, prompt_templates, get_predict_result_filepath

from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain_core.retrievers import BaseRetriever

timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TOP_K = 50
VECTOR_DB_CACHE_BASE_DIR = os.path.join(PATHS["base_path"], "vectordb_item_cache")
os.makedirs(VECTOR_DB_CACHE_BASE_DIR, exist_ok=True)


class ManualRetriever(BaseRetriever):
    docs: List[Document]

    def __init__(self, docs: List[Document], **kwargs: Any):
        super().__init__(docs=docs, **kwargs)
        self.docs = docs

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        return self.docs

    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)


def get_prompt(key, value, prompt_setting):
    parts = key.split("_")
    domain = parts[0]
    task_index = parts[-2] if len(parts) > 2 else parts[-1]  # Handle both old and new formats
    query = value["query"]
    prompt_template_lang = prompt_templates[value["language"]]

    example_str = lambda p_template, d, t: p_template[f"Get_predict_{d}"][f"Task{'' if d in ['financial', 'legal'] else '_' + t}"]['Example']

    if prompt_setting == "all":
        instruction_key = 'Instruction'
        instruction = prompt_template_lang[f"Get_predict_{domain}"][f"Task{'' if domain in ['financial', 'legal'] else '_' + task_index}"][instruction_key]
        output = prompt_template_lang[f"Get_predict_{domain}"][f"Task{'' if domain in ['financial', 'legal'] else '_' + task_index}"]['Output']
        example = example_str(prompt_template_lang, domain, task_index)
        input_prompt = f"{instruction}\n\n{example}\n\n{output}"
    else:
        raise ValueError(f"Unknown prompt_setting: {prompt_setting}")
    return input_prompt


def initialize_vectorstore(documents, persist_directory=None, behavior="load_or_create"):
    """
    Initialize or load vector database with documents.
    
    Args:
        documents: List of documents for building vector store
        persist_directory: Path for vector database persistence. If None, creates in-memory store
        behavior: Behavior mode
                  "load_or_create": Load if persist_directory exists with data, otherwise create new
                  "force_create_new": Always create new. Clear persist_directory if it exists
    """
    embeddings = OpenAIEmbeddings(
        model="bge-m3",
        openai_api_base=PATHS["cip_base_url"],
        openai_api_key=PATHS["api_key"]
    )

    if persist_directory:
        if behavior == "force_create_new":
            if os.path.exists(persist_directory):
                print(f"Behavior 'force_create_new': Directory {persist_directory} exists, clearing it.")
                shutil.rmtree(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)
            print(f"Creating new persistent vector database in {persist_directory}...")
            vectorstore = Chroma.from_documents(
                documents,
                embeddings,
                persist_directory=persist_directory
            )
            print("Persistent vector database created.")

        elif behavior == "load_or_create":
            if os.path.exists(persist_directory) and os.listdir(persist_directory):
                print(f"Behavior 'load_or_create': Loading vector database from {persist_directory}...")
                vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embeddings
                )
                print("Vector database loaded.")
            else:
                os.makedirs(persist_directory, exist_ok=True)
                print(f"Behavior 'load_or_create': Directory {persist_directory} empty or doesn't exist, creating new...")
                vectorstore = Chroma.from_documents(
                    documents,
                    embeddings,
                    persist_directory=persist_directory
                )
                print("New persistent vector database created.")
        else:
            raise ValueError(f"Unknown behavior: {behavior}")
    else:
        print("Creating new in-memory vector database...")
        vectorstore = Chroma.from_documents(documents, embeddings)
        print("In-memory vector database created.")

    return vectorstore


def load_and_split_documents(key, value):
    """
    Load and split documents based on key and value.
    key format examples: "academic_WikiText-103_1_en", "financial_23", "legal_0"
    """
    try:
        domain = key.split("_")[0]
        # Handle new format with language suffix
        parts = key.split("_")
        if len(parts) >= 4 and parts[-1] in ['en', 'zh']:  # New format with language
            task_id = parts[-2]
        else:  # Old format
            task_id = parts[-1]
    except IndexError:
        logging.error(f"Invalid key format: {key}. Expected format like 'domain_dataset_taskid_lang'.")
        return []

    all_documents = []

    # Determine base document path
    if domain == 'academic':
        doc_path = PATHS['data']['academic']
    elif domain == 'financial':
        doc_path = PATHS['data']['financial']
    elif domain == 'legal':
        doc_path = PATHS['data']['law_docx']
    else:
        logging.error(f"Unknown domain: {domain}")
        return []

    print(f"Processing key: {key}, domain: {domain}, task_id: {task_id}, doc_path: {doc_path}")

    # Handle legal domain document loading
    if domain == "legal":
        input_doc_list = value.get('input_doc', [])

        # Process text content in input_doc
        if isinstance(input_doc_list, list):
            for item in input_doc_list:
                if isinstance(item, dict):
                    case_name = item.get("案件名", "未命名案件")
                    case_id = item.get("入库编号", "")
                    case_content = item.get("基本案情", "")
    
                    text_content = f"""
                    案件名称：{case_name}
                    入库编号：{case_id}
                    案件详情：{case_content}
                    """

                    metadata = {"source": case_name}
                    all_documents.append(Document(page_content=text_content, metadata=metadata))
                    print(f"Loaded text content from value['input_doc']: {case_name}")

        # If task_id == "0", also load legal document .docx files
        if task_id == "0":
            unique_law_names = extract_law_names(value.get("answers", []))
            print(f"task_id is 0, loading legal documents: {unique_law_names}")
            try:
                for law_name in unique_law_names:
                    if isinstance(law_name, str) and law_name:
                        law_path = os.path.join(PATHS['data']['law_docx'], f'{law_name}.docx')
                        if os.path.exists(law_path):
                            print(f"Loading legal document: {law_path}")
                            law_loader = UnstructuredWordDocumentLoader(law_path)
                            law_documents = law_loader.load()
                            all_documents.extend(law_documents)
                        else:
                            logging.warning(f"Legal document file not found: {law_path}")
                    else:
                        logging.warning(f"Invalid legal document name: {law_name}")
            except Exception as e:
                logging.error(f"Failed to load legal .docx files: {str(e)}")

    # Handle file names in input_doc (e.g., .md files)
    else:
        input_doc_files = value.get('input_doc', [])
        if isinstance(input_doc_files, list):
            for doc_item in input_doc_files:
                if isinstance(doc_item, str) and doc_item.endswith(".md"):
                    current_doc_path = None
                    if domain == 'academic':
                        current_doc_path = PATHS['data']['academic']
                    elif domain == 'financial':
                        current_doc_path = PATHS['data']['financial']

                    if current_doc_path:
                        filepath = os.path.join(current_doc_path, doc_item)
                        if os.path.exists(filepath):
                            print(f"Loading .md file: {filepath}")
                            try:
                                loader = UnstructuredMarkdownLoader(
                                    filepath,
                                    mode="single",
                                    strategy="fast",
                                )
                                documents = loader.load()
                                all_documents.extend(documents)
                            except Exception as e:
                                logging.error(f"Failed to load .md file {filepath}: {str(e)}")
                        else:
                            logging.warning(f".md file not found: {filepath}")
                    else:
                        logging.warning(f"No .md file path configured for domain {domain}. Cannot load: {doc_item}")

    if not all_documents:
        print(f"No documents found for key: {key}")
        return []
    
    print(f"Total loaded {len(all_documents)} document pages.")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    split_documents = text_splitter.split_documents(all_documents)
    print(f"Generated {len(split_documents)} text chunks after splitting.")
    return split_documents


class DataProcessor:
    def __init__(self, api_key, domain="legal", model_name="llama_31_70b_instruct_int4", 
                 base_url="http://124.16.138.150:2730/v1", prompt_setting="all", others="", max_workers=5):
        self.domain = domain
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.prompt_setting = prompt_setting
        self.others = others
        self.max_workers = max_workers
        self.file_lock = threading.Lock()

        self.input_path = PATHS['data']['data_input']
        
        self.predict_result_path = get_predict_result_filepath(
            prompt_setting=self.prompt_setting,
            others=self.others,
            model_name=self.model_name,
            domain=domain
        )
        
        if self.domain == 'academic':
            self.doc_path = PATHS['data']['academic']
        elif self.domain == 'financial':
            self.doc_path = PATHS['data']['financial']
        elif self.domain == 'legal':
            self.doc_path = PATHS['data']['law_docx']
            
        self.rag_llm = ChatOpenAI(
            model_name=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=0
        )
        
        os.makedirs(os.path.dirname(self.predict_result_path), exist_ok=True)

    def retrieve_and_answer_option2(self, vectorstore: Chroma, user_query: str, full_prompt_template_str: str, top_k=TOP_K):
        """
        Option 2: Group retrieval by file, return answer, document count, total chunks.
        """
        print("\n--- Using Option 2 for retrieval and generation ---")
        
        # 1. Initial retrieval
        preliminary_retriever = vectorstore.as_retriever(search_kwargs={"k": max(top_k * 5, 10)})
        preliminary_docs = preliminary_retriever.get_relevant_documents(user_query)

        # 2. Group by file source and filter
        docs_by_file = {}
        for doc in preliminary_docs:
            source_file = doc.metadata.get('source', 'unknown_source')
            if source_file not in docs_by_file:
                docs_by_file[source_file] = []
            docs_by_file[source_file].append(doc)
        
        final_retrieved_docs = []
        for file_source, docs_list in docs_by_file.items():
            selected_docs_from_source = docs_list[:top_k]
            final_retrieved_docs.extend(selected_docs_from_source)

        # Remove duplicates
        unique_final_docs_map = {doc.page_content: doc for doc in final_retrieved_docs}
        final_retrieved_docs = list(unique_final_docs_map.values())
        print(f"Final retrieved {len(final_retrieved_docs)} chunks (after deduplication).")

        # Handle no results case
        if not final_retrieved_docs:
            print("Could not retrieve enough chunks from any files.")
            try:
                total_chunks = vectorstore._collection.count()
            except AttributeError:
                total_chunks = 0
            return {
                "query": user_query,
                "result": "No relevant information found.",
                "num_retrieved_docs": 0,
                "num_chunks_total": total_chunks
            }
            
        input_doc_text = ""
        sources_count = {}
        for doc in final_retrieved_docs:
            source_file = doc.metadata.get('source', 'unknown_source')
            sources_count[source_file] = sources_count.get(source_file, 0) + 1
            input_doc_text += (
                f"[Source: {source_file}, {sources_count[source_file]} chunks]\n"
                f"{doc.page_content}\n\n"
            )

        # 3. Prepare for LLM Call
        prompt_for_llm = PromptTemplate(
            input_variables=['query', 'input_doc'],
            template=full_prompt_template_str
        )

        llm_chain = LLMChain(llm=self.rag_llm, prompt=prompt_for_llm)
        
        response = llm_chain.invoke({
            "query": user_query,
            "input_doc": input_doc_text
        })

        # Assemble final result
        result = {
            "query": user_query,
            "result": response.get('text', response.get('result', '')),
            "final_retrieved_docs": final_retrieved_docs,
            "num_retrieved_docs": len(final_retrieved_docs),
            "source_breakdowns": sources_count
        }
        print(f"RAG answer: {result['result']}")
        
        try:
            result['num_chunks_total'] = vectorstore._collection.count()
        except AttributeError:
            result['num_chunks_total'] = 0

        return result
    
    def _get_input_doc_content(self, item_value, domain, task_index):
        """Helper to prepare input_doc content based on domain and task."""
        all_law_data = ""
        input_doc_content = ""

        if domain == "legal":
            if task_index == "0":
                input_doc_content = item_value['input_doc']
                unique_law_names = extract_law_names(item_value["answers"])
                try:
                    law_str = str(unique_law_names)
                    all_law_data = law_str
                    for law_name in unique_law_names:
                        law_path = os.path.join(self.doc_path, f'{law_name}.docx')
                        if os.path.exists(law_path):
                            law_data = load_data(law_path, 'docx')
                            all_law_data += "\n" + law_data
                        else:
                            logging.warning(f"Law file not found: {law_path}")
                    
                except Exception as e:
                    logging.error(f"Error processing law names for legal task 0: {str(e)}")
                input_doc_content = str(item_value["input_doc"]) + "\nRelevant Laws:\n" + all_law_data
            elif task_index == "1":
                input_doc_content = str(item_value["input_doc"])
        else:  # academic or financial
            doc_path_base = self.doc_path
            markdown_context_str = ""
            doc_list = item_value.get("input_doc", [])
            if not isinstance(doc_list, list):
                logging.warning(f"input_doc for non-legal domain is not a list: {doc_list}. Treating as empty.")
                doc_list = []

            for doc_name in doc_list:
                if not isinstance(doc_name, str):
                    logging.warning(f"Skipping invalid document name: {doc_name}")
                    continue
                
                if doc_name.endswith(".md"):
                    markdown_path = os.path.join(doc_path_base, doc_name) 
                else:
                    markdown_path = os.path.join(doc_path_base, f'{doc_name}.md') 
                
                try:
                    if os.path.exists(markdown_path):
                        with open(markdown_path, "r", encoding="utf-8") as f:
                            markdown_context = f.read()
                        markdown_context_str += markdown_context + "\n\n"
                    else:
                        logging.warning(f"Markdown file not found: {markdown_path}")
                except Exception as e:
                    logging.error(f"Error reading markdown file {markdown_path}: {e}")
            
            input_doc_content = "Input Documents: " + str(item_value.get("input_doc", "")) + "\n\nContext:\n" + markdown_context_str
        
        return input_doc_content.strip()

    def _get_llm_prediction(self, prompt_template_lang, domain, task_index, query, schema, input_doc_content):
        """Calls the LLM for a prediction, handles prompt selection."""
        if_bad_case = False
        
        # Common output formatting part of the prompt
        output_format_str = lambda p_template, d, t: p_template[f"Get_predict_{d}"][f"Task{'' if d in ['financial', 'legal'] else '_' + t}"]['Output'].format(
            query=query, input_doc=input_doc_content
        )
        example_str = lambda p_template, d, t: p_template[f"Get_predict_{d}"][f"Task{'' if d in ['financial', 'legal'] else '_' + t}"]['Example']

        if self.prompt_setting == "all":
            instruction_key = 'Instruction'
            instruction = prompt_template_lang[f"Get_predict_{domain}"][f"Task{'' if domain in ['financial', 'legal'] else '_' + task_index}"][instruction_key]
            output = output_format_str(prompt_template_lang, domain, task_index)
            example = example_str(prompt_template_lang, domain, task_index)
            input_prompt = f"{instruction}\n\n{example}\n\n{output}"
        elif self.prompt_setting == "no_cot":
            instruction_key = f'Instruction_{self.prompt_setting}'
            instruction = prompt_template_lang[f"Get_predict_{domain}"][f"Task{'' if domain in ['financial', 'legal'] else '_' + task_index}"][instruction_key]
            output = output_format_str(prompt_template_lang, domain, task_index)
            example = example_str(prompt_template_lang, domain, task_index)
            input_prompt = f"{instruction}\n\n{example}\n\n{output}"
        elif self.prompt_setting == "no_example":
            instruction_key = 'Instruction'
            instruction = prompt_template_lang[f"Get_predict_{domain}"][f"Task{'' if domain in ['financial', 'legal'] else '_' + task_index}"][instruction_key]
            output = output_format_str(prompt_template_lang, domain, task_index)
            input_prompt = f"{instruction}\n\n{output}"
        elif self.prompt_setting == "no_teach":
            instruction_key = f'Instruction_no_cot'
            instruction = prompt_template_lang[f"Get_predict_{domain}"][f"Task{'' if domain in ['financial', 'legal'] else '_' + task_index}"][instruction_key]
            output = output_format_str(prompt_template_lang, domain, task_index)
            input_prompt = f"{instruction}\n\n{output}"
        else:
            raise ValueError(f"Unknown prompt_setting: {self.prompt_setting}")

        answer, if_bad_case, max_retries = call_llm_generate(input_prompt, self.model_name, self.api_key, self.base_url)
        return answer, if_bad_case, max_retries
    
    def _process_single_item_value(self, task_data_tuple):
        """
        Processes a single item's value part based on flags.
        Returns (item_key, updated_item_value, was_modified_flag).
        """
        item_key, item_value_from_input, existing_data_for_item, \
        domain_for_item, task_index_for_item, language_for_item, \
        do_direct_llm, do_rag = task_data_tuple

        was_modified_in_this_call = False
        
        # Initialize current_processing_value
        current_processing_value = {}
        if existing_data_for_item:
            current_processing_value = existing_data_for_item.copy()
        
        # Overlay essential fields from input
        for k, v in item_value_from_input.items():
            if k not in current_processing_value or current_processing_value[k] is None:
                 current_processing_value[k] = v
        
        # Ensure key fields are from input
        current_processing_value['query'] = item_value_from_input['query']
        current_processing_value['language'] = item_value_from_input['language']
        current_processing_value['input_doc'] = item_value_from_input.get('input_doc')
        current_processing_value['answers'] = item_value_from_input.get('answers')

        rag_predict_key = f"predict_rag_{TOP_K}_{CHUNK_SIZE}_option2"

        try:
            # --- Direct LLM Call (Non-RAG) ---
            if do_direct_llm:
                logging.info(f"Running direct LLM for {item_key} (prompt: {self.prompt_setting})...")
                input_doc_content_for_direct_llm = self._get_input_doc_content(current_processing_value, domain_for_item, task_index_for_item)
                prompt_template_for_lang = prompt_templates[language_for_item]
                
                answer, if_bad_case, attempt = self._get_llm_prediction(
                    prompt_template_for_lang, domain_for_item, task_index_for_item,
                    current_processing_value['query'],
                    current_processing_value.get('table_schema'),
                    input_doc_content_for_direct_llm
                )
                
                current_processing_value["predict_answer"] = answer
                if if_bad_case:
                    current_processing_value["attempt"] = attempt
                else:
                    current_processing_value.pop("attempt", None)
                
                current_processing_value.pop("error_processing", None) 
                current_processing_value.pop("traceback", None)
                was_modified_in_this_call = True
                logging.info(f"Direct LLM for {item_key} completed.")

            # --- RAG Processing Block ---
            if do_rag and "rag" in self.others:
                logging.info(f"Running RAG for {item_key}...")
                split_docs = load_and_split_documents(item_key, current_processing_value)
                
                if not split_docs:
                    logging.warning(f"No documents for RAG on {item_key}. Marking RAG as failed.")
                    current_processing_value[rag_predict_key] = {
                        "query": current_processing_value["query"], 
                        "result": "RAG_FAILED_NO_DOCUMENTS",
                        "num_retrieved_docs": 0, 
                        "num_chunks_total": 0, 
                        "source_breakdowns": {}
                    }
                else:
                    safe_key = "".join(c if c.isalnum() else "_" for c in item_key)
                    current_cache_dir = os.path.join(VECTOR_DB_CACHE_BASE_DIR, safe_key)
                    vectorstore = initialize_vectorstore(
                        split_docs, persist_directory=current_cache_dir, behavior="load_or_create"
                    )
                    prompt_template_for_rag = get_prompt(item_key, current_processing_value, self.prompt_setting)
                    
                    rag_result_option2 = self.retrieve_and_answer_option2(
                        vectorstore, current_processing_value["query"], prompt_template_for_rag, TOP_K
                    )
                    current_processing_value[rag_predict_key] = rag_result_option2
                
                current_processing_value.pop("error_processing", None) 
                current_processing_value.pop("traceback", None)
                was_modified_in_this_call = True
                logging.info(f"RAG for {item_key} completed.")

                # If RAG successful, clear direct LLM answer
                if rag_predict_key in current_processing_value and \
                   isinstance(current_processing_value[rag_predict_key], dict) and \
                   current_processing_value[rag_predict_key].get("result") not in [None, "", "RAG_FAILED_NO_DOCUMENTS", "No relevant information found."]:
                    current_processing_value.pop("predict_answer", None)
                    current_processing_value.pop("attempt", None)
                    current_processing_value.pop("error_processing", None)
                    current_processing_value.pop("traceback", None)

            if was_modified_in_this_call and not current_processing_value.get("error_processing"):
                 current_processing_value.pop("traceback", None)

            return item_key, current_processing_value, was_modified_in_this_call

        except Exception as e:
            logging.error(f"GENERAL EXCEPTION in _process_single_item_value for {item_key}: {e}", exc_info=True)
            import traceback
            error_value = existing_data_for_item.copy() if existing_data_for_item else item_value_from_input.copy()
            for k_essential in ['query', 'language']:
                 if k_essential in item_value_from_input: 
                     error_value[k_essential] = item_value_from_input[k_essential]

            error_value["predict_answer"] = f"FATAL_ERROR_IN_WORKER ({type(e).__name__}): {str(e)}"
            error_value["error_processing"] = True
            error_value["traceback"] = traceback.format_exc()
            return item_key, error_value, True

    def get_predict_answer(self):
        datas_from_input_file = load_data(self.input_path, "jsonl")
        
        existing_results_map: Dict[str, Dict] = {}
        if os.path.exists(self.predict_result_path):
            logging.info(f"Loading existing processed data from: {self.predict_result_path}")
            try:
                loaded_existing_data_list = load_data(self.predict_result_path, "jsonl")
                for item_dict_loaded in loaded_existing_data_list:
                    if item_dict_loaded and isinstance(item_dict_loaded, dict):
                        # Handle both old format {key: value} and new format with record_id
                        if len(item_dict_loaded) == 1 and 'record_id' not in item_dict_loaded:
                            # Old format
                            key = list(item_dict_loaded.keys())[0]
                            existing_results_map[key] = item_dict_loaded[key]
                        elif 'record_id' in item_dict_loaded:
                            # New format
                            key = item_dict_loaded['record_id']
                            existing_results_map[key] = item_dict_loaded
                        
                logging.info(f"Successfully loaded {len(existing_results_map)} unique items from {self.predict_result_path}.")
            except Exception as e:
                logging.error(f"Failed to load or parse {self.predict_result_path}: {e}. Will proceed as if no prior results.")
        
        tasks_for_executor = []
        
        for item_dict_from_input in datas_from_input_file:
            if not item_dict_from_input or not isinstance(item_dict_from_input, dict):
                logging.warning(f"Skipping malformed item from input file: {item_dict_from_input}")
                continue
            
            # Handle both old and new data formats
            if 'record_id' in item_dict_from_input:
                # New format
                item_key = item_dict_from_input['record_id']
                item_value_from_input = item_dict_from_input
            elif len(item_dict_from_input) == 1:
                # Old format
                item_key = list(item_dict_from_input.keys())[0]
                item_value_from_input = item_dict_from_input[item_key]
            else:
                logging.warning(f"Skipping malformed item from input file: {item_dict_from_input}")
                continue

            current_item_domain, _, current_item_task_index = get_domain_folder_task(item_key)
            
            if current_item_domain != self.domain:
                continue  # Skip items not for the current domain processor instance

            language = item_value_from_input.get("language")
            if not language or language not in prompt_templates:
                logging.error(f"Item {item_key} missing language or lang not in templates. Skipping task generation.")
                continue

            existing_data = existing_results_map.get(item_key)
            
            process_this_item_direct_llm = False
            process_this_item_rag = False
            
            rag_predict_key = f"predict_rag_{TOP_K}_{CHUNK_SIZE}_option2"
                    
            if existing_data is None:
                logging.debug(f"Item {item_key} (lang: {language}): No existing data. Queuing for full processing.")
                process_this_item_direct_llm = True
                if "rag" in self.others:
                    process_this_item_rag = True
            else:
                # Key exists, check language and processing needs
                existing_data_language = existing_data.get("language")

                if language != existing_data_language:
                    # Different languages, treat as new item
                    logging.info(f"Item {item_key}: Existing data found (lang: {existing_data_language}), but current input language is different (lang: {language}). Queuing for full processing for '{language}'.")
                    process_this_item_direct_llm = True
                    if "rag" in self.others:
                        process_this_item_rag = True
                    existing_data_for_processing = None
                else:
                    # Same language, check what needs reprocessing
                    existing_data_for_processing = existing_data
                    logging.debug(f"Item {item_key} (lang: {language}): Existing data found with matching language. Checking if re-processing is needed.")

                    general_error_exists = existing_data.get("error_processing") is True or \
                                           ("predict_answer" in existing_data and \
                                            isinstance(existing_data["predict_answer"], str) and \
                                            "FATAL_ERROR" in existing_data["predict_answer"].upper())
                    
                    # Direct LLM re-processing logic
                    needs_direct_llm_reprocessing = False
                    if "rag" not in self.others:  # Direct LLM is primary output
                        if "predict_answer" not in existing_data or \
                           existing_data["predict_answer"] is None or \
                           existing_data["predict_answer"] == "" or \
                           general_error_exists:
                            needs_direct_llm_reprocessing = True
                            logging.debug(f"Item {item_key} (lang: {language}, no RAG primary): Needs direct LLM (missing, empty, or error).")

                    if needs_direct_llm_reprocessing:
                        process_this_item_direct_llm = True

                    # RAG re-processing logic
                    if "rag" in self.others:
                        rag_data_value = existing_data.get(rag_predict_key)
                        rag_is_successful_and_present = False
                        if isinstance(rag_data_value, dict):
                            rag_result_text = rag_data_value.get("result", "")
                            if rag_result_text and str(rag_result_text).strip() and \
                               rag_result_text not in ["RAG_FAILED_NO_DOCUMENTS", "No relevant information found."]:
                                rag_is_successful_and_present = True
                        
                        if not rag_is_successful_and_present or general_error_exists:
                            logging.debug(f"Item {item_key} (lang: {language}): Needs RAG processing (RAG result missing/failed/empty, or general error).")
                            process_this_item_rag = True
                            
            if process_this_item_direct_llm or process_this_item_rag:
                logging.info(f"Item {item_key} will be processed. DirectLLM: {process_this_item_direct_llm}, RAG: {process_this_item_rag}")
                tasks_for_executor.append(
                    (item_key, item_value_from_input, existing_data_for_processing,
                     current_item_domain, current_item_task_index, language,
                     process_this_item_direct_llm, process_this_item_rag)
                )
            else:
                logging.info(f"Item {item_key} is already complete and will be skipped.")

        if not tasks_for_executor:
            logging.info(f"No items to process or re-process for domain '{self.domain}'.")
            return

        logging.info(f"Found {len(tasks_for_executor)} items to process/re-process for domain '{self.domain}'.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task_args = {
                executor.submit(self._process_single_item_value, task_args): task_args 
                for task_args in tasks_for_executor
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_task_args), 
                               total=len(tasks_for_executor), 
                               desc=f"Processing items for domain '{self.domain}'"):
                
                original_task_args = future_to_task_args[future]
                original_item_key = original_task_args[0]

                try:
                    processed_key, processed_value, was_modified = future.result()
                    
                    if was_modified:
                        # For new format, save with record_id structure
                        if 'record_id' not in processed_value:
                            processed_value['record_id'] = processed_key
                        
                        save_data_json_add(self.predict_result_path, processed_value)
                        existing_results_map[processed_key] = processed_value
                        logging.info(f"Item {processed_key} processed/updated and appended to {self.predict_result_path}")
                    else:
                        logging.info(f"Item {processed_key} was processed by worker but reported no modifications needed to save.")
                
                except Exception as exc:
                    logging.error(f"Item {original_item_key} generated an unhandled exception in main loop: {exc}", exc_info=True)
                    # Save error placeholder
                    error_placeholder = original_task_args[1].copy()
                    error_placeholder["predict_answer"] = f"FATAL_ERROR_IN_EXECUTOR_LOOP: {type(exc).__name__} - {str(exc)}"
                    error_placeholder["error_processing"] = True
                    import traceback
                    error_placeholder["traceback"] = traceback.format_exc()
                    
                    # Ensure record_id is set for new format
                    if 'record_id' not in error_placeholder:
                        error_placeholder['record_id'] = original_item_key
                    
                    save_data_json_add(self.predict_result_path, error_placeholder)
                    existing_results_map[original_item_key] = error_placeholder

    def process(self):
        """Main processing method that orchestrates the workflow"""
        logging.info(f"Processing domain: {self.domain} with model {self.model_name}")
        logging.info(f"Input file path: {self.input_path}")
        logging.info(f"Output will be saved to: {self.predict_result_path}")
        
        self.get_predict_answer()
        
        logging.info(f"Finished processing domain: {self.domain}")


def setup_logging(log_dir, model_name):
    os.makedirs(log_dir, exist_ok=True)
    safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name)
    log_filename = f"process_log_{safe_model_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # Remove all handlers associated with the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, mode='a'),
            logging.StreamHandler(sys.stdout) 
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_filepath}")
    

def main():
    parser = argparse.ArgumentParser(description='Data processing script with concurrent LLM calls.')
    parser.add_argument('--domains', nargs='+', default=["legal", "academic", "financial"],
                        help='Specify one or more domains to process, separated by spaces.')
    parser.add_argument('--base-url', default="",
                        help='API base URL.')
    parser.add_argument('--model-name', default="QwQ-32B",
                        help='Name of the model to use.')
    parser.add_argument('--log-dir', default="",
                        help='Directory to save log files.')
    parser.add_argument('--api_key', default="",
                        help='API key for the LLM service.')
    parser.add_argument('--prompt_setting', default="all", choices=["all", "no_cot", "no_example", "no_teach"],
                        help='Prompt setting to use.')
    parser.add_argument('--others', default="",
                        help='Other identifying string for output file naming.')
    parser.add_argument('--max_workers', type=int, default=1,
                        help='Maximum number of concurrent worker threads.')
    
    args = parser.parse_args()
    print(args)

    for domain_to_process in args.domains:
        processor = DataProcessor(
            domain=domain_to_process,
            model_name=args.model_name,
            base_url=args.base_url,
            api_key=args.api_key,
            prompt_setting=args.prompt_setting,
            others=args.others,
            max_workers=args.max_workers
        )
        processor.process()
    
    logging.info("All specified domains processed.")


if __name__ == "__main__":
    main()