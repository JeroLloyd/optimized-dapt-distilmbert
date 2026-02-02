import argparse
import logging
import pandas as pd
from datasets import load_dataset
import re
import os

from nlp_thesis.utils import get_logger, set_seed, ensure_dir

# --- DEFAULT CONFIGURATION ---
# NOTE: DAPT uses the Lazada local corpus (unlabeled) by default. No synthetic data is generated.
DEFAULT_LOCAL_QA_PATH = "LazadaQA-Taglish-7k.csv"
DEFAULT_OUTPUT_FILE = "dapt_corpus_clean.txt"
DEFAULT_USE_LOCAL_ONLY = True
DEFAULT_SEED = 42

logger = get_logger(__name__) 



# --- CLEANING FUNCTION (Strict adherence to Manuscript Sec 3.3.3) ---
def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Lowercase (Manuscript Constraint)
    text = text.lower()
    
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 4. Remove PII (emails, phone numbers)
    text = re.sub(r'\S+@\S+', '', text) 
    text = re.sub(r'\d{11}', '', text) 
    
    # 5. Remove redundant whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def prepare_corpus():
    logger.info("Starting data preparation (text-only, ignoring ratings)...")
    all_texts = []

    # --- PART A: LOAD HUGGING FACE DATASET (Reviews) ---
    if not USE_LOCAL_ONLY:
        logger.info("Loading SEACrowd/lazada_review_filipino dataset...")
        try:
            # Load all splits
            hf_dataset = load_dataset("SEACrowd/lazada_review_filipino", trust_remote_code=True)
            
            for split in hf_dataset.keys():
                logger.debug(f"Processing split: {split}")
                dataset_split = hf_dataset[split]
                
                # Target text columns, ignore ratings
                target_col = None
                if 'review_text' in dataset_split.column_names:
                    target_col = 'review_text'
                elif 'text' in dataset_split.column_names:
                    target_col = 'text'
                
                if target_col:
                    texts = dataset_split[target_col]
                    all_texts.extend(texts)
                    logger.debug(f"Extracted {len(texts)} reviews from split {split}.")
                    
        except Exception as e:
            logger.warning(f"Error loading Hugging Face dataset: {e}")
    else:
        logger.info("Skipping Hugging Face sources: using local file only for DAPT corpus.")

    # --- PART B: LOAD LOCAL CSV (QA) ---
    logger.info(f"Loading local file: {LOCAL_QA_PATH}...")
    if os.path.exists(LOCAL_QA_PATH):
        try:
            df = pd.read_csv(LOCAL_QA_PATH)
            
            # Heuristic: Look for text columns, ignore ratings/IDs
            text_cols = []
            possible_headers = ['question', 'answer', 'text', 'body', 'review', 'content']
            
            for col in df.columns:
                col_lower = col.lower()
                if any(x in col_lower for x in possible_headers) and 'rating' not in col_lower and 'id' not in col_lower:
                    text_cols.append(col)
            
            if not text_cols:
                for col in df.select_dtypes(include=['object']).columns:
                    if 'rating' not in col.lower() and 'label' not in col.lower():
                        text_cols.append(col)

            logger.debug(f"Extracting text from columns: {text_cols}")
            
            for col in text_cols:
                col_texts = df[col].dropna().astype(str).tolist()
                all_texts.extend(col_texts)
                
        except Exception as e:
            logger.warning(f"Error loading CSV: {e}")
    else:
        logger.warning(f"File not found: {LOCAL_QA_PATH}")

    # --- PART C: CLEANING & DEDUPLICATION ---
    logger.info(f"Cleaning {len(all_texts)} raw samples...")
    
    cleaned_texts = []
    for t in all_texts:
        clean = clean_text(t)
        if len(clean.split()) > 3: # Must be > 3 words
            cleaned_texts.append(clean)
    
    unique_texts = list(set(cleaned_texts))
    
    logger.info(f"Final unique unlabeled samples: {len(unique_texts)}")

    # --- PART D: SAVE TO DISK ---
    logger.info(f"Saving to {OUTPUT_FILE}...")
    ensure_dir(os.path.dirname(OUTPUT_FILE) or '.')
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for text in unique_texts:
            f.write(text + "\n")
    logger.info("Data preparation complete. Ready for training.")

def main():
    parser = argparse.ArgumentParser(description='Prepare DAPT corpus (cleaning + deduplication). Uses Lazada local unlabeled data for DAPT (no synthetic data).')
    parser.add_argument('--input', '-i', default=DEFAULT_LOCAL_QA_PATH, help='Local QA CSV path')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT_FILE, help='Output text file for DAPT')
    parser.add_argument('--use-local-only', action='store_true', default=DEFAULT_USE_LOCAL_ONLY, help='Use only the local dataset (skip HF)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # apply CLI config
    global LOCAL_QA_PATH, OUTPUT_FILE, USE_LOCAL_ONLY
    LOCAL_QA_PATH = args.input
    OUTPUT_FILE = args.output
    USE_LOCAL_ONLY = args.use_local_only

    set_seed(args.seed)
    logger.info(f"Seed set to {args.seed}")

    prepare_corpus()

if __name__ == "__main__":
    main()