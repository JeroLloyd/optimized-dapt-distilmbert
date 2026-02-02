import argparse
import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
from sklearn.metrics import f1_score
from datasets import load_dataset, concatenate_datasets

from nlp_thesis.utils import get_logger, set_seed

# --- DEFAULT CONFIG ---
# DEFAULT_DATASET_NAME is a LABELED dataset used for evaluation (default: FiReCS)
DEFAULT_DATASET_NAME = "ccosme/FiReCS"
DEFAULT_MODELS = {
    "Model_A_Base": "./models/model_A_finetuned",
    "Model_B_DAPT": "./models/model_B_finetuned",
    "Model_C_XLMR": "./models/model_C_finetuned",
    "Model_D_Opt":  "./models/model_D_optimized"
}
# Module-level active models map (can be overridden later if needed)
MODELS = DEFAULT_MODELS

DEFAULT_SEED = 42

logger = get_logger(__name__)

def load_test_data(dataset_name=DEFAULT_DATASET_NAME, seed=DEFAULT_SEED):
    dataset = load_dataset(dataset_name)
    label_map = {"negative": 0, "positive": 1, "neutral": 2}
    
    def transform_data(example):
        current_label = example["label"]
        if isinstance(current_label, (int, float)):
            label_int = int(current_label)
        elif isinstance(current_label, str):
            label_int = label_map.get(current_label.lower(), 2)
        else:
            label_int = 2
        return {"text": example["review"], "label": label_int}

    all_available = []
    if 'train' in dataset: all_available.append(dataset['train'])
    if 'test' in dataset: all_available.append(dataset['test'])
    if 'validation' in dataset: all_available.append(dataset['validation'])
    
    full_data = concatenate_datasets(all_available)
    full_data = full_data.map(transform_data, remove_columns=full_data.column_names)
    
    train_temp = full_data.train_test_split(test_size=0.2, seed=seed)
    val_test = train_temp['test'].train_test_split(test_size=0.5, seed=seed)
    return val_test['test']

def get_model_size(path):
    """
    Calculates size of the specific model file only, ignoring checkpoint folders.
    """
    total_size = 0
    # Files that constitute the model weights
    target_files = ["model.safetensors", "pytorch_model.bin", "model_quantized.onnx", "model.onnx"]
    
    found = False
    for f in os.listdir(path):
        if f in target_files:
            fp = os.path.join(path, f)
            total_size = os.path.getsize(fp)
            found = True
            break # Count only the main weight file
            
    if not found:
        # Fallback: if no specific file found, sum root dir only (excluding subdirs)
        total_size = sum(os.path.getsize(os.path.join(path, f)) 
                         for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))

    return total_size / (1024 * 1024) # Size in MB

def benchmark():
    test_data = load_test_data()
    results = []
    
    logger.info(f"Benchmarking on {len(test_data)} examples...")

    for name, path in MODELS.items():
        if not os.path.exists(path):
            logger.warning(f"Skipping {name}: Path {path} not found.")
            continue
            
        logger.info(f"Testing {name}...")
        tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Load Model
        if "Model_D" in name:
            # FIX: Explicitly specify the file name to silence warnings
            model = ORTModelForSequenceClassification.from_pretrained(
                path, 
                file_name="model_quantized.onnx", 
                provider="CPUExecutionProvider"
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(path).to("cpu")
            model.eval()

        all_preds = []
        all_labels = []
        latencies = []

        for i in range(len(test_data)):
            text = test_data[i]["text"]
            label = test_data[i]["label"]
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            
            # Remove token_type_ids for DistilBERT
            if "token_type_ids" in inputs:
                inputs.pop("token_type_ids")
            
            # Warmup
            if i == 0:
                # Warm-up
                with torch.no_grad():
                    model(**inputs)

            start = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
            end = time.time()
            
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            
            if isinstance(logits, torch.Tensor):
                pred = torch.argmax(logits, dim=-1).item()
            else:
                pred = np.argmax(logits, axis=-1).item()

            latencies.append((end - start) * 1000)
            all_preds.append(pred)
            all_labels.append(label)

        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        avg_latency = np.mean(latencies)
        model_size = get_model_size(path)
        
        results.append({
            "Model": name,
            "F1_Score": macro_f1,
            "Latency_ms": avg_latency,
            "Size_MB": model_size
        })
        logger.info(f"  > F1: {macro_f1:.4f} | Latency: {avg_latency:.2f}ms | Size: {model_size:.2f}MB")

    # --- VISUALIZATION ---
    if not results:
        logger.warning("No models were benchmarked.")
        return

    df = pd.DataFrame(results)
    logger.info("\nFinal Results Summary:")
    logger.info('\n%s', df)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. F1 Score
    ax[0].bar(df['Model'], df['F1_Score'], color='skyblue')
    ax[0].set_title('Macro F1-Score (Higher is Better)')
    ax[0].set_ylim(0, 1.0)
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Latency
    ax[1].bar(df['Model'], df['Latency_ms'], color='salmon')
    ax[1].set_title('Latency (ms) (Lower is Better)')
    ax[1].grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Size
    ax[2].bar(df['Model'], df['Size_MB'], color='lightgreen')
    ax[2].set_title('Model Size (MB) (Lower is Better)')
    ax[2].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    out_png = 'thesis_benchmarks.png'
    plt.savefig(out_png)
    logger.info(f"Graphs saved as '{out_png}'")
    df.to_csv("thesis_results.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DEFAULT_DATASET_NAME, help='Labeled dataset name or local CSV to use for evaluation (default: ccosme/FiReCS)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    logger.info(f"Seed set to {args.seed}")

    benchmark()