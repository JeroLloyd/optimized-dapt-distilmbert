import argparse
import os
import torch
import gc
import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score, accuracy_score

from nlp_thesis.utils import get_logger, set_seed, ensure_dir

# --- DEFAULT CONFIG ---
# DEFAULT_DATASET_NAME is a LABELED dataset used for supervised fine-tuning and evaluation
DEFAULT_DATASET_NAME = "ccosme/FiReCS"
DEFAULT_OUTPUT_DIR = "./models"
DEFAULT_SEED = 42

# Hardware Optimization (RTX 4060)
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 4
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_MAX_LEN = 128

logger = get_logger(__name__)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# (duplicate removed)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {"macro_f1": macro_f1, "accuracy": acc}

def prepare_dataset(dataset_name=DEFAULT_DATASET_NAME, seed=DEFAULT_SEED):
    logger.info(f"Loading {dataset_name} dataset...")
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
            
        return {
            "text_content": example["review"], 
            "label_int": label_int
        }

    logger.info("Standardizing columns and cleaning labels...")
    dataset = dataset.map(transform_data, remove_columns=dataset['train'].column_names)
    all_available_splits = []
    if 'train' in dataset: all_available_splits.append(dataset['train'])
    if 'test' in dataset: all_available_splits.append(dataset['test'])
    if 'validation' in dataset: all_available_splits.append(dataset['validation'])
    
    full_data = concatenate_datasets(all_available_splits)
    
    train_temp = full_data.train_test_split(test_size=0.2, seed=seed)
    val_test = train_temp['test'].train_test_split(test_size=0.5, seed=seed)
    
    final_splits = {
        "train": train_temp['train'],
        "validation": val_test['train'],
        "test": val_test['test']
    }
    
    logger.info(f"Splits -> Train: {len(final_splits['train'])}, Val: {len(final_splits['validation'])}, Test: {len(final_splits['test'])}")
    return final_splits

def run_training_phase(model_key, model_source, dataset_splits, tokenizer_name, batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCHS, max_len=DEFAULT_MAX_LEN, lr=DEFAULT_LEARNING_RATE):
    logger.info("\n" + "="*50)
    logger.info(f"STARTING PHASE: {model_key}")
    logger.info("="*50)

    clean_memory()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_func(examples):
        return tokenizer(examples["text_content"], padding="max_length", truncation=True, max_length=max_len)

    tokenized_datasets = {}
    for split, ds in dataset_splits.items():
        td = ds.map(tokenize_func, batched=True)
        td = td.rename_column("label_int", "labels")
        tokenized_datasets[split] = td

    model = AutoModelForSequenceClassification.from_pretrained(
        model_source, 
        num_labels=3
    ).to(get_device())

    save_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{model_key}_finetuned")
    ensure_dir(save_path)
    
    args = TrainingArguments(
        output_dir=save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        seed=None,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save training log history and plot training curves (loss / eval macro_f1)
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        logs = pd.DataFrame(trainer.state.log_history)
        os.makedirs(save_path, exist_ok=True)
        logs.to_csv(os.path.join(save_path,'train_log_history.csv'), index=False)

        plt.figure(figsize=(6,4))
        if 'loss' in logs.columns and logs['loss'].notnull().any():
            loss_df = logs[logs['loss'].notnull()].groupby('epoch')['loss'].mean()
            loss_df.plot(label='train loss')
        if 'eval_macro_f1' in logs.columns and logs['eval_macro_f1'].notnull().any():
            evals = logs[logs['eval_macro_f1'].notnull()]
            plt.plot(evals['epoch'], evals['eval_macro_f1'], marker='o', label='eval macro_f1')

        plt.legend(); plt.xlabel('epoch'); plt.tight_layout()
        plt.savefig(os.path.join(save_path,'training_curves.png'))
    except Exception as e:
        logger.warning(f"Could not save finetune training logs/plots: {e}")

    trainer.save_model(save_path)
    logger.info(f"SUCCESS: {model_key} saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Run fine-tuning phases for models A/B/C')
    parser.add_argument('--dataset', default=DEFAULT_DATASET_NAME, help='Labeled dataset name or local CSV for supervised fine-tuning (default: ccosme/FiReCS)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--max-len', type=int, default=DEFAULT_MAX_LEN)
    args = parser.parse_args()

    set_seed(args.seed)
    logger.info(f"Seed set to {args.seed}")

    if not os.path.exists(DEFAULT_OUTPUT_DIR):
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        
    dataset_splits = prepare_dataset(args.dataset)

    # Model A: Baseline
    run_training_phase("model_A", "distilbert-base-multilingual-cased", dataset_splits, "distilbert-base-multilingual-cased", batch_size=args.batch_size, epochs=args.epochs, max_len=args.max_len)

    # Model B: DAPT
    dapt_weights = "./models/dapt-distilmbert"
    if os.path.exists(dapt_weights):
        run_training_phase("model_B", dapt_weights, dataset_splits, "distilbert-base-multilingual-cased", batch_size=args.batch_size, epochs=args.epochs, max_len=args.max_len)
    else:
        logger.warning(f"DAPT weights not found at {dapt_weights}. Skipping Model B.")
    
    # Model C: XLM-R
    run_training_phase("model_C", "xlm-roberta-base", dataset_splits, "xlm-roberta-base", batch_size=args.batch_size, epochs=args.epochs, max_len=args.max_len)

if __name__ == "__main__":
    main()