import argparse
import os
import math
import shutil
import torch
from transformers import (
    AutoTokenizer,
    DistilBertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

from nlp_thesis.utils import get_logger, set_seed, ensure_dir

# --- DEFAULT CONFIGURATION & HYPERPARAMETERS ---
DEFAULT_MODEL_CHECKPOINT = "distilbert-base-multilingual-cased"
DEFAULT_TRAIN_FILE = "dapt_corpus_clean.txt"
DEFAULT_OUTPUT_DIR = "./models/dapt-distilmbert"
DEFAULT_BLOCK_SIZE = 128
DEFAULT_SEED = 42

# Hardware-friendly defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_GRAD_ACCUMULATION = 1
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_NUM_EPOCHS = 3
DEFAULT_MLM_PROBABILITY = 0.15

logger = get_logger(__name__)

# Backwards-compatible module-level names used by downstream functions
BLOCK_SIZE = DEFAULT_BLOCK_SIZE
BATCH_SIZE = DEFAULT_BATCH_SIZE
GRAD_ACCUMULATION = DEFAULT_GRAD_ACCUMULATION
LEARNING_RATE = DEFAULT_LEARNING_RATE
NUM_EPOCHS = DEFAULT_NUM_EPOCHS
MLM_PROBABILITY = DEFAULT_MLM_PROBABILITY
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


# --- VISUALIZATION HELPERS ---
def generate_masking_figure(tokenizer, data_collator, sample_texts, outpath="figure3_masking_example.png"):
    """Generate a small figure showing which tokens were masked / replaced / unchanged.
    Uses the data_collator to produce masked inputs & labels, then highlights masked positions.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    enc = tokenizer(sample_texts, return_tensors='pt', padding=True, truncation=True)
    # Prepare a simple batch in the format expected by DataCollatorForLanguageModeling
    batch = [{k: enc[k][i].tolist() for k in enc.keys()} for i in range(len(sample_texts))]
    collated = data_collator(batch)

    input_ids = collated['input_ids'].cpu().numpy() if hasattr(collated['input_ids'], 'cpu') else collated['input_ids']
    labels = collated['labels'].cpu().numpy() if hasattr(collated['labels'], 'cpu') else collated['labels']

    fig, axs = plt.subplots(len(sample_texts), 1, figsize=(14, 2 * max(1, len(sample_texts))))
    if len(sample_texts) == 1:
        axs = [axs]

    for i, txt in enumerate(sample_texts):
        orig_ids = enc['input_ids'][i].numpy()
        input_row = input_ids[i]
        label_row = labels[i]

        display_tokens = []
        colors = []

        for j, (orig_id, inp_id, lbl_id) in enumerate(zip(orig_ids, input_row, label_row)):
            token_str = tokenizer.convert_ids_to_tokens(orig_id)
            if lbl_id == -100:
                display_tokens.append(token_str)
                colors.append('black')
            else:
                # masked position
                if inp_id == tokenizer.mask_token_id:
                    display_tokens.append('[MASK]')
                    colors.append('red')
                elif inp_id == lbl_id:
                    # unchanged
                    display_tokens.append(tokenizer.convert_ids_to_tokens(inp_id))
                    colors.append('orange')
                else:
                    # replaced with random
                    display_tokens.append(tokenizer.convert_ids_to_tokens(inp_id))
                    colors.append('blue')

        axs[i].axis('off')
        axs[i].set_title(txt[:120] + ('...' if len(txt)>120 else ''))
        # Render tokens with simple color coding by joining colored tokens via matplotlib text (single color to keep simple)
        axs[i].text(0, 0.5, ' '.join(display_tokens), fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath)
    logger.info(f"Masking example saved to {outpath}")


def main():
    parser = argparse.ArgumentParser(description='Run DAPT (masked LM) on domain corpus')
    parser.add_argument('--train-file', default=DEFAULT_TRAIN_FILE)
    parser.add_argument('--checkpoint', default=DEFAULT_MODEL_CHECKPOINT)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument('--mlm-prob', type=float, default=DEFAULT_MLM_PROBABILITY)
    args = parser.parse_args()

    set_seed(args.seed)
    logger.info(f"Seed set to {args.seed}")

    train_file = args.train_file
    model_checkpoint = args.checkpoint

    # expose CLI-configured hyperparams to module-level names used below
    global BATCH_SIZE, NUM_EPOCHS, MLM_PROBABILITY, OUTPUT_DIR
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    MLM_PROBABILITY = args.mlm_prob
    OUTPUT_DIR = args.output_dir

    # 2. Check for GPU and warn if not present
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Proceeding on CPU (slower).")
    else:
        logger.info(f"GPU Detected: {torch.cuda.get_device_name(0)}")

    logger.info(f"Dataset: {train_file}")

    # 3. Manual Cleanup
    if os.path.exists(OUTPUT_DIR):
        logger.info(f"Cleaning up existing output directory: {OUTPUT_DIR}")
        try:
            shutil.rmtree(OUTPUT_DIR)
        except Exception as e:
            logger.warning(f"Could not delete folder: {e}. Proceeding anyway.")

    # 4. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # 5. Load Dataset
    logger.info("Loading dataset...")
    datasets = load_dataset("text", data_files={"train": train_file})

    # 6. Tokenization
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    logger.info("Tokenizing dataset...")
    tokenized_datasets = datasets.map(
        tokenize_function, 
        batched=True, 
        num_proc=4, 
        remove_columns=["text"]
    )

    # 7. Grouping (Chunking)
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        if total_length >= BLOCK_SIZE:
            total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
            
        result = {
            k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info("Grouping texts into blocks...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
    )

    # 8. Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=MLM_PROBABILITY
    )

    # Generate a small masking example (Figure 3)
    try:
        logger.info("Generating MLM masking example (Figure 3)...")
        sample_lines = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for _ in range(3):
                l = f.readline().strip()
                if not l:
                    break
                sample_lines.append(l)
        if sample_lines:
            generate_masking_figure(tokenizer, data_collator, sample_lines, outpath=os.path.join(OUTPUT_DIR, 'figure3_masking_example.png'))
    except Exception as e:
        logger.warning(f"Could not create masking example: {e}")

    # 9. Model Initialization
    logger.info("Initializing model...")
    model = DistilBertForMaskedLM.from_pretrained(model_checkpoint)

    # 10. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        report_to="none",
        disable_tqdm=False
    )

    # 11. Trainer Initialization (Fixed for Transformers v5.0)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        data_collator=data_collator,
        # Tokenizer removed here to prevent TypeError
    )

    # 12. Execution
    logger.info("Starting Domain-Adaptive Pre-Training (DAPT)...")
    train_result = trainer.train()

    logger.info("Saving model and tokenizer...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR) # Manually save tokenizer

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save trainer log history and plot training loss curve
    try:
        import json
        import pandas as pd
        import matplotlib.pyplot as plt
        log_history = trainer.state.log_history
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, 'train_log_history.json'), 'w', encoding='utf-8') as fh:
            json.dump(log_history, fh, indent=2)
        df = pd.DataFrame(log_history)
        df.to_csv(os.path.join(OUTPUT_DIR, 'train_log_history.csv'), index=False)
        if 'loss' in df.columns and df['loss'].notnull().any():
            s = df[df['loss'].notnull()]
            plt.figure()
            plt.plot(s['step'], s['loss'], marker='o')
            plt.xlabel('step'); plt.ylabel('loss'); plt.title('DAPT Training Loss')
            plt.savefig(os.path.join(OUTPUT_DIR, 'dapt_training_loss.png'))
    except Exception as e:
        logger.warning(f"Could not save DAPT log history or plot: {e}")

    perplexity = math.exp(metrics["train_loss"])
    logger.info(f"DAPT completed. Perplexity: {perplexity:.2f}")
    logger.info(f"Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()