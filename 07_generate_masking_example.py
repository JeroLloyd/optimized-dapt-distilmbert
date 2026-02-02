import os, sys
import argparse
try:
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling
except Exception:
    print("Missing required package 'transformers' or its dependencies in the active environment.")
    print(f"Install into the active environment with:\n  {sys.executable} -m pip install -r requirements.txt")
    sys.exit(1)

from nlp_thesis.utils import get_logger, ensure_dir
logger = get_logger(__name__)
# Note: this script only generates the masking example figure and does not require GPU.
MODEL_CHECKPOINT = "distilbert-base-multilingual-cased"
TRAIN_FILE = "dapt_corpus_clean.txt"
MLM_PROBABILITY = 0.15


def generate_masking_figure_local():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CHECKPOINT)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY)

    sample_lines = []
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        for _ in range(3):
            l = f.readline().strip()
            if not l:
                break
            sample_lines.append(l)

    outdir = 'models/dapt-distilmbert'
    os.makedirs(outdir, exist_ok=True)

    # Import plotting lib
    import matplotlib.pyplot as plt

    if not sample_lines:
        logger.warning('No sample lines found in TRAIN_FILE; skipping masking example generation.')
        return

    enc = tokenizer(sample_lines, return_tensors='pt', padding=True, truncation=True)
    batch = [{k: enc[k][i].tolist() for k in enc.keys()} for i in range(len(sample_lines))]
    collated = data_collator(batch)

    input_ids = collated['input_ids'].cpu().numpy() if hasattr(collated['input_ids'], 'cpu') else collated['input_ids']
    labels = collated['labels'].cpu().numpy() if hasattr(collated['labels'], 'cpu') else collated['labels']

    fig, axs = plt.subplots(len(sample_lines), 1, figsize=(14, 2 * max(1, len(sample_lines))))
    if len(sample_lines) == 1:
        axs = [axs]

    for i, txt in enumerate(sample_lines):
        orig_ids = enc['input_ids'][i].numpy()
        input_row = input_ids[i]
        label_row = labels[i]

        display_tokens = []

        for j, (orig_id, inp_id, lbl_id) in enumerate(zip(orig_ids, input_row, label_row)):
            # Ensure scalar ids are converted safely (convert_ids_to_tokens expects iterable)
            token_str = tokenizer.convert_ids_to_tokens([int(orig_id)])[0]
            if lbl_id == -100:
                display_tokens.append(token_str)
            else:
                # masked position
                if int(inp_id) == tokenizer.mask_token_id:
                    display_tokens.append('[MASK]')
                elif int(inp_id) == int(lbl_id):
                    # unchanged
                    display_tokens.append(tokenizer.convert_ids_to_tokens([int(inp_id)])[0])
                else:
                    # replaced with random
                    display_tokens.append(tokenizer.convert_ids_to_tokens([int(inp_id)])[0])

        axs[i].axis('off')
        axs[i].set_title(txt[:120] + ('...' if len(txt)>120 else ''))
        axs[i].text(0, 0.5, ' '.join(display_tokens), fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(outdir, 'figure3_masking_example.png')
    ensure_dir(outdir)
    plt.savefig(outpath)
    logger.info(f"Masking example saved to {outpath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=MODEL_CHECKPOINT)
    parser.add_argument('--train-file', default=TRAIN_FILE)
    args = parser.parse_args()
    MODEL_CHECKPOINT = args.checkpoint
    TRAIN_FILE = args.train_file
    generate_masking_figure_local()
