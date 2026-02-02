import os
import time
import torch
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# --- CONFIGURATION ---
# Filter warnings to keep terminal clean
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

MODELS = {
    "Model A (Base)": "./models/model_A_finetuned",
    "Model B (DAPT)": "./models/model_B_finetuned",
    "Model C (XLM-R)": "./models/model_C_finetuned",
    "Model D (Optimized)": "./models/model_D_optimized"
}
LABELS = ["Negative", "Positive", "Neutral"]
COLORS = ['#ff9999', '#99ff99', '#cccccc'] # Red, Green, Grey

def get_model_size(path):
    """Calculates folder size in MB, excluding checkpoint folders"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        if "checkpoint" in dirpath:
            continue
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

def load_models_safely():
    loaded = {}
    print(f"\n{'='*60}")
    print(f"  INITIALIZING THESIS BENCHMARK SUITE")
    print(f"{'='*60}")
    
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"  MISSING: {name} (Path: {path})")
            continue
            
        print(f"   Loading {name}...", end="\r")
        try:
            try:
                tokenizer = AutoTokenizer.from_pretrained(path, fix_mistral_regex=True)
            except TypeError:
                tokenizer = AutoTokenizer.from_pretrained(path)

            size_mb = get_model_size(path)
            
            if "Optimized" in name:
                onnx_files = [f for f in os.listdir(path) if f.endswith('.onnx')]
                onnx_name = onnx_files[0] if onnx_files else 'model_quantized.onnx'

                model = ORTModelForSequenceClassification.from_pretrained(
                    path, 
                    file_name=onnx_name, 
                    provider="CPUExecutionProvider"
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(path)
            
            loaded[name] = {
                "model": model, 
                "tokenizer": tokenizer,
                "size_mb": size_mb
            }
            print(f"  Loaded {name:<20} | Size: {size_mb:.0f} MB")
        except Exception as e:
            print(f"  Error loading {name}: {e}")
            
    print(f"{'='*60}\n")
    return loaded

def run_live_test(models, text):
    results = []
    
    print(f"  Analyzing Input: \"{text}\"")
    print(f"{'-'*100}")
    print(f"{'MODEL':<20} | {'PRED':<10} | {'CONF':<8} | {'LATENCY':<10} | {'SIZE':<10}")
    print(f"{'-'*100}")

    for name, artifacts in models.items():
        model = artifacts["model"]
        tokenizer = artifacts["tokenizer"]
        size = artifacts["size_mb"]
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        # Safe Input Handling
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        logits = outputs.logits[0] if hasattr(outputs, "logits") else outputs[0]
        if isinstance(logits, np.ndarray): logits = torch.tensor(logits) 
            
        probs = torch.nn.functional.softmax(logits, dim=-1).tolist()
        winner_idx = np.argmax(probs)
        prediction = LABELS[winner_idx]
        confidence = probs[winner_idx] * 100
        
        # Truncate model name for cleaner print
        short_name = name.replace("Model ", "").replace(" (Base)", "").replace(" (DAPT)", "").replace(" (XLM-R)", "").replace(" (Optimized)", "")
        
        print(f"{name:<20} | {prediction:<10} | {confidence:>6.1f}% | {latency_ms:>8.2f} ms | {size:>8.1f} MB")
        
        results.append({
            "Model": name,
            "ShortName": short_name, # Used for graphs to save space
            "Prediction": prediction,
            "Confidence": confidence,
            "Latency": latency_ms,
            "Size": size,
            "Probs": probs
        })

    return results

def visualize(results, text):
    # Set global font size for this plot
    plt.rcParams.update({'font.size': 8}) 
    
    # Create Figure with high DPI for sharpness
    fig = plt.figure(figsize=(10, 6), dpi=120) 
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
    # Title
    clean_text = (text[:40] + '..') if len(text) > 40 else text
    fig.suptitle(f"Thesis Benchmark: \"{clean_text}\"", fontsize=10, fontweight='bold')

    # Data Prep
    names = [r["ShortName"] for r in results]
    conf = [r["Confidence"] for r in results]
    preds = [r["Prediction"] for r in results]
    lat = [r["Latency"] for r in results]
    siz = [r["Size"] for r in results]
    colors = [COLORS[LABELS.index(p)] for p in preds]

    # --- 1. Confidence Bar Chart (Top) ---
    ax1 = fig.add_subplot(gs[0, :]) 
    bars = ax1.bar(names, conf, color=colors, edgecolor='black', width=0.5)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Confidence %")
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Labels inside bars
    for bar, pred, c in zip(bars, preds, conf):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 2, 
                 f"{pred}\n{c:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # --- 2. Latency (Bottom Left) ---
    ax2 = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(len(names))
    ax2.barh(y_pos, lat, color='salmon', align='center', height=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.invert_yaxis()
    ax2.set_xlabel("Time (ms)")
    ax2.set_title("Speed (Lower is Better)", fontsize=9)
    
    for i, v in enumerate(lat):
        ax2.text(v + 1, i, f"{v:.1f} ms", va='center', fontsize=7)

    # --- 3. Size (Bottom Right) ---
    ax3 = fig.add_subplot(gs[1, 1])
    bars3 = ax3.bar(names, siz, color='skyblue', width=0.5)
    ax3.set_ylabel("MB")
    ax3.set_title("Size (Lower is Better)", fontsize=9)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 5, 
                 f"{height:.0f} MB", ha='center', va='bottom', fontsize=7)

    # Magic command to prevent overlap
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    loaded_models = load_models_safely()
    
    if not loaded_models:
        print(" CRITICAL: No models found. Run previous training scripts first.")
        exit()

    try:
        while True:
            print("\n" + "-"*40)
            user_text = input(" Enter Taglish Sentence (or 'q'): ")
            
            if user_text.lower() in ['q', 'quit', 'exit']:
                break
                
            if user_text.strip():
                data = run_live_test(loaded_models, user_text)
                print(" Generating Visualization...")
                visualize(data, user_text)
    except KeyboardInterrupt:
        print("\nExiting.")