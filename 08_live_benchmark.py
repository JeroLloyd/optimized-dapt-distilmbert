import os
import time
import torch
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

# --- CONFIGURATION ---
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
    print(f"{'MODEL':<20} | {'PREDICTION':<10} | {'CONFIDENCE':<10} | {'LATENCY (ms)':<15} | {'SIZE (MB)':<10}")
    print(f"{'-'*100}")

    for name, artifacts in models.items():
        model = artifacts["model"]
        tokenizer = artifacts["tokenizer"]
        size = artifacts["size_mb"]
        
        # 1. Prepare Input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        # --- CRITICAL FIX: Remove token_type_ids for DistilBERT ---
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        # ----------------------------------------------------------
        
        # 2. Measure Inference Speed (Latency)
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        # 3. Process Output
        logits = outputs.logits[0] if hasattr(outputs, "logits") else outputs[0]
        if isinstance(logits, np.ndarray): logits = torch.tensor(logits) 
            
        probs = torch.nn.functional.softmax(logits, dim=-1).tolist()
        winner_idx = np.argmax(probs)
        prediction = LABELS[winner_idx]
        confidence = probs[winner_idx] * 100
        
        print(f"{name:<20} | {prediction:<10} | {confidence:>8.1f}%  | {latency_ms:>10.2f} ms   | {size:>8.1f} MB")
        
        results.append({
            "Model": name,
            "Prediction": prediction,
            "Confidence": confidence,
            "Latency": latency_ms,
            "Size": size,
            "Probs": probs
        })

    return results

def visualize(results, text):
    # Setup Figure with smaller size
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Main Title (Reduced Font)
    fig.suptitle(f"Thesis Live Benchmark\nInput: \"{text}\"", fontsize=12, fontweight='bold')

    # --- Chart 1: Confidence (Bar) ---
    ax1 = fig.add_subplot(gs[0, :]) 
    model_names = [r["Model"] for r in results]
    confidences = [r["Confidence"] for r in results]
    predictions = [r["Prediction"] for r in results]
    colors = [COLORS[LABELS.index(p)] for p in predictions]
    
    bars = ax1.bar(model_names, confidences, color=colors, edgecolor='black', width=0.6)
    ax1.set_title("Prediction Confidence (%) - Higher is Better", fontsize=10)
    ax1.set_ylim(0, 115) # Extra space for labels
    ax1.set_ylabel("Confidence Score", fontsize=9)
    ax1.tick_params(axis='both', which='major', labelsize=8) # Smaller ticks
    
    # Add labels on bars (Smaller Font)
    for bar, pred in zip(bars, predictions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 2, 
                 f"{pred}\n{height:.1f}%", ha='center', va='bottom', 
                 fontsize=9, fontweight='bold')

    # --- Chart 2: Latency (Horizontal Bar) ---
    ax2 = fig.add_subplot(gs[1, 0])
    latencies = [r["Latency"] for r in results]
    y_pos = np.arange(len(model_names))
    
    ax2.barh(y_pos, latencies, color='salmon', align='center', height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_names, fontsize=8)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('Milliseconds (ms)', fontsize=9)
    ax2.set_title('Inference Speed (Latency) - Lower is Faster', fontsize=10)

    # Label values inside bars
    for i, v in enumerate(latencies):
        ax2.text(v + 1, i, f"{v:.1f} ms", va='center', fontsize=8)

    # --- Chart 3: Model Size (Bar) ---
    ax3 = fig.add_subplot(gs[1, 1])
    sizes = [r["Size"] for r in results]
    
    bars3 = ax3.bar(model_names, sizes, color='skyblue', width=0.6)
    ax3.set_title('Resource Usage (Storage) - Lower is Better', fontsize=10)
    ax3.set_ylabel('Megabytes (MB)', fontsize=9)
    ax3.tick_params(axis='both', labelsize=8)
    
    # Label values
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 5, 
                 f"{height:.0f} MB", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    loaded_models = load_models_safely()
    
    if not loaded_models:
        print(" CRITICAL: No models found. Run previous training scripts first.")
        exit()

    try:
        while True:
            print("\n" + "-"*30)
            user_text = input(" Enter Taglish Sentence (or 'q' to quit): ")
            
            if user_text.lower() in ['q', 'quit', 'exit']:
                print(" Exiting Benchmark Suite.")
                break
                
            if user_text.strip():
                data = run_live_test(loaded_models, user_text)
                print(" Generating Visualization...")
                visualize(data, user_text)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")