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
    """Calculates folder size in MB, excluding checkpoint folders to get true size"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        # Skip 'checkpoint' directories to avoid inflating size
        if "checkpoint" in dirpath:
            continue
            
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp): # Check existence to avoid race conditions
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
            # Try loading tokenizer with Mistral regex fix when available
            try:
                tokenizer = AutoTokenizer.from_pretrained(path, fix_mistral_regex=True)
            except TypeError:
                tokenizer = AutoTokenizer.from_pretrained(path)

            size_mb = get_model_size(path)
            
            # Load Architecture
            if "Optimized" in name:
                # Detect ONNX filename
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

        # Filter inputs to match model.forward signature to avoid unexpected kwargs
        import inspect
        try:
            model_forward = getattr(model, "forward", None)
            sig = inspect.signature(model_forward) if model_forward is not None else None
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()) if sig else False
            if not accepts_kwargs and sig:
                accepted_params = set(p for p in sig.parameters.keys() if p != 'self')
                filtered_inputs = {k: v for k, v in inputs.items() if k in accepted_params}
            else:
                filtered_inputs = inputs
        except Exception:
            filtered_inputs = inputs

        # Move tensors to model device when possible
        device = None
        try:
            if hasattr(model, 'parameters'):
                params = list(model.parameters())
                if params:
                    device = params[0].device
        except Exception:
            device = None

        if device is not None:
            for k, v in list(filtered_inputs.items()):
                if isinstance(v, torch.Tensor):
                    filtered_inputs[k] = v.to(device)

        # 2. Measure Inference Speed (Latency)
        start_time = time.perf_counter()
        with torch.no_grad():
            try:
                outputs = model(**filtered_inputs)
            except TypeError:
                # As a fallback, remove token_type_ids and retry
                if 'token_type_ids' in filtered_inputs:
                    filtered_inputs.pop('token_type_ids', None)
                    print(f"  Note: removed 'token_type_ids' for model {name} due to incompatible forward signature.")
                    outputs = model(**filtered_inputs)
                else:
                    raise
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        # 3. Process Output
        logits = outputs.logits[0] if hasattr(outputs, "logits") else outputs[0]
        if isinstance(logits, np.ndarray): logits = torch.tensor(logits) 
            
        probs = torch.nn.functional.softmax(logits, dim=-1).tolist()
        winner_idx = np.argmax(probs)
        prediction = LABELS[winner_idx]
        confidence = probs[winner_idx] * 100
        
        # 4. Print Row
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
    # Setup Figure
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2)
    fig.suptitle(f"Thesis Live Benchmark\nInput: \"{text}\"", fontsize=14, fontweight='bold')

    # Chart 1: Confidence Comparison (Bar)
    ax1 = fig.add_subplot(gs[0, :]) # Top row spans both columns
    
    model_names = [r["Model"] for r in results]
    confidences = [r["Confidence"] for r in results]
    predictions = [r["Prediction"] for r in results]
    colors = [COLORS[LABELS.index(p)] for p in predictions]
    
    bars = ax1.bar(model_names, confidences, color=colors, edgecolor='black')
    ax1.set_title("Prediction Confidence (%) - Higher is Better")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Confidence Score")
    
    # Add labels on bars
    for bar, pred in zip(bars, predictions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 1, 
                 f"{pred}\n{height:.1f}%", ha='center', va='bottom', fontweight='bold')

    # Chart 2: Latency (Horizontal Bar)
    ax2 = fig.add_subplot(gs[1, 0])
    latencies = [r["Latency"] for r in results]
    y_pos = np.arange(len(model_names))
    
    ax2.barh(y_pos, latencies, color='salmon', align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_names)
    ax2.invert_yaxis()  # Labels read top-to-bottom
    ax2.set_xlabel('Milliseconds (ms)')
    ax2.set_title('Inference Speed (Latency) - Lower is Faster')

    # Chart 3: Model Size (Bar)
    ax3 = fig.add_subplot(gs[1, 1])
    sizes = [r["Size"] for r in results]
    
    bars3 = ax3.bar(model_names, sizes, color='skyblue')
    ax3.set_title('Resource Usage (Storage) - Lower is Better')
    ax3.set_ylabel('Megabytes (MB)')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Load Everything
    loaded_models = load_models_safely()
    
    if not loaded_models:
        print(" CRITICAL: No models found. Run previous training scripts first.")
        exit()

    # 2. Main Loop
    try:
        while True:
            print("\n" + "-"*30)
            user_text = input(" Enter Taglish Sentence (or 'q' to quit): ")
            
            if user_text.lower() in ['q', 'quit', 'exit']:
                print(" Exiting Benchmark Suite.")
                break
                
            if user_text.strip():
                # Run Test
                data = run_live_test(loaded_models, user_text)
                
                # Show Graphs
                print(" Generating Visualization...")
                visualize(data, user_text)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")