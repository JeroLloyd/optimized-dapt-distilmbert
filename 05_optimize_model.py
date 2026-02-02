import argparse
import os
import torch
import onnx
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

from nlp_thesis.utils import get_logger, ensure_dir

# --- DEFAULT CONFIGURATION ---
DEFAULT_SOURCE_MODEL_PATH = "./models/model_B_finetuned"
DEFAULT_OPTIMIZED_OUTPUT_PATH = "./models/model_D_optimized"

logger = get_logger(__name__)

def run_optimization(source_path=DEFAULT_SOURCE_MODEL_PATH, optimized_path=DEFAULT_OPTIMIZED_OUTPUT_PATH):
    logger.info("\n" + "="*50)
    logger.info("STARTING STAGE 3: OPTIMIZATION (MODEL D)")
    logger.info("="*50)

    if not os.path.exists(source_path):
        logger.error(f"Source model (Model B) not found at {source_path}")
        return

    # 1. Export Model B to ONNX format
    logger.info("Exporting Model B to ONNX format...")
    model = ORTModelForSequenceClassification.from_pretrained(source_path, export=True)
    tokenizer = AutoTokenizer.from_pretrained(source_path)

    # 2. Apply Dynamic Quantization (INT8)
    logger.info("Applying dynamic quantization (INT8)...")
    quantizer = ORTQuantizer.from_pretrained(model)
    
    # Define quantization configuration (Arm/x86 optimized)
    dq_config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    
    ensure_dir(optimized_path)
    # Export the quantized model
    quantizer.quantize(
        save_dir=optimized_path,
        quantization_config=dq_config,
    )

    # 3. Save the tokenizer alongside the optimized model
    tokenizer.save_pretrained(optimized_path)
    
    logger.info(f"Model D (Optimized DAPT) generated at {optimized_path}")
    logger.info("Original Size: ~540MB | Optimized Size: ~135MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=DEFAULT_SOURCE_MODEL_PATH)
    parser.add_argument('--out', default=DEFAULT_OPTIMIZED_OUTPUT_PATH)
    args = parser.parse_args()
    run_optimization(args.source, args.out)