"""
scripts/download_model.py
=========================
Pre-downloads the lightweight ONNX embedding model (~118 MB) and
SentencePiece tokenizer (~5 MB) during the application build phase.
Ensures zero runtime download delay and instant server startup on Render Free.
"""

import sys
from huggingface_hub import hf_hub_download

def main():
    print("[1/2] Pre-downloading SentencePiece tokenizer (~5 MB)...")
    sp_path = hf_hub_download(
        repo_id="intfloat/multilingual-e5-small",
        filename="onnx/sentencepiece.bpe.model"
    )
    print(f"      Cached at: {sp_path}")

    print("[2/2] Pre-downloading Quantized INT8 ONNX model (~118 MB)...")
    model_path = hf_hub_download(
        repo_id="Xenova/multilingual-e5-small",
        filename="onnx/model_quantized.onnx"
    )
    print(f"      Cached at: {model_path}")

    print("\n[OK] All embedding assets successfully pre-cached for zero-latency deployment.")

if __name__ == "__main__":
    main()
