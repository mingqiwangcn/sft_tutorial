"""
Download model and tokenizer into the Hugging Face cache.

Usage:
    python scripts/download_model.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.model_cfg import MODEL_NAME


def main():
    print(f"Downloading tokenizer: {MODEL_NAME}")
    AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    print(f"Downloading model: {MODEL_NAME}")
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    print("Download complete.")


if __name__ == "__main__":
    main()