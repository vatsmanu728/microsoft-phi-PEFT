##### src/model.py

"""
Loads Phi-4 in 4-bit precision (QLoRA) and attaches trainable LoRA adapters on top of the frozen base weights,
using the quantization and LoRA settings defined in config.py.

Fix note: device_map is forced to {"": 0} instead of "auto".
On amemory-constrained GPU (e.g. free-tier T4, 15GB), "auto" can decide to offload some layers to CPU/disk —
which bitsandbytes 4-bit quantization does not support without extra opt-in flags, causing a ValueError.
Forcing everything onto GPU 0 avoids that split entirely.

REQUIRES A GPU — 4-bit quantization still needs CUDA.
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.config import get_config


def _build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    # Builds the 4-bit quantization config from config.py settings
    compute_dtype = getattr(torch, cfg["bnb_4bit_compute_dtype"])

    return BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _build_lora_config(cfg: dict) -> LoraConfig:
    # Builds the LoRA adapter config from config.py settings
    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias=cfg["lora_bias"],
        task_type=cfg["lora_task_type"],
    )


def load_lora_model():
    # Loads Phi-4 in 4-bit, prepares it for k-bit training, and attaches LoRA adapters.
    # Returns the resulting PEFT model, ready to pass to a Trainer.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. 4-bit quantized loading requires CUDA — "
            "check Runtime > Change runtime type > GPU in Colab."
        )

    cfg = get_config()
    bnb_config = _build_bnb_config(cfg)

    print(f"Loading base model: {cfg['base_model_id']} (this downloads ~8GB in 4-bit)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_id"],
        quantization_config=bnb_config,
        device_map={"": 0},          # fixed — was "auto". Now forces the entire quantized model onto GPU 0 rather than letting accelerate attempt a CPU/disk split that bitsandbytes 4-bit doesn't support.
        trust_remote_code=True,
        low_cpu_mem_usage=True,      # reduces host RAM spike during load
        attn_implementation="sdpa",  # flash-attn2 unsupported on Turing/T4; sdpa is the efficient option that works
    )


    # Required prep step before attaching LoRA to a k-bit (4-bit) model —
    # handles things like casting layer norms to fp32 for stability.
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=cfg["gradient_checkpointing"],
    )

    lora_config = _build_lora_config(cfg)
    model = get_peft_model(base_model, lora_config)

    return model


if __name__ == "__main__":
    # Quick manual check: `python -m src.model` (or `!python -m src.model` in Colab)

    model = load_lora_model()
    model.print_trainable_parameters()
