from __future__ import annotations

from src.config import get_base_model_id, get_device
from src.logging import get_logger

logger = get_logger(__name__)

_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def get_bnb_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=_LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def load_model_for_training():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = get_base_model_id()
    device = get_device()

    logger.info("Loading %s in 4-bit QLoRA config on %s", model_id, device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=get_bnb_config() if device != "cpu" else None,
        device_map="auto" if device != "cpu" else None,
        torch_dtype=dtype,
    )

    try:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    except Exception as e:
        logger.warning("prepare_model_for_kbit_training failed: %s", e)

    return model, tokenizer
