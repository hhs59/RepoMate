from __future__ import annotations

import time
from pathlib import Path

from src.config import get_settings
from src.logging import get_logger

logger = get_logger(__name__)


def train(
    slug: str,
    epochs: int = 1,
    lr: float = 2e-4,
    batch_size: int = 2,
    max_seq_length: int = 2048,
) -> dict:
    from src.train.dataset import load_train_eval, to_dataset
    from src.train.qlora_config import load_model_for_training, get_lora_config

    settings = get_settings()
    adapter_dir = settings.data_dir / slug / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train_data, eval_data = load_train_eval(slug)
    if not train_data:
        raise ValueError(f"No training data for {slug}")

    train_ds, eval_ds = to_dataset(train_data, eval_data)

    model, tokenizer = load_model_for_training()
    lora_config = get_lora_config()

    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        output_dir=str(adapter_dir / "checkpoints"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=epochs,
        bf16=True,
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        seed=42,
        max_seq_length=max_seq_length,
        packing=True,
        dataset_text_field="text",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if eval_ds else None,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    t0 = time.perf_counter()
    logger.info("Starting QLoRA training: %d examples, %d epochs", len(train_data), epochs)
    trainer.train()
    elapsed = time.perf_counter() - t0

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Saved adapter to %s", adapter_dir)

    final_loss = trainer.state.log_history[-1].get("train_loss", 0.0) if trainer.state.log_history else 0.0

    return {
        "examples": len(train_data),
        "epochs": epochs,
        "final_loss": final_loss,
        "elapsed_s": round(elapsed, 1),
        "adapter_path": str(adapter_dir),
    }
