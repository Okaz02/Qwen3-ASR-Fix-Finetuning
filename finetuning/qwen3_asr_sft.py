# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


# LLM の注意層と MLP だけに LoRA をかける。音声エンコーダにも q_proj 等の同名の層があるので、パスで絞る
LORA_TARGET = r"thinker\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


SAMPLING_RATE = 16000  # データセットが返す波形のサンプリング周波数


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prefix_texts = self.processor.apply_chat_template(
            [build_prefix_messages(f.get("prompt", ""), None) for f in features],
            add_generation_prompt=True, tokenize=False,
        )
        targets = [f["text"] for f in features]
        audios = [f["audio"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control


@dataclass
class TrainConfig:
    model_path: str = "Qwen/Qwen3-ASR-1.7B"
    output_dir: str = "./qwen3-asr-finetuning-out"

    batch_size: int = 32
    grad_acc: int = 4
    lr: float = 2e-5
    epochs: float = 1
    log_steps: int = 10
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.02

    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 5

    freeze_audio_tower: bool = False
    gradient_checkpointing: bool = False
    attn_implementation: str = "sdpa"

    lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    resume_from: str = ""
    resume: bool = False


def train(cfg: TrainConfig, train_dataset, eval_dataset=None):
    """train_dataset / eval_dataset は len() と [i] が使えるもの。
    [i] は {"audio": SAMPLING_RATE の1次元波形, "text": 正解テキスト, "prompt"(任意)} を返す。
    [i] は DataLoader のワーカーの中で呼ばれるので、音声の取得は学習と並行して進む。"""

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        cfg.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
        attn_implementation=cfg.attn_implementation,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    if cfg.freeze_audio_tower:
        model.thinker.audio_tower.requires_grad_(False)
    if cfg.lora:
        model = get_peft_model(model, LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=LORA_TARGET,
        ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[params] trainable {trainable:,} / total {total:,}")

    collator = DataCollatorForQwen3ASRFinetuning(processor=processor)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_acc,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        logging_steps=cfg.log_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        dataloader_num_workers=cfg.num_workers,
        dataloader_pin_memory=cfg.pin_memory,
        dataloader_persistent_workers=cfg.persistent_workers,
        dataloader_prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=cfg.save_steps,
        do_eval=eval_dataset is not None,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=cfg.model_path)],
    )

    resume_from = (cfg.resume_from or "").strip()
    if not resume_from and cfg.resume:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    if cfg.lora and trainer.args.process_index == 0:
        merged_dir = os.path.join(training_args.output_dir, "merged")
        model.merge_and_unload().save_pretrained(merged_dir)
        copy_required_hf_files_for_qwen_asr(cfg.model_path, merged_dir)
        print(f"[lora] merged model saved to {merged_dir}")

