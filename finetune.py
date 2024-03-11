from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
import torch
import warnings
from builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from moellava.train.train import preprocess_openchat
from llava.train.train import preprocess_multimodal, _add_speaker_and_signal, _tokenize_fn, _mask_targets, preprocess
from llava.mm_utils import tokenizer_image_token

from llava.train.train import LazySupervisedDataset, DataCollatorForSupervisedDataset
from llava.train.train import make_supervised_data_module

from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field

import pathlib

warnings.filterwarnings('ignore')


device = "cuda" if torch.cuda.is_available() else "cpu"



checkpoint = 'liuhaotian/llava-v1.6-mistral-7b'
print('Loading model...')

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path = checkpoint,
    model_base = None,
    model_name = get_model_name_from_path(checkpoint),
    load_4bit= True,
    device_map="auto",
    device=device,
)

# need import DataArguments
data_args = DataArguments(
    train_data_path = 'train_dataset.json',
    val_data_path = 'eval_dataset.json',
    is_multimodal = True,
    train_image_folder = 'training-images',
    val_image_folder = 'dev-images'
)

data_args.image_processor = image_processor


data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)


#LoRA
config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj"],
    lora_dropout = 0.05,
    bias="none"
)

print('LoRA process...')
model = get_peft_model(model, config)
print(model.print_trainable_parameters())

training_args = TrainingArguments(
    output_dir = f"LLaVA-Mistral-finetuned",
    learning_rate = 2e-4,
    fp16 = True,
    num_train_epochs = 1,
    warmup_steps = 50,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    gradient_accumulation_steps = 8,
    dataloader_pin_memory = False,
    save_total_limit = 3,
    evaluation_strategy ="steps",
    eval_steps=1000,
    save_strategy = "steps",
    save_steps = 100,
    max_steps = 1000,
    logging_steps = 20,
    remove_unused_columns = False,
    push_to_hub=False,
    label_names = ["labels"],
    load_best_model_at_end = False,
    report_to = "none",
    optim = "paged_adamw_8bit",
)

trainer = Trainer(
    model = model,
    args = training_args,
    **data_module
)

def train():
    if list(pathlib.Path("LLaVA-Mistral-finetuned").glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


if __name__ == '__main__':
    train()
