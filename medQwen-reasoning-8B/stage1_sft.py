from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Config
# MODEL = "unsloth/Qwen3-8B"
MODEL = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
SFT_DATASET = "FreedomIntelligence/medical-o1-reasoning-SFT"
OUTPUT_DIR = "medQwen3-reasoning-8B"

MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
LORA_RANK = 16
LORA_ALPHA = 32
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16
NUM_EPOCHS = 3
WARMUP_STEPS = 100,
LEARNING_RATE = 2e-4
LOGGING_STEPS = 25
OPTIM = "paged_adamw_8bit"
SAVE_STRATEGY="steps"
SAVE_STEPS=750
SAVE_TOTAL_LIMIT=2
SEED = 42
PACKING=False
DDP_FIND_UNUSED_PARAMETERS=False #Required for LoRA multi-GPU
FP16=True

### Load dataset
dataset = load_dataset(SFT_DATASET, "en")
train_dataset = dataset["train"]

### Formatting Function
def preprocess_dataset(example):
    assistant_response = f"""
<think>
{example["Complex_CoT"]}
</think>

<answer>
{example["Response"]}
</answer>"""
    return {
        "messages": [
            {"role": "user", "content": example["Question"]},
            {"role": "assistant", "content": assistant_response}
        ]
    }

train_dataset = train_dataset.map(preprocess_dataset)
train_dataset = train_dataset.remove_columns(["Question", "Complex_CoT", "Response"])

## Load Model and Lora Config

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = LOAD_IN_4BIT, 
    fast_inference = True,
    max_lora_rank = LORA_RANK,
    gpu_memory_utilization = 0.7,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = LORA_ALPHA, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", 
    random_state = 42,
)

## Chat template
chat_template = """
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '\n\n' %}
{{ content }}
    {%- elif message['role'] == 'assistant' %}
{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + eos_token %}
{{ content }}
    {%- endif %}
{%- endfor %}
"""

tokenizer.chat_template = chat_template
train_dataset = train_dataset.map(
    lambda example: {
        "text": tokenizer.apply_chat_template(example["messages"], tokenize=False)
    }
)

### SFFT Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    max_seq_length = MAX_SEQ_LENGTH,
    args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs = NUM_EPOCHS, 
        warmup_steps = WARMUP_STEPS,
        learning_rate = LEARNING_RATE,
        logging_steps = LOGGING_STEPS,
        fp16=FP16,
        ddp_find_unused_parameters=DDP_FIND_UNUSED_PARAMETERS,
        optim = OPTIM,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        seed = SEED,
        output_dir = OUTPUT_DIR,
        report_to = "wandb",
        packing=PACKING,
    )
)

trainer.train()