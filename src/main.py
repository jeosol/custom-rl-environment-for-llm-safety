import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
from peft import LoraConfig
from data_utils import get_dpo_safety_dataset

max_prompt_length = 256

def filter_prompts(example):
    # Adjust 'prompt' to match the column name in your dataset
    return len(tokenizer.tokenize(example["prompt"])) < max_prompt_length

def run_dpo_alignment():
    model_id = "meta-llama/Llama-3.2-1B" # Highly optimized for Colab T4/A100 GPUs
    
    # 1. Initialize Tokenizer and Quantized Model for Memory Efficiency
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # We load the base policy and reference model simultaneously 
    # (TRL handles copying the reference weights automatically to save space)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    # 2. Mocking the Sycophancy/Safety Dataset 
    # In DPO, each sample needs: a prompt, a chosen (safe) answer, and a rejected (unsafe) answer.
    # We will load a public preference dataset or use your custom structured environment data.
    # dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]") # Example Anthropic dataset

    #
    # train_dataset, eval_dataset = get_dpo_safety_dataset()

    # use synthetic data for test
    dataset = load_dataset("json", data_files="synthetic_safety_dpo.jsonl", split="train")

    # filter dataset
    dataset = dataset.filter(filter_prompts)
    
    # 3. Configure Parameter-Efficient Fine-Tuning (LoRA)
    # This ensures we only update ~1-2% of the parameters, matching your ML systems mindset
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Configure DPO Hyperparameters
    # 'beta' is the implicit KL-divergence penalty weight. If beta is high, 
    # the model is penalized heavily for drifting away from the original Llama weights.
    training_args = DPOConfig(
        output_dir="./dpo_safety_results",
        beta=0.1, 
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        max_length=512,
        # max_prompt_length=256, # this is now deprecated
        remove_unused_columns=False,
    )

    # 5. Initialize and Run Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # Passing None makes TRL automatically optimize memory via peft reference hooks
        args=training_args,
        beta=training_args.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        #eval_dataset=eval_dataset
    )

    print("Starting DPO Optimization Loop...")
    trainer.train()

if __name__ == "__main__":
    run_dpo_alignment()
