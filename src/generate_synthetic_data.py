import json
import torch
from transformers import pipeline

def build_generation_pipeline():
    """Initializes a lightweight, un-gated instruct model for generation."""
    print("Initializing free generator model (Qwen-2.5)...")
    # Qwen-2.5-1.5B-Instruct does not require repository approval and fits in 4GB VRAM
    generator = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        dtype=torch.float16,
        device_map="auto"
    )
    return generator

def generate_safety_triplet(generator, topic):
    """Uses targeted prompting to generate a single DPO training sample."""
    
    # Context prompt instructing the model to behave as a data-generation agent
    system_prompt = (
        "You are an expert AI safety data generator. Generate data exactly in JSON format. "
        "Do not include any introductory or concluding text. Output raw JSON only."
    )
    
    user_prompt = f"""
    Create a dataset entry for testing AI safety regarding 'Sycophancy' (when an AI agrees with a user's wrong/biased opinion instead of being truthful).
    Topic area: {topic}
    
    Output a JSON object with exactly three fields:
    1. "prompt": A user prompt where the human introduces a clear bias, incorrect statement, or controversial premise, pushing the AI to agree with them.
    2. "chosen": A safe, helpful, polite, and objective response that corrects the user or remains neutral, refusing to blindly agree.
    3. "rejected": A sycophantic response that blindly praises the user's biased or incorrect perspective just to be agreeable.
    
    JSON Template:
    {{
      "prompt": "Human: ... \\n\\nAssistant: ",
      "chosen": "...",
      "rejected": "..."
    }}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Run inference
    outputs = generator(messages, max_new_tokens=512, temperature=0.7, do_sample=True)
    raw_response = outputs[0]["generated_text"][-1]["content"]
    
    try:
        # Parse output to ensure it matches strict JSON format requirements
        data_triplet = json.loads(raw_response.strip())
        # Basic schema verification
        if all(k in data_triplet for k in ("prompt", "chosen", "rejected")):
            return data_triplet
    except Exception:
        # Fallback if the LLM output is malformed or includes extra conversational text
        pass
    return None

def run_synthetic_pipeline():
    generator = build_generation_pipeline()
    
    # A list of diverse safety seeds to feed our pipeline
    seed_topics = [
        "Flat earth arguments",
        "Biased political opinions",
        "Flawed mathematical logic",
        "Conspiracy theories about history",
        "Ineffective medical home remedies"
    ]
    
    synthetic_dataset = []
    print(f"Starting pipeline run across {len(seed_topics)} seed topics...")
    
    for topic in seed_topics:
        print(f"Generating samples for topic: {topic}")
        # Try generating until we get a valid JSON string back
        for _ in range(3): 
            sample = generate_safety_triplet(generator, topic)
            if sample:
                synthetic_dataset.append(sample)
                break
                
    # Save the synthetic artifact to a standard JSONL file
    output_file = "synthetic_safety_dpo.jsonl"
    with open(output_file, "w") as f:
        for item in synthetic_dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"Pipeline complete. Successfully generated {len(synthetic_dataset)} safe records in '{output_file}'.")

if __name__ == "__main__":
    run_synthetic_pipeline()
