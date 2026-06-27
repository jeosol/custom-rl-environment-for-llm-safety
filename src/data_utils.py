from datasets import load_dataset, DownloadMode
import time

def preprocess_anthropic_hh(sample):
    """
    Anthropic's hh-rlhf dataset formats conversations as a single string:
    "Human: <prompt>\n\nAssistant: <response>"
    We need to extract the prompt, the chosen safe response, and the rejected response.
    """

    # Locate the last turn of the assistant to separate prompt from response
    assistant_marker = "\n\nAssistant: "

    if assistant_marker in chosen_str:
        # Split from the last occurrence of the assistant marker
        prompt = chosen_str.split(assistant_marker)[0] + assistant_marker # extract prompt + append assistant_marker
        chosen_response = chosen_str.split(assistant_marker)[-1]      # take assistant's last response
        rejected_response = rejected_str.split(assistant_marker)[-1]  # take assistant's last response
    else:
        # Fallback handling for malformed rows
        prompt = "Human: Treat all people with respect.\n\nAssistant: "
        chosen_response = "I agree."
        rejected_response = "I refuse."

    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response
    }

def get_dpo_safety_dataset(dataset_size=2000):
    print("Loading Anthropic Harmless Base Dataset...")
    # Loading the harmless subset specifically targets safety/refusal behaviors
    raw_dataset = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir="harmless-base",
        split="train",
        #download_mode=DownloadMode.FORCE_REDOWNLOAD,
        #verification_mode="no_checks" 
    )
    if dataset_size > len(raw_dataset):
        dataset_size = len(raw_dataset)
        
    # Full raw dataset size = 42,537
    # Take a small slice (e.g., 2,000 samples) to ensure fast iteration in Colab
    small_dataset = raw_dataset.select(range(dataset_size))
    
    # Map the preprocessing pipeline and remove the old unformatted columns
    formatted_dataset = small_dataset.map(
        preprocess_anthropic_hh,
        remove_columns=raw_dataset.column_names
    )
    
    # Split into train/validation for empirical tracking
    split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    return split_dataset["train"], split_dataset["test"]

# Verification test
if __name__ == "__main__":
    train_data, val_data = get_dpo_safety_dataset()
    print(f"Dataset ready. Train size: {len(train_data)}, Eval size: {len(val_data)}")
    print(f"Sample Entry:\n{train_data[0]}")
