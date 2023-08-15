from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# Load custom vocabulary and tokens
custom_tokens = [
    "<startofprompt>", "<startoftask>", "<endofpromt>", "<endoftask>",
    "<sum>", "<totd>", "<spec_time>", "<prio>", "<status>", "<cate>",
    "<diff>", "<imp>", "<exp_min>", "<deadline>"
]

special_tokens = {
    "pad_token": "<pad>",
    "bos_token": "<startofstring>",
    "eos_token": "<endofstring>",
    "additional_special_tokens": custom_tokens
}

# Load tokenizer with custom vocabulary and tokens
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens(special_tokens)

# Load GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Load trained model state
model.load_state_dict(torch.load("./output/checkpoint-710/pytorch_model.bin", map_location=torch.device('cuda')))
model.eval()

# Example usage
input_entry = {
    "prompt": "Prepare presentation for tomorrow's conference",
}

def infer(entry, max_length):
    prompt = entry["prompt"]
    input_text = "<startofstring>" + prompt # Use the raw prompt string
    input_encoded = tokenizer(input_text, max_length=200, truncation=False, return_tensors="pt")
    input_ids = input_encoded['input_ids'].to(device)  # Move to the appropriate device
    attention_mask = input_encoded['attention_mask'].to(device)  # Move to the appropriate device
    
    # Generate text with temperature and top-k sampling
    temperature = 0.2  # Adjust this value
    top_k = 50  # Adjust this value
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)  # Skip special tokens
    
    return generated_text

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

input_entry = {
    "prompt": "Prepare presentation for tomorrow's conference",
}

max_length = 100  # Adjust this value

response = infer(input_entry, max_length)
print(response)

