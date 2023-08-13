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
model.load_state_dict(torch.load("model_state.pt", map_location=torch.device('cpu')))
model.eval()

# Example usage
input_entry = {
    "prompt": "Prepare presentation for tomorrow's conference",
}

def infer(entry):
    prompt = entry["prompt"]
    input_text = f"<startofstring><startofprompt>{prompt}<endofpromt><endofstring>"
    input_encoded = tokenizer(input_text, max_length=500, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = input_encoded['input_ids']
    attention_mask = input_encoded['attention_mask']
    output = model.generate(input_ids, attention_mask=attention_mask)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)

    return generated_text

if __name__ == "__main__":
    response = infer(input_entry)
    print(response)
