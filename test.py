from transformers import GPT2Tokenizer

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Encode the time "10:00:00" using the tokenizer
time_text = "10:00:00"
time_token_ids = tokenizer.encode(time_text, add_special_tokens=False)

print("Token IDs for '10:00:00':", time_token_ids)
