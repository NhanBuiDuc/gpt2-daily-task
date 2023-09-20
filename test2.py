from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Tạo một tokenizer cho mô hình GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Chuỗi cần mã hóa
input_text = "<time>07:30:24</time>"
custom_tokens = [
    "<time>", "</time>"
]

special_tokens = {
    "pad_token": "<pad>",
    "bos_token": "<sot>",
    "eos_token": "<eot>",
    "additional_special_tokens": custom_tokens
}

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens(special_tokens)

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Mã hóa chuỗi thành các token
tokens = tokenizer.tokenize(input_text)

# In ra ID của từng token
for token in tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"Token: {token}, Token ID: {token_id}")
