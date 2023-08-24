from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import PromptResultMergedDataset
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
import tqdm
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
# Example usage
input_entry = {
    "prompt": "Prepare presentation for tomorrow's conference",
}


def train(dataset, model, max_length, temperature):
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=1000,
        per_device_train_batch_size=1,
        save_steps=10,
        save_total_limit=2,
        overwrite_output_dir=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    # Start training
    trainer.train()

    torch.save(model.state_dict(), "model_state.pt")
    print(infer(input_entry, max_length))


def infer(entry, max_length):
    prompt = entry["prompt"]
    input_encoded = tokenizer(prompt, max_length=100, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = input_encoded['input_ids'].to("cuda")
    attention_mask = input_encoded['attention_mask'].to("cuda")
    
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
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    return generated_text

device = "cuda" if torch.cuda.is_available() else "cpu"
custom_tokens = [
    "<startofprompt>", "<startoftask>",
    "<endofpromt>", "<endoftask>", 
    '<sum>', '<cate>', '<prio>', '<diff>', '<imp>', '<status>', '<exp_min>', 
    '<totd>', '<spec_time>', '<dow>', '<day>', '<month>', '<no_date>', '<no_week>', '<no_month>'
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

model = model.to(device)

dailyTaskDataset = PromptResultMergedDataset("./daily_task_data.json", tokenizer)
dailyTaskDataLoader = DataLoader(dailyTaskDataset, batch_size=64)

model.train()

optim = Adam(model.parameters(), lr=1e-2)

print("training .... ")
temperature = 0.2
max_length = 50
train(dailyTaskDataset, model, max_length, temperature)
