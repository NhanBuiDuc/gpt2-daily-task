import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer as transformers_trainer, TrainingArguments
from loss import TEALoss
from dataset import DailyTaskDataset
import tqdm
import json
def special_tokens_from_annotation():
    # Load the JSON file
    with open('token_annotation.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Initialize an empty list to store the "value" strings
    token_values = []

    # Iterate through the levels and tokens
    for level_key, level_data in data.items():
        if level_key == 'level_3':
            for sub_level_key, sub_level_data in level_data.items():
                if sub_level_key == "3.1":
                    for token_info in sub_level_data["token"]:
                        token_values.append(token_info["value"])
                if sub_level_key == "3.2":
                    for token_info in sub_level_data["token"]:
                        token_values.append(token_info["value"])
                if sub_level_key == "3.3":
                    for token_info in sub_level_data["token"]:
                        token_values.append(token_info["value"])
        else:
            for token_info in level_data["token"]:
                token_values.append(token_info["value"])

    # Print the list of "value" strings
    print(token_values)
    return(token_values)

class Trainer:
    def __init__(self, batch_size=1, learning_rate=1e-4):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_texts = [
            "<sum>Prepare presentation for 18 this month<cate>Work<prio>3<diff>4<imp>4<status>0<exp_min>120<totd>2<spec_time>null<dow>null<day>18<month>null<no_date>null<no_week>null<no_month>0",
            "<sum>Finish coding assignment before tommorow<cate>Study<prio>2<diff>1<imp>1<status>0<exp_min>180<totd>4<spec_time>null<dow>null<day>null<month>null<no_date>1<no_week>null<no_month>null",
            "<sum>Buy groceries in Thursday<cate>Errands<prio>5<diff>5<imp>3<status>0<exp_min>60<totd>3<spec_time>null<dow>5<day>null<month>null<no_date>null<no_week>null<no_month>null"
        ]

        self.custom_tokens = special_tokens_from_annotation()
        self.special_tokens = {
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "additional_special_tokens": self.custom_tokens
        }
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens(self.special_tokens)
        self.tokenizer = tokenizer

        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_model.resize_token_embeddings(len(self.tokenizer))
        self.train_dataset = DailyTaskDataset(self.target_texts, self.tokenizer)

    def train(self, num_epochs=10):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

            for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                input_ids = batch["input_ids"].to(device)
                input_ids = input_ids.squeeze(1)
                attention_mask = batch["attention_mask"].to(device)
                attention_mask = attention_mask.squeeze(1)
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.loss(input_ids, attention_mask, outputs)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            average_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")

        # Save the trained model
        torch.save(self.model.state_dict(), "model_state.pt")