import torch
import torch.nn as nn
from transformers import DataCollatorForLanguageModeling
# TaskEfficientArrangingLoss
class TEALoss(nn.Module):
    def __init__(self, tokenizer, model):
        super(TEALoss, self).__init__()
        
        self.tokenizer = tokenizer
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()  
        # Convert custom tokens to their corresponding token IDs
        self.time_token_id = torch.tensor(self.tokenizer.convert_tokens_to_ids("<time>"), dtype=torch.long).to("cuda")
        self.id_token_id = torch.tensor(self.tokenizer.convert_tokens_to_ids("<id>"), dtype=torch.long).to("cuda")


    def forward(self, input_ids, attention_mask, model_output):  # Assuming you have the generated token IDs
        
        # Calculate loss between the first token and "<time>" token
        first_token_logits = torch.tensor(model_output[0], dtype=torch.long).to("cuda")
        time_loss = self.loss_fn(first_token_logits, self.time_token_id)

        # Calculate loss between "<time>" token and "<id>" token (if present)
        id_loss = torch.tensor(0.0).to(time_loss.device)  # Initialize loss to zero
        if id_token_id in model_output:
            id_token_position = model_output.index(id_token_id)
            id_token_logits = first_token_logits[id_token_position].unsqueeze(0)
            id_token_labels = torch.tensor([id_token_id]).to(id_token_logits.device)
            id_loss = self.loss_fn(id_token_logits, id_token_labels)

        # Calculate total loss
        total_loss = time_loss + id_loss
        return total_loss