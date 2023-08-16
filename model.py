import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define a custom PyTorch model that uses the GPT-2 architecture
# TimeLapsesTasksAssignment
class TLTA(nn.Module):
    def __init__(self, gpt2_model, tokenizer):
        super(TLTA, self).__init__()
        self.gpt2_model = gpt2_model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None):
            # Get the maximum sequence length from input_ids
            max_len = input_ids.size(1)
            
            # Generate multiple results by autoregressively decoding
            generated_token_ids_list = []
            for i in range(max_len):
                output = self.gpt2_model(input_ids, attention_mask=attention_mask)
                # generated_token_ids = output.logits.argmax(dim=-1).squeeze()
                generated_token_ids_list.append(output)
            
            # Stack the list of tensors into a single tensor
            stacked_generated_token_ids = torch.stack(generated_token_ids_list, dim=1).to("cuda")
            
            return stacked_generated_token_ids





