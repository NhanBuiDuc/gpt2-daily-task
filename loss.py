import torch
import torch.nn as nn

# TaskEfficientArrangingLoss
class TEALoss(nn.Module):
    def __init__(self):
        super(TEALoss, self).__init__()

    def forward(self, original_input, model_output):
        # Extract time and task information from the model output
        model_times = model_output['times']
        model_tasks = model_output['tasks']
        
        # Initialize loss to zero
        loss = torch.tensor(0.0)
        
        # Calculate overlap loss
        for i in range(len(model_times)):
            for j in range(i + 1, len(model_times)):
                overlap = max(0, min(model_times[i][1], model_times[j][1]) -
                                    max(model_times[i][0], model_times[j][0]))
                if overlap > 0:
                    loss += overlap
        
        return loss
