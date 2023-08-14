from torch.utils.data import Dataset
import json

class ChatData(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = json.load(open(path, "r"))

        self.X = []
        for i in self.data:
            for j in i['dialog']:
                self.X.append(j['text'])

        for idx, i in enumerate(self.X):
            try:
                self.X[idx] = "<startofstring> "+i+" <bot>: "+self.X[idx+1]+" <endofstring>"
            except:
                break

        self.X = self.X[:5000]
        
        print(self.X[0])

        self.X_encoded = tokenizer(self.X,max_length=40, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])
    

class DailyTaskData(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = json.load(open(path, "r"))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        prompt = entry['prompt']
        result_str = entry['result']
        
        # Add start and end of string tokens to the prompt and result strings
        prompt = "<startofstring><startofprompt>" + prompt + "<endofpromt><endofstring>"

        result_str = "<startofstring><startoftask>" + result_str + "<endoftask><endofstring>"
        input_encoded = self.tokenizer(prompt, max_length=100, truncation=True, padding="max_length", return_tensors="pt")
        output_encoded = self.tokenizer(result_str, max_length=100, truncation=True, padding="max_length", return_tensors="pt")

        input_ids = input_encoded['input_ids']
        attention_mask = input_encoded['attention_mask']
        labels = output_encoded['input_ids']

        return (input_ids, attention_mask, labels)

    def _parse_result_string(self, result_str):
        result_dict = {}
        parts = result_str.split("<")
        for part in parts[1:]:
            key, value = part.split(">")
            result_dict[key] = value
        return result_dict

    def _create_result_string(self, result_dict):
        result_str = ""
        for key, value in result_dict.items():
            result_str += f"<{key}>{value}"
        return result_str
    


class DailyTaskSequenceData(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = json.load(open(path, "r"))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        prompt = entry['prompt']
        result_str = entry['result']
        
        # Add start and end of string tokens to the prompt and result strings
        input = "<startofstring><startofprompt>" + prompt + "<endofpromt><endofstring>"
        result_str = "<startofstring><startofprompt>" + prompt + "<endofpromt><startoftask>" + result_str + "<endoftask><endofstring>"
        input_encoded = self.tokenizer(input, max_length=100, truncation=True, padding="max_length", return_tensors="pt")
        output_encoded = self.tokenizer(result_str, max_length=100, truncation=True, padding="max_length", return_tensors="pt")

        input_ids = input_encoded['input_ids']
        attention_mask = input_encoded['attention_mask']
        labels = output_encoded['input_ids']

        return (input_ids, attention_mask, labels)