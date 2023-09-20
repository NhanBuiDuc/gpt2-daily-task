import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def make_data_1(json_data, num=9):
    comp_arr = []
    with open(json_data, 'r') as file:
        data = json.load(file)
        for entry in data[:num]:
            # Get the 'target_text' value, or an empty string if it doesn't exist
            target_text = entry.get('target_text', '')
            # text_parts = target_text.split('<target_text>')
            comp_arr.append(target_text)
        for entry in data[num:num+1]:
            new_command = entry.get('input_text', '')
            comp_arr.append(new_command)
    return comp_arr


def make_data_2(json_data):
    tensor = []
    counter = 10
    with open(json_data, 'r') as file:
        data = json.load(file)
        target_texts = [entry.get('target_text', '') for entry in data]
        input_texts = [entry.get('input_text', '') for entry in data]
        for i in range(0, len(target_texts), 9):  # Group 9 "target_text" entries at a time
            group = target_texts[i:i+9]
            try:
                if counter < len(input_texts):  # Ensure there are input_texts left
                    # Append 1 "input_text" entry
                    group.append(input_texts[counter])
            except:
                group.append(len(input_texts)-input_texts[counter])
            tensor.append(group)
            counter += 10

    return tensor

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
x = make_data_2('dataa.json')
for i in x:
    print(i, '\n')

string = "<bs><bp>Remind me to meditate for 10 minutes every morning.<ep><betl><bt><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null'<et><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null'<et><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null'<et><id>2<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null<et><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null<et><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null'<et><eetl><es>"


