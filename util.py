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

def create_data_dict(jsonfile):
    with open(jsonfile, 'r') as file:
        data = json.load(file)
        target_texts = [entry.get('target_text', '') for entry in data]
        input_texts = [entry.get('input_text', '') for entry in data]
        diction = []
        for x in range(0, len(input_texts)-15):
            k = 0
            sm = {}
            sm[k] = input_texts[x]
            for y in range(x+1, x+15):
                sm[k+1] = target_texts[y]
                k += 1
            diction.append(sm)
    return diction

def divide_data_dict(jsonfile):
    with open(jsonfile, 'r') as file:
        data = json.load(file)
        target_texts = [entry.get('target_text', '') for entry in data]
    return 0

            
def create_task_core(jsonfile):
    lst = []
    i = 0
    with open(jsonfile, 'r') as file:
        data = json.load(file)
        
        target_texts = [entry.get('target_text', '') for entry in data]
        for input_str in target_texts: 
            start_pos = input_str.find("<sum>")
            end_pos = input_str.find("<totd>")
            
            if start_pos != -1 and end_pos != -1:
                lst.append(input_str[(start_pos + 5):end_pos])
    return lst

def check_time(x, y):
    session = ['morning', 'afternoon', 'evenning', 'noon', 'tonight']
    time = ['am', 'pm', "o'clock"]
    day = ['tommorow', 'next day']
    count_session = 
    return 0

def intra_data(jsonfile):
    with open (jsonfile, 'r') as file:
        data = json.load(file)
        input_texts = [entry.get('input_text', '') for entry in data]
        target_texts = [entry.get('target_text', '') for entry in data]
        for i in range(0, len(input_texts)):
            input_texts[i] = 
    return 0




# custom_tokens = [
#     "<startofprompt>", "<startoftask>",
#     "<endofpromt>", "<endoftask>", 
#     '<sum>', '<cate>', '<prio>', '<diff>', '<imp>', '<status>', '<exp_min>', 
#     '<totd>', '<spec_time>', '<dow>', '<day>', '<month>', '<no_date>', '<no_week>', '<no_month>'
# ]

# special_tokens = {
#     "pad_token": "<pad>",
#     "bos_token": "<sot>",
#     "eos_token": "<eot>",
#     "additional_special_tokens": custom_tokens
# }

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.add_special_tokens(special_tokens)

# model = GPT2LMHeadModel.from_pretrained("gpt2")
# model.resize_token_embeddings(len(tokenizer))
# x = make_data_2('dataa.json')
# for i in x:
#     print(i, '\n')

string = "<bs><bp>Remind me to meditate for 10 minutes every morning.<ep><betl><bt><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null'<et><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null'<et><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null'<et><id>2<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null<et><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null<et><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null'<et><eetl><es>"

        # <begin>
        #     <prompt>Remind me to meditate for 10 minutes every morning.</prompt>
        #     <task_list>
        #         <task><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null</task>
        #         <task><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null</task>
        #         <task><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null</task>
        #         <task><id>2<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null</task>
        #         <task><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null</task>
        #         <task><id>1<sum>Watering plants<totd>2<spec_time>08:00:00<prio>5<status>0<cate>daily<diff>3<imp>4<exp_min>15<dow>null<day>null<month>null<no_date>null<no_week>null<no_month>null</task>
        #     </task_list>
        # <end>

