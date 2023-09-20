import json
import tensorflow as tf


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


x = make_data_2('dataa.json')
for i in x:
    print(i, '\n')
