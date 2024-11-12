import torch
from transformers import AutoTokenizer
import numpy as np
from constant_tokens import * 


def mask_long_sequences(tensor, target_value=220, mask_value=-100, min_length=2):
    """
    Mask all sequences of the target_value in a tensor that are longer than or equal to min_length with the mask_value.

    Args:
    tensor (torch.Tensor): The input tensor.
    target_value (int): The value to look for long sequences.
    mask_value (int): The value to replace long sequences with.
    min_length (int): The minimum length of a sequence to be considered long.

    Returns:
    torch.Tensor: The tensor with long sequences masked.
    """
    # tensor = tensor.clone()  # Clone the tensor to avoid in-place modifications

    count = 0
    start_index = None

    for i in range(len(tensor)):
        if tensor[i] == target_value:
            if start_index is None:
                start_index = i
            count += 1
        else:
            if count >= min_length:
                tensor[start_index:i] = mask_value
            count = 0
            start_index = None
    
    # Check if the sequence at the end of the tensor needs to be masked
    if count >= min_length:
        tensor[start_index:] = mask_value

    return tensor

def flatten_to_tuples(flattened_list):
    """
    Convert a flattened list to a list of tuples.

    Parameters:
    - flattened_list (list): Input flattened list.

    Returns:
    - list of tuples: List containing tuples with 2 elements from the flattened list.
    """
    tuples_list = [(flattened_list[i], flattened_list[i + 1]) if i + 1 < len(flattened_list) else (flattened_list[i], None) for i in range(0, len(flattened_list), 2)]
    return tuples_list


def mask_values(tensor, start_index, end_index=None, mask=-100):
    """
    Set values between start_index and end_index (inclusive) to -100 in the given tensor.

    Parameters:
    - tensor (torch.Tensor): Input tensor.
    - start_index (int): Starting index.
    - end_index (int): Ending index.

    Returns:
    - torch.Tensor: Modified tensor.
    """
    #print("inside mask_values: ", start_index, end_index)
   
    if end_index:
        if start_index < 0 or end_index >= len(tensor) or start_index > end_index:
            raise ValueError("Invalid indices")

        tensor[start_index : end_index + 1] = mask
    else:
        print(start_index, end_index)
        tensor[start_index:] = mask
    return tensor


#TODO: fix bug for cases where there are 2 tags in the list and one of them is the first or last element in the list.
def process_labels(tokens, mask=-100):
    # Assuming translate_token, yor, eng, ibo, hau, and pcm are tensors
    translate_condition = (
       (tokens == translate_token) | (tokens == yor) | (tokens == eng) | (tokens == ibo) | (tokens == hau) | (tokens == pcm) | (tokens == ff) | (tokens == fuv) | (tokens == ful) | (tokens == urh) | (tokens == efik)
    )   
    language_condition = (
        (tokens == lang_id_label_token) | (tokens == lang_id_label_token2) | (tokens == lang_id_token )
    )
    classification_condition = (
        (tokens == classify_token) | (tokens == sentiment_token) | (tokens == topic_token)
    )
    qa_condition = (
        (tokens == qa_token) | (tokens == answer_token)
    )
    ner_condition = (
        (tokens == ner_token) | (tokens == ner_token2) | (tokens == tag_token)
    )
    diacritize_condition = (
        (tokens == diacritize_token) | (tokens  == correct_token)
    )
    clean_condition = (
        (tokens == clean_token) | (tokens == correct_token)
    )
    summarize_condition = (
        (tokens == summarize_token) | (tokens == summary_token)
    )
    title_condition = (
        (tokens == title_token) | (tokens == headline_token)
    )
    prompt_condition = (
        (tokens == prompt_token) | (tokens == response_token)
    )
    
    lang_tag = torch.where(language_condition, True, False)
    classify_tag = torch.where(classification_condition, True, False)
    qa_tag = torch.where(qa_condition, True, False)
    ner_tag = torch.where(ner_condition, True, False)
    diacritize_tag = torch.where(diacritize_condition, True, False)
    clean_tag = torch.where(clean_condition, True, False)
    summarize_tag = torch.where(summarize_condition, True, False)
    title_tag = torch.where(title_condition, True, False)
    translate_tag = torch.where(translate_condition, True, False)
    prompt_tag = torch.where(prompt_condition, True , False)
    
    indices_list = list(torch.where(((translate_tag)|
                                     (lang_tag)|
                                     (classify_tag) |
                                     (qa_tag)|
                                     (ner_tag) |
                                     (diacritize_tag) |
                                     (clean_tag) |
                                     (summarize_tag) |
                                     (title_tag) |
                                     (prompt_tag)
                                     ))[0].numpy())
    
    if len(indices_list) == 0:
        return tokens
    
    elif len(indices_list) == 1:
        # if tokens starts with any of the language tags:
        if int(tokens[indices_list[0]]) in action_tokens:
            tokens[:indices_list[0]+1] = mask
            indices_list.pop(0)
            
        else:
            tokens[indices_list[-1]:] = mask
            indices_list.pop()
            return tokens
        
    elif len(indices_list) % 2 == 0:
        # if tokens starts with any of the language tags:
        if int(tokens[indices_list[0]]) in action_tokens:
            # tokens[indices_list[0]:indices_list[1]] = mask
            tokens[:indices_list[0]+1] = mask
            tokens[indices_list[-1]:] = mask
            indices_list.pop(0)
            indices_list.pop(-1)
       
        indices_list = flatten_to_tuples(indices_list)
            
    else:
        # if tokens starts with any of the language tags:
        if int(tokens[indices_list[0]]) in action_tokens:
            tokens[:indices_list[0]+1] = mask
            indices_list.pop(0)
            
        else:
            tokens[indices_list[-1]:] = mask
            indices_list.pop()
       
        indices_list = flatten_to_tuples(indices_list)

    for each in indices_list:
        start, end = each
        tokens = mask_values(tokens,start_index=start, end_index=end)
    # tokens = mask_long_sequences(tokens, mask_value=mask)    
    # print("inside process_labels: ", tokens)
    return tokens


if __name__ == '__main__':    

    #case 1: some text generation followed by translate without an end_of_text at the end
    label1= [1,2,32,54,61,4,6,76,87,98,90,91,50,21,123,45,16,88,92,6]
    #case 2: some text generation followed by translate with an end_of_text at the end followed by a part of classification task
    label2= [1,2,32,54,61,4,6,76,87,98,90,91,50,21,123,45,16,88,92,4, 5, 90,128,69]
    #case 3
    label3= [12,15,77,88, 24, 100, 52, 34,4, 6]

    print(label1)
    print(process_labels(torch.Tensor(label1)))
    print()
    print(label2)
    print(process_labels(torch.Tensor(label2)))
    print()
    print(label3)
    print(process_labels(torch.Tensor(label3)))
    
