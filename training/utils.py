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
    tuples_list = [
        (
            (flattened_list[i], flattened_list[i + 1])
            if i + 1 < len(flattened_list)
            else (flattened_list[i], None)
        )
        for i in range(0, len(flattened_list), 2)
    ]
    return tuples_list

def mask_values(tensor, start_index, end_index=None, mask=-100):
    """
    Optimized version with better error handling and edge case management.
    
    Parameters:
        tensor (torch.Tensor): Input tensor.
        start_index (int): Starting index.
        end_index (int): Ending index (optional).
        mask (int): Mask value.
    
    Returns:
        torch.Tensor: Modified tensor.
    """
    if start_index < 0 or start_index >= len(tensor):
        return tensor
    
    if end_index is not None:
        if end_index < start_index or end_index >= len(tensor):
            return tensor
        tensor[start_index:end_index + 1] = mask
    else:
        tensor[start_index:] = mask
    
    return tensor


def mask_long_sequences_optimized(tensor, target_value=220, mask_value=-100, min_length=2):
    """
    Optimized version: Mask all sequences of the target_value that are >= min_length.
    Uses vectorized operations instead of explicit loops.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        target_value (int): The value to look for long sequences.
        mask_value (int): The value to replace long sequences with.
        min_length (int): The minimum length of a sequence to be considered long.
    
    Returns:
        torch.Tensor: The tensor with long sequences masked.
    """
    if len(tensor) == 0:
        return tensor
    
    # Find positions where target_value occurs
    target_mask = (tensor == target_value)
    
    if not target_mask.any():
        return tensor
    
    # Find sequence boundaries using diff
    # Add padding to handle edge cases
    padded_mask = torch.cat([torch.tensor([False], device=tensor.device), 
                           target_mask, 
                           torch.tensor([False], device=tensor.device)])
    
    # Find start and end positions of sequences
    diff = padded_mask[1:].int() - padded_mask[:-1].int()
    starts = torch.where(diff == 1)[0]  # Positions where sequences start
    ends = torch.where(diff == -1)[0]   # Positions where sequences end
    
    # Calculate sequence lengths
    lengths = ends - starts
    
    # Find sequences that are too long
    long_sequences = lengths >= min_length
    
    if not long_sequences.any():
        return tensor
    
    # Create a mask for positions to be masked
    result = tensor.clone()
    
    # Vectorized masking using advanced indexing
    for start, end in zip(starts[long_sequences], ends[long_sequences]):
        result[start:end] = mask_value
    
    return result

def create_token_conditions():
    """
    Pre-compute token condition tensors for better performance.
    This avoids repeated tensor comparisons.
    """
    # Group tokens by category for efficient lookup
    token_groups = {
        'prompting_tokens': torch.tensor(prompting_tokens),
        'action_tokens': torch.tensor(action_tokens)
    }
    
    return token_groups


def find_tag_indices(tokens, token_groups):
    """
    Vectorized approach to find all tag indices at once.
    
    Args:
        tokens (torch.Tensor): Input token tensor
        token_groups (dict): Pre-computed token groups
    
    Returns:
        torch.Tensor: Indices where any tag tokens appear
    """
    # Combine all tokens into a single tensor
    all_tag_tokens = torch.cat(list(token_groups.values()))
    
    # Create a mask for all tag positions
    # Use broadcasting to compare tokens against all tag tokens at once
    tag_mask = (tokens.unsqueeze(1) == all_tag_tokens.unsqueeze(0)).any(dim=1)
    
    # Get indices
    indices = torch.where(tag_mask)[0]
    
    return indices

def process_labels(tokens, mask=-100):
    # Assuming translate_token, yor, eng, ibo, hau, and pcm are tensors
    translate_condition = (
        (tokens == translate_token)
        | (tokens == yor)
        | (tokens == eng)
        | (tokens == ibo)
        | (tokens == hau)
        | (tokens == pcm)
        | (tokens == ff)
        | (tokens == fuv)
        | (tokens == ful)
        | (tokens == urh)
        | (tokens == efik)
    )
    language_condition = (
        (tokens == lang_id_label_token)
        | (tokens == lang_id_label_token2)
        | (tokens == lang_id_token)
    )
    classification_condition = (
        (tokens == classify_token)
        | (tokens == sentiment_token)
        | (tokens == topic_token)
    )
    qa_condition = (tokens == qa_token) | (tokens == answer_token)
    ner_condition = (
        (tokens == ner_token) | (tokens == ner_token2) | (tokens == tag_token)
    )
    diacritize_condition = (tokens == diacritize_token) | (tokens == correct_token)
    clean_condition = (tokens == clean_token) | (tokens == correct_token)
    summarize_condition = (tokens == summarize_token) | (tokens == summary_token)
    title_condition = (tokens == title_token) | (tokens == headline_token)
    prompt_condition = (tokens == prompt_token) | (tokens == response_token)

    lang_tag = torch.where(language_condition, True, False)
    classify_tag = torch.where(classification_condition, True, False)
    qa_tag = torch.where(qa_condition, True, False)
    ner_tag = torch.where(ner_condition, True, False)
    diacritize_tag = torch.where(diacritize_condition, True, False)
    clean_tag = torch.where(clean_condition, True, False)
    summarize_tag = torch.where(summarize_condition, True, False)
    title_tag = torch.where(title_condition, True, False)
    translate_tag = torch.where(translate_condition, True, False)
    prompt_tag = torch.where(prompt_condition, True, False)

    indices_list = list(
        torch.where(
            (
                (translate_tag)
                | (lang_tag)
                | (classify_tag)
                | (qa_tag)
                | (ner_tag)
                | (diacritize_tag)
                | (clean_tag)
                | (summarize_tag)
                | (title_tag)
                | (prompt_tag)
            )
        )[0].numpy()
    )

    if len(indices_list) == 0:
        return tokens

    elif len(indices_list) == 1:
        # if tokens starts with any of the language tags:
        if int(tokens[indices_list[0]]) in action_tokens:
            tokens[: indices_list[0] + 1] = mask
            indices_list.pop(0)

        else:
            tokens[indices_list[-1] :] = mask
            indices_list.pop()
            return tokens

    elif len(indices_list) % 2 == 0:
        # if tokens starts with any of the language tags:
        if int(tokens[indices_list[0]]) in action_tokens:
            # tokens[indices_list[0]:indices_list[1]] = mask
            tokens[: indices_list[0] + 1] = mask
            tokens[indices_list[-1] :] = mask
            indices_list.pop(0)
            indices_list.pop(-1)

        indices_list = flatten_to_tuples(indices_list)

    else:
        # if tokens starts with any of the language tags:
        if int(tokens[indices_list[0]]) in action_tokens:
            tokens[: indices_list[0] + 1] = mask
            indices_list.pop(0)

        else:
            tokens[indices_list[-1] :] = mask
            indices_list.pop()

        indices_list = flatten_to_tuples(indices_list)

    for each in indices_list:
        start, end = each
        tokens = mask_values(tokens, start_index=start, end_index=end)
    # tokens = mask_long_sequences(tokens, mask_value=mask)
    # print("inside process_labels: ", tokens)
    return tokens


def process_labels_optimized(tokens, mask=-100):
    """
    Highly optimized version of process_labels with vectorized operations.
    
    Args:
        tokens (torch.Tensor): Input token tensor
        mask (int): Mask value
    
    Returns:
        torch.Tensor: Processed token tensor
    """
    if len(tokens) == 0:
        return tokens
    
    # Pre-compute token groups
    token_groups = create_token_conditions()
    
    # Find all tag indices vectorized
    indices = find_tag_indices(tokens, token_groups)
    
    if len(indices) == 0:
        return tokens
    
    # Convert to list for easier manipulation (only once)
    indices_list = indices.tolist()
    
    # Clone tensor to avoid in-place modifications
    result = tokens.clone()
    
    # Optimized logic for different cases
    num_indices = len(indices_list)
    
    if num_indices == 1:
        idx = indices_list[0]
        if tokens[idx].item() in action_tokens:
            result[:idx + 1] = mask
        else:
            result[idx:] = mask
        return result
    
    # Handle starting action token
    start_offset = 0
    if tokens[indices_list[0]].item() in action_tokens:
        result[:indices_list[0] + 1] = mask
        start_offset = 1
    
    # Handle ending token
    end_offset = 0
    remaining_indices = indices_list[start_offset:]
    if len(remaining_indices) % 2 == 1:  # Odd number remaining
        result[remaining_indices[-1]:] = mask
        end_offset = 1
    
    # Process pairs efficiently
    final_indices = remaining_indices[:len(remaining_indices) - end_offset]
    
    # Vectorized pair processing
    if len(final_indices) >= 2:
        # Convert to pairs using tensor operations
        pairs = torch.tensor(final_indices).view(-1, 2)
        
        # Apply masking for each pair
        for start_idx, end_idx in pairs:
            result[start_idx:end_idx + 1] = mask
    
    return result


def process_labels_ultra_optimized(tokens, mask=-100):
    """
    Ultra-optimized version using pure tensor operations where possible.
    
    Args:
        tokens (torch.Tensor): Input token tensor
        mask (int): Mask value
    
    Returns:
        torch.Tensor: Processed token tensor
    """
    if len(tokens) == 0:
        return tokens
    
    # Create all tag tokens tensor once
    all_tags = torch.tensor(prompting_tokens + action_tokens , device=tokens.device)
    
    # Remove duplicates
    all_tags = torch.unique(all_tags)
    
    # Find tag positions using vectorized comparison
    is_tag = torch.isin(tokens, all_tags)
    tag_indices = torch.where(is_tag)[0]
    
    if len(tag_indices) == 0:
        return tokens
    
    result = tokens.clone()
    
    # Convert action_tokens to tensor for efficient comparison
    action_tokens_tensor = torch.tensor(action_tokens, device=tokens.device)
    
    # Process based on number of indices
    if len(tag_indices) == 1:
        idx = tag_indices[0]
        if torch.isin(tokens[idx], action_tokens_tensor):
            result[:idx + 1] = mask
        else:
            result[idx:] = mask
        return result
    
    # Check if first token is action token
    first_is_action = torch.isin(tokens[tag_indices[0]], action_tokens_tensor)
    
    # Handle complex cases
    if first_is_action:
        result[:tag_indices[0] + 1] = mask
        remaining_indices = tag_indices[1:]
    else:
        remaining_indices = tag_indices
    
    # Handle odd number of remaining indices
    if len(remaining_indices) % 2 == 1:
        result[remaining_indices[-1]:] = mask
        remaining_indices = remaining_indices[:-1]
    
    # Process pairs
    if len(remaining_indices) >= 2:
        # Reshape to pairs and process
        pairs = remaining_indices.view(-1, 2)
        for start_idx, end_idx in pairs:
            result[start_idx:end_idx + 1] = mask
    
    return result


def benchmark_label_processing():
    """
    Benchmark function to compare performance of different versions.
    """
    import time
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate realistic token sequences
    test_tokens = []
    for _ in range(100):
        # Create a sequence with some tag tokens
        tokens = torch.randint(0, 1000, (200,), device=device)
        # Insert some tag tokens
        tokens[10] = topic_token
        tokens[20] = classify_token
        tokens[50] = sentiment_token
        tokens[100] = qa_token
        tokens[150] = answer_token
        test_tokens.append(tokens)
    
    print("Benchmarking label processing functions...")
    
    # Warmup
    for tokens in test_tokens[:50]:
        first_result = process_labels_optimized(tokens.clone())
        second_result = process_labels_ultra_optimized(tokens.clone())
        base_result = process_labels(tokens.clone())
    
    # Benchmark optimized version
    start_time = time.time()
    for tokens in test_tokens:
        _ = process_labels_optimized(tokens.clone())
    optimized_time = time.time() - start_time
    
    # Benchmark ultra-optimized version
    start_time = time.time()
    for tokens in test_tokens:
        _ = process_labels_ultra_optimized(tokens.clone())
    ultra_optimized_time = time.time() - start_time
    
    # Benchmark baseline version
    start_time = time.time()
    for tokens in test_tokens:
        _ = process_labels(tokens.clone())
    baseline_time = time.time() - start_time
    
    print(f"Optimized version result: {first_result}")
    print(f"Ultra-optimized version result: {second_result}")
    print(f"Baseline version result: {second_result}")
    
    print(f"Optimized version: {optimized_time:.4f}s")    
    print(f"Ultra-optimized version: {ultra_optimized_time:.4f}s")
    print(f"Baseline version: {baseline_time:.4f}s")
    print(f"Speedup (ultra-optimized vs optimized): {optimized_time/ultra_optimized_time:.2f}x")
    print(f"Speedup (ultra-optimized vs baseline): {baseline_time/ultra_optimized_time:.2f}x")
    assert torch.equal(first_result, second_result)
    assert torch.equal(second_result, base_result)
    assert True

def mask_long_sequences_ultra_optimized(tensor, target_value=220, mask_value=-100, min_length=2):
    """
    Ultra-optimized version using pure tensor operations and run-length encoding approach.
    """
    if len(tensor) == 0:
        return tensor
    
    # Create mask for target values
    is_target = (tensor == target_value)
    
    if not is_target.any():
        return tensor
    
    # Use cumulative sum approach to identify runs
    # Change points where target status changes
    is_target_padded = torch.cat([torch.tensor([False], device=tensor.device), 
                                is_target, 
                                torch.tensor([False], device=tensor.device)])
    
    changes = is_target_padded[1:].int() - is_target_padded[:-1].int()
    
    # Find run starts and ends
    run_starts = torch.where(changes == 1)[0]
    run_ends = torch.where(changes == -1)[0]
    
    # Calculate run lengths
    run_lengths = run_ends - run_starts
    
    # Find long runs
    long_runs_mask = run_lengths >= min_length
    
    if not long_runs_mask.any():
        return tensor
    
    # Create result tensor
    result = tensor.clone()
    
    # Mask long runs
    long_starts = run_starts[long_runs_mask]
    long_ends = run_ends[long_runs_mask]
    
    # Create a boolean mask for all positions to be masked
    mask_positions = torch.zeros_like(tensor, dtype=torch.bool)
    
    for start, end in zip(long_starts, long_ends):
        mask_positions[start:end] = True
    
    # Apply mask
    result[mask_positions] = mask_value
    
    return result


if __name__ == "__main__":
    benchmark_label_processing()






# # TODO: fix bug for cases where there are 2 tags in the list and one of them is the first or last element in the list
# if __name__ == "__main__":

#     # case 1: some text generation followed by translate without an end_of_text at the end
#     label1 = [
#         1,
#         2,
#         32,
#         54,
#         61,
#         4,
#         6,
#         76,
#         87,
#         98,
#         90,
#         91,
#         50,
#         21,
#         123,
#         45,
#         16,
#         88,
#         92,
#         6,
#     ]
#     # case 2: some text generation followed by translate with an end_of_text at the end followed by a part of classification task
#     label2 = [
#         1,
#         2,
#         32,
#         54,
#         61,
#         4,
#         6,
#         76,
#         87,
#         98,
#         90,
#         91,
#         50,
#         21,
#         123,
#         45,
#         16,
#         88,
#         92,
#         4,
#         5,
#         90,
#         128,
#         69,
#     ]
#     # case 3
#     label3 = [12, 15, 77, 88, 24, 100, 52, 34, 4, 6]

#     print(label1)
#     print(process_labels(torch.Tensor(label1)))
#     print()
#     print(label2)
#     print(process_labels(torch.Tensor(label2)))
#     print()
#     print(label3)
#     print(process_labels(torch.Tensor(label3)))
