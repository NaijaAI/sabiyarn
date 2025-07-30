import torch
from time import time


def create_causal_mask(tensor, original_mask, id_val=30, min_value=-1e-25):
    """
    Generates a new causal attention mask based on the location of ID 30 (end_of_text_id).

    Args:
        mask: A torch tensor with shape (batch_size, seq_len, seq_len) representing
              the original square attention mask.

    Returns:
        A new torch tensor with the same shape as the input mask,
        where elements before the most recent ID 30 (with mask value 1)
        are masked (set to 0).
    """
    if torch.any(tensor == id_val):
        original_mask = original_mask.squeeze(1)
        new_mask = original_mask.clone()  # Clone the original mask
        max_value, mask_min_value = torch.max(original_mask), torch.min(original_mask)
        # print("mask_min_value: ", max_value, mask_min_value)
        # print("Original mask: ", original_mask.shape)
        original_mask = torch.where(original_mask == max_value, 1, 0)
        batch_size = original_mask.size()[0]

        device = tensor.device
        original_mask = original_mask.to(device)
        new_mask = new_mask.to(device)

        for i in range(batch_size):
            id_positions = (tensor[i] == id_val).nonzero(as_tuple=False)
            # print("ID positions: ", id_positions.shape)
            id_positions = id_positions.squeeze()

            if len(original_mask.shape) < 3:
                original_mask = original_mask.unsqueeze(0)

            # print("squeezed ID positions: ", id_positions.shape)
            # print("Original mask (after conditioning): ", original_mask.shape)#, original_mask.unsqueeze(0).shape)
            # print("Original mask (per batch i): ", original_mask[i].shape)
            # print("condition : ", (original_mask[i, id_positions, id_positions] == 1).shape)
            valid_positions = id_positions[
                original_mask[i, id_positions, id_positions] == 1
            ]
            test = original_mask[i, :, valid_positions] == 1
            if test.numel() == 0:
                continue

            if len(test.shape) < 2:
                test = test.unsqueeze(-1)

            is_all_false = torch.where(torch.sum(test.to(float), dim=-1) == 0, -1, 1)
            reversed_tensor = test.flip(dims=[-1])
            indices = reversed_tensor.to(torch.float).argmax(dim=1)

            # Step 3: Adjust the indices to refer to the original tensor's indexing
            adjusted_indices = test.size(1) - 1 - indices
            negative_indices = (adjusted_indices + 1) * is_all_false
            adjusted_indices = torch.where(negative_indices < 0, -1, adjusted_indices)
            rows = (adjusted_indices >= 0).nonzero(as_tuple=False)
            if rows.shape == 2:
                rows = rows.squeeze()
            adjusted_indices = adjusted_indices[adjusted_indices >= 0]
            pos = valid_positions[adjusted_indices]
            # print("rows: ", rows)
            for index, k in enumerate(rows):
                new_mask[i, k, : pos[index]] = min_value

        # new_mask= torch.where(new_mask == mask_min_value, min_value, new_mask)
        return new_mask.unsqueeze(1)
    else:
        return original_mask

def create_causal_mask_optimized(tensor, original_mask, id_val=30, min_value=-1e-25):
    """
    Optimized version: Generates a new causal attention mask based on the location of ID 30 (end_of_text_id).
    
    Key optimizations:
    1. Early exit if no id_val found
    2. Vectorized operations instead of loops
    3. Reduced tensor operations and memory allocations
    4. Batch processing using advanced indexing
    
    Args:
        tensor: Input tensor to search for id_val
        original_mask: Original attention mask (batch_size, 1, seq_len, seq_len) or (batch_size, seq_len, seq_len)
        id_val: ID value to search for (default: 30)
        min_value: Value to use for masking (default: -1e-25)
    
    Returns:
        New attention mask with causal masking applied
    """
    # Early exit if id_val not found
    if not torch.any(tensor == id_val):
        return original_mask
    
    # Handle mask dimensions
    squeeze_needed = False
    if original_mask.dim() == 4:
        original_mask = original_mask.squeeze(1)
        squeeze_needed = True
    
    batch_size, seq_len, _ = original_mask.shape
    device = original_mask.device
    
    # Clone mask once
    new_mask = original_mask.clone()
    
    # Convert to binary mask efficiently
    max_val = original_mask.max()
    binary_mask = (original_mask == max_val).float()
    
    # Find all positions where id_val occurs for all batches at once
    id_positions = (tensor == id_val)  # (batch_size, seq_len)
    
    # Process each batch
    batch_indices = torch.arange(batch_size, device=device)
    
    for i in range(batch_size):
        if not id_positions[i].any():
            continue
            
        # Get valid id positions for this batch
        batch_id_pos = torch.where(id_positions[i])[0]
        
        # Check which positions are valid (have mask value 1 in diagonal)
        valid_mask = binary_mask[i, batch_id_pos, batch_id_pos] == 1
        valid_positions = batch_id_pos[valid_mask]
        
        if len(valid_positions) == 0:
            continue
        
        # Find the last valid position for each row
        # This replaces the complex flip and argmax logic
        mask_at_valid_pos = binary_mask[i, :, valid_positions]  # (seq_len, num_valid_pos)
        
        if mask_at_valid_pos.numel() == 0:
            continue
        
        # Find last True position in each row
        # Flip and find first True, then convert back to original indexing
        flipped = mask_at_valid_pos.flip(dims=[1])
        last_true_flipped = torch.argmax(flipped.float(), dim=1)
        
        # Convert back to original indexing
        last_true_pos = mask_at_valid_pos.size(1) - 1 - last_true_flipped
        
        # Only process rows that have at least one True value
        has_true = mask_at_valid_pos.any(dim=1)
        
        if not has_true.any():
            continue
        
        # Get the actual positions to mask up to
        row_indices = torch.where(has_true)[0]
        col_positions = valid_positions[last_true_pos[has_true]]
        
        # Apply masking efficiently using advanced indexing
        for row_idx, col_pos in zip(row_indices, col_positions):
            new_mask[i, row_idx, :col_pos] = min_value
    
    # Restore original dimensions if needed
    if squeeze_needed:
        new_mask = new_mask.unsqueeze(1)
    
    return new_mask


def create_causal_mask_ultra_optimized(tensor, original_mask, id_val=30, min_value=-1e-25):

    if not torch.any(tensor == id_val):
        return original_mask

    # Handle dimensions: Ensure original_mask is (batch_size, seq_len, seq_len)
    squeeze_needed = False
    if original_mask.dim() == 4:
        original_mask = original_mask.squeeze(1)
        squeeze_needed = True

    batch_size, seq_len, _ = original_mask.shape
    device = original_mask.device

    new_mask = original_mask.clone()

    # Convert original_mask to binary for valid positions (where original_mask == max_value)
    # The original function uses max_value, so we adhere to that.
    max_val = original_mask.max()
    binary_original_mask = (original_mask == max_val).float()

    # Find where id_val occurs in the tensor
    id_positions_in_tensor = (tensor == id_val) # (batch_size, seq_len)

    # Create a diagonal mask for the original_mask
    # This checks if the original mask allows attention at the id_val's position (diagonal element is 1)
    diag_mask = torch.eye(seq_len, device=device).bool()
    # Expand diag_mask to (batch_size, seq_len, seq_len) for element-wise multiplication
    diag_mask_expanded = diag_mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Get the diagonal values from the binary_original_mask
    # This effectively gets original_mask[i, j, j]
    original_mask_diag_values = binary_original_mask[:, torch.arange(seq_len), torch.arange(seq_len)] # (batch_size, seq_len)

    valid_id_occurrence = id_positions_in_tensor * original_mask_diag_values.bool()
    reversed_valid_id_occurrence = torch.flip(valid_id_occurrence, dims=[1])
   
    # Create a tensor of indices (0 to seq_len-1)
    indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1) # (batch_size, seq_len)
    
    # Mask out non-valid occurrences with a very small number for argmax
    # This makes sure argmax picks a valid position if available, otherwise a very small index.
    masked_indices_for_argmax = torch.where(valid_id_occurrence, indices.float(), torch.tensor(-1.0, device=device))
    
    # Find the index of the largest value (which corresponds to the rightmost valid ID)
    # If no valid ID, this will be 0 (the index of -1.0), so we handle that later.
    last_valid_pos_per_batch = torch.argmax(masked_indices_for_argmax, dim=1) # (batch_size,)

    # Handle batches where no valid id_val was found.
    # If max value of masked_indices_for_argmax for a batch is -1.0, it means no valid ID.
    no_valid_id_found = (masked_indices_for_argmax.max(dim=1).values == -1.0)
    
    # For these batches, set their last_valid_pos to -1 (or a value indicating no masking)
    # This ensures they don't incorrectly apply a mask based on index 0.
    last_valid_pos_per_batch[no_valid_id_found] = -1
 
    # Create a `col_indices` tensor (1, 1, seq_len) for broadcasting
    col_indices = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0) # (1, 1, seq_len)

    # Expand valid_id_occurrence to (batch_size, 1, seq_len) for broadcasting
    valid_id_occurrence_for_masking = valid_id_occurrence.unsqueeze(1) # (batch_size, 1, seq_len)

    # Condition 1: original_mask[batch, row, col] == max_val
    cond1 = (original_mask == max_val) # (batch_size, seq_len, seq_len)

    # Condition 2: col is a valid_id_occurrence
    cond2 = valid_id_occurrence_for_masking # (batch_size, 1, seq_len)

    # Combined condition: original_mask connects to a valid id_val position
    combined_connection_mask = cond1 & cond2 # (batch_size, seq_len, seq_len)

    # Now, for each (batch, row), find the maximum `col_indices` where `combined_connection_mask` is True.
    # If no True, it means no connection to a valid `id_val`.
    
    # Set non-True elements to a very small number for argmax
    masked_col_indices = torch.where(combined_connection_mask, 
                                     col_indices.float().expand_as(combined_connection_mask), 
                                     torch.tensor(-1.0, device=device))

    # Find the rightmost connected valid ID position for each (batch, row)
    # last_connected_pos will be (batch_size, seq_len)
    last_connected_pos = torch.argmax(masked_col_indices, dim=2)

    # Identify where no valid connection was found (max value is -1.0)
    no_connection_found = (masked_col_indices.max(dim=2).values == -1.0)
    last_connected_pos[no_connection_found] = -1 # Sentinel value for no mask

    # Create a range tensor for columns: (1, 1, seq_len)
    mask_col_range = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0)

    # Expand last_connected_pos to (batch_size, seq_len, 1) for broadcasting
    last_pos_expanded = last_connected_pos.unsqueeze(2)

    # Condition for masking: col_idx < last_connected_pos AND last_connected_pos is valid
    mask_condition = (mask_col_range < last_pos_expanded) & (last_pos_expanded != -1) # (batch_size, seq_len, seq_len)

    # Apply the mask
    new_mask = torch.where(mask_condition, torch.tensor(min_value, device=device), new_mask)

    # Restore dimensions if original_mask was 4D
    if squeeze_needed:
        new_mask = new_mask.unsqueeze(1)

    return new_mask

# Benchmarking function
def benchmark_mask_functions():
    """
    Compare performance of original vs optimized versions
    """
    import time
    
    # Test parameters
    batch_size, seq_len = 10, 4096
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    tensor = torch.randint(0, 100, (batch_size, seq_len), device=device)
    tensor[:, seq_len//2] = 30  # Ensure id_val exists
    
    # original_mask = torch.rand(batch_size, 1, seq_len, seq_len, device=device)
    # original_mask = torch.where(original_mask > 0.5, 1.0, -1e-25)
    
     # Generate the original causal mask
    original_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.int))

    # Add a batch dimension and a channel dimension
    expanded_mask = original_mask.unsqueeze(0).unsqueeze(
        0
    )  # Now shape is (1, 1, 20, 20)

    # Repeat the tensor along the batch dimension
    original_mask = expanded_mask.repeat(batch_size, 1, 1, 1)
    
    # Warmup
    for _ in range(10):
        _ = create_causal_mask_optimized(tensor, original_mask.clone())
        _ = create_causal_mask_ultra_optimized(tensor, original_mask.clone())
    
    iters = 10
    # Benchmark optimized version
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(iters):
        result1 = create_causal_mask_optimized(tensor, original_mask.clone())
    torch.cuda.synchronize() if device.type == 'cuda' else None
    optimized_time = time.time() - start_time
    
    # Benchmark ultra-optimized version
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(iters):
        result2 = create_causal_mask_ultra_optimized(tensor, original_mask.clone())
    torch.cuda.synchronize() if device.type == 'cuda' else None
    ultra_optimized_time = time.time() - start_time
    
    # Benchmark baseline version
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(iters):
        result3 = create_causal_mask(tensor, original_mask.clone())
    torch.cuda.synchronize() if device.type == 'cuda' else None
    baseline_time = time.time() - start_time
    
    print(f"Input: {tensor}")
    print(f"Results optimized : { result1.to(torch.int32)[0]}")
    print(f"Input: {tensor}")
    print(f"Results ultraoptimized : { result2.to(torch.int32)[0]}") 
    print(f"Input: {tensor}")
    print(f"Results baseline: {result3.to(torch.int32)[0]}") 
    
    print(f"baseline version: {baseline_time:.4f}s")
    print(f"Optimized version: {optimized_time:.4f}s")
    print(f"Ultra-optimized version: {ultra_optimized_time:.4f}s")
    print(f"Speedup (ultra-optimized vs baseline): {baseline_time/ultra_optimized_time:.2f}x")
    print(f"Speedup (ultra-optimized vs optimized): {optimized_time/ultra_optimized_time:.2f}x")
    
    # Verify results are similar
    print(f"Results match: {torch.equal(result1.to(torch.int32), result2.to(torch.int32))}")
    print(f"Results match: {torch.equal(result3.to(torch.int32), result2.to(torch.int32))}") 
    
    # print(f"Results match: {torch.allclose(result1, result2, atol=1e-6)}")
    # print(f"Results match: {torch.allclose(result3, result2, atol=1e-6)}")
    


if __name__ == "__main__":
    import torch
    benchmark_mask_functions()
    # Generate example IDs
    # ids = torch.randint(0, 100, (3, 20))  # Shape: (3, 20)

    # # Ensure that ID 30 appears at least 3 times anywhere in each sequence
    # for i in range(ids.shape[0]):
    #     # Choose random indices to replace with 30
    #     replace_indices = torch.randint(0, ids.shape[1], (3,))
    #     ids[i, replace_indices] = 30
    #     # ids[i, replace_indices-1]=30

    # # Generate the original causal mask
    # original_mask = torch.tril(torch.ones(20, 20, dtype=torch.int))

    # # Add a batch dimension and a channel dimension
    # expanded_mask = original_mask.unsqueeze(0).unsqueeze(
    #     0
    # )  # Now shape is (1, 1, 20, 20)

    # # Repeat the tensor along the batch dimension
    # original_mask = expanded_mask.repeat(3, 1, 1, 1)

    # # Print the IDs and the original mask
    # print("IDs:")
    # print(ids)
    # print("\nOriginal Mask:")

    # start_time = time()
    # causal_mask = create_causal_mask(ids, original_mask)
    # # print(causal_mask.shape, causal_mask[1])
    # end_time = time()
    # print(f"Time taken: {end_time - start_time} seconds..")

    # for i in range(3):
    #     print(ids[i])
    #     print(causal_mask[i])

    # print("Original mask: " , original_mask[1])

    # Example tensor and mask
    # tensor = torch.tensor([
    #     [1, 26, 30, 5, 5, 7, 30, 12, 16],
    #     [23, 45, 67, 30, 62, 30, 97, 80, 85]
    # ])

    # original_mask = torch.tensor([
    #     [[1, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [1, 1, 0, 0, 0, 0, 0, 0, 0],
    #      [1, 1, 1, 0, 0, 0, 0, 0, 0],
    #      [1, 1, 1, 1, 0, 0, 0, 0, 0],
    #      [1, 1, 1, 1, 1, 0, 0, 0, 0],
    #      [1, 1, 1, 1, 1, 1, 0, 0, 0],
    #      [1, 1, 1, 1, 1, 1, 1,  0, 0],
    #      [1, 1, 1, 1, 1, 1, 1, 1,  0],
    #      [1, 1, 1, 1, 1, 1, 1, 1, 1]],
    #     [[1, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [1, 1, 0, 0, 0, 0, 0, 0, 0],
    #      [1, 1, 1, 0, 0, 0, 0, 0, 0],
    #      [1, 1, 1, 1, 0, 0, 0, 0, 0],
    #      [1, 1, 1, 1, 1, 0, 0, 0, 0],
    #      [1, 1, 1, 1, 1, 1, 0, 0, 0],
    #      [1, 1, 1, 1, 1, 1, 1,  0, 0],
    #      [1, 1, 1, 1, 1, 1, 1, 1,  0],
    #      [1, 1, 1, 1, 1, 1, 1, 1, 1]]
    # ], dtype=torch.int)

    # start_time = time()
    # causal_mask = create_causal_mask(tensor, original_mask)
    # print(causal_mask.shape, causal_mask)
    # end_time = time()
    # print(f"Time taken: {end_time - start_time} seconds..")

    # start_time = time()
    # print(create_causal_mask2(tensor, original_mask))
    # end_time = time()
    # print(f"Time taken: {end_time - start_time} seconds..")

    # def create_causal_mask(tensor, mask, id_val=30):
    #   """
    #   Generates a new causal attention mask based on the location of ID 30.

    #   Args:
    #       mask: A torch tensor with shape (batch_size, seq_len, seq_len) representing
    #             the original square attention mask.

    #   Returns:
    #       A new torch tensor with the same shape as the input mask,
    #       where elements before the most recent ID 30 (with mask value 1)
    #       are masked (set to 0).
    #   """
    #   batch_size, seq_len = mask.size()[:2]
    #   new_mask = mask.clone()  # Clone the original mask
    #   for i in range(batch_size):
    #     id_positions = (tensor[i] == id_val).nonzero(as_tuple=False).squeeze()
    #     # Filter those positions where the original mask is 1 (i.e., unmasked)
    #     #print("ID positions: ", id_positions)
    #     valid_positions = id_positions[original_mask[i, id_positions, id_positions] == 1]
    #     print("valid positions : ", valid_positions)
    #     test = original_mask[i, :, id_positions] == 1

    #     print("Test condition : ", test)

    #     if len(test.shape) <2 :
    #         test= test.unsqueeze(-1)

    #     is_all_false = torch.where(torch.sum(test.to(float), dim=-1) == 0, -1, 1)
    #     print("is all false: ", is_all_false)
    #     reversed_tensor = test.flip(dims=[-1])
    #     print("reversed tensor: ", reversed_tensor)
    #     # Step 2: Use argmax to find the index of the first True value; this also works if there are no True values, returning 0
    #     indices = reversed_tensor.to(torch.float).argmax(dim=1)

    #     # Step 3: Adjust the indices to refer to the original tensor's indexing
    #     adjusted_indices = test.size(1) - 1 - indices
    #     print("Adjusted indices before is_false: ", adjusted_indices)
    #     negative_indices = (adjusted_indices + 1) * is_all_false
    #     print("Adjusted indices after is_false: ", adjusted_indices)
    #     adjusted_indices = torch.where(negative_indices < 0 , -1 , adjusted_indices)
    #     print("Adjusted indices after condition: ", adjusted_indices)
    #     rows = (adjusted_indices >= 0).nonzero(as_tuple=False).squeeze()
    #     adjusted_indices = adjusted_indices[adjusted_indices >= 0]
    #     pos = valid_positions[adjusted_indices]
    #     print("Rows : ", rows)
    #     #print("Adjusted indices : ", adjusted_indices)
    #     print("final: ", pos)

    #     for index, k in enumerate(rows):
    #         new_mask[i, k , : pos[index]] = 0

    #   return new_mask
