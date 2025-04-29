import torch
from time import time


def create_causal_mask(tensor, original_mask, id_val=30, min_value=-1e-25):
    """
    Generates a new causal attention mask based on the location of ID 30.

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


if __name__ == "__main__":
    import torch

    # Generate example IDs
    ids = torch.randint(0, 100, (3, 20))  # Shape: (3, 20)

    # Ensure that ID 30 appears at least 3 times anywhere in each sequence
    for i in range(ids.shape[0]):
        # Choose random indices to replace with 30
        replace_indices = torch.randint(0, ids.shape[1], (3,))
        ids[i, replace_indices] = 30

    # Generate the original causal mask
    original_mask = torch.tril(torch.ones(20, 20, dtype=torch.int))

    # Add a batch dimension and a channel dimension
    expanded_mask = original_mask.unsqueeze(0).unsqueeze(
        0
    )  # Now shape is (1, 1, 20, 20)

    # Repeat the tensor along the batch dimension
    original_mask = expanded_mask.repeat(3, 1, 1, 1)

    # Print the IDs and the original mask
    print("IDs:")
    print(ids)
    print("\nOriginal Mask:")

    start_time = time()
    causal_mask = create_causal_mask(ids, original_mask)
    # print(causal_mask.shape, causal_mask[1])
    end_time = time()
    print(f"Time taken: {end_time - start_time} seconds..")

    for i in range(3):
        print(ids[i])
        print(causal_mask[i])

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
