import torch

def expand_arithmetic(tensor):
    # Assuming tensor shape is (Batch, Features, ..., N-dimensions)
    # We perform the arithmetic operations
    multiplied = tensor.unsqueeze(-1) * tensor.unsqueeze(-2)
    divided = tensor.unsqueeze(-1) / tensor.unsqueeze(-2)
    added = tensor.unsqueeze(-1) + tensor.unsqueeze(-2)
    subtracted = tensor.unsqueeze(-1) - tensor.unsqueeze(-2)
    
    # Now, we stack them together, we'll have a new dimension at the first position
    # indicating the operation: 0 for multiply, 1 for divide, 2 for add, 3 for subtract
    result = torch.stack([multiplied, divided, added, subtracted], dim=1)
    
    return result

# Example usage:
batch, features = 10, 5
# For 1D case
tensor_1d = torch.randn(batch, features)
result_1d = expand_arithmetic(tensor_1d)
print("1D Result shape:", result_1d.shape)

# For 2D case (just for demonstration, adding an extra dimension)
tensor_2d = torch.randn(batch, features, features)
result_2d = expand_arithmetic(tensor_2d)
print("2D Result shape:", result_2d.shape)