import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel, destroy_model_parallel
from fairscale.nn.model_parallel import mpu
from model import *
from tokenizer import *
from torch import nn

# Assuming that ModelArgs, Attention, FeedForward, and other relevant classes are defined above
# Initialize model parallelism if needed
initialize_model_parallel(1)  # Use 1 model parallel size for simplicity

def test_attention():
    # Model configuration
    args = ModelArgs(
        dim=4096,
        n_heads=32,
        n_kv_heads=None,
        max_batch_size=2,
        max_seq_len=128
    )
    
    # Initialize Attention Module
    attention = Attention(args).cuda()
    
    # Dummy inputs for testing
    batch_size = 2
    seq_len = 128
    x = torch.rand(batch_size, seq_len, args.dim).cuda()
    start_pos = 0
    freqs_cis = precompute_freqs_cis(dim=args.dim, end=seq_len).cuda()
    mask = None  # No mask for this test

    # Run attention module
    output = attention(x, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
    print("Attention output shape:", output.shape)

    # Check output shape
    assert output.shape == (batch_size, seq_len, args.dim), "Attention output shape is incorrect"

    print("Attention module test passed!")


def test_feedforward():
    # FeedForward configuration
    dim = 4096
    hidden_dim = dim * 4  # typically hidden_dim is larger than dim
    multiple_of = 256
    ffn_dim_multiplier = None  # Use default multiplier

    # Initialize FeedForward Module
    feedforward = FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier).cuda()

    # Dummy inputs for testing
    batch_size = 2
    seq_len = 128
    x = torch.rand(batch_size, seq_len, dim).cuda()

    # Run feedforward module
    output = feedforward(x)
    print("FeedForward output shape:", output.shape)

    # Check output shape
    assert output.shape == (batch_size, seq_len, dim), "FeedForward output shape is incorrect"

    print("FeedForward module test passed!")

# Test function for the tokenizer
def test_tokenizer():
    model_path = "path/to/your/sentencepiece.model"  # Replace with the actual path
    tokenizer = Tokenizer(model_path)
    
    # Sample text
    sample_text = "Hello, world!"
    print("Original Text:", sample_text)
    
    # Encode
    encoded = tokenizer.encode(sample_text, bos=True, eos=True)
    print("Encoded:", encoded)
    
    # Decode
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    # Check that decode(encode(text)) == text
    assert decoded == sample_text, "Decoded text does not match the original text"
    
    print("Tokenizer test passed!")
    

# Assuming ModelArgs, Transformer, Attention, and FeedForward modules are defined and imported.

def build_and_test_transformer():
    # Model configuration
    args = ModelArgs(
        dim=4096,
        n_layers=12,
        n_heads=32,
        n_kv_heads=None,
        vocab_size=50257,
        max_batch_size=2,
        max_seq_len=128,
    )
    
    # Initialize the Transformer model
    model = Transformer(args).cuda()

    # Create dummy input data
    batch_size = 2
    seq_len = args.max_seq_len
    dummy_input = torch.randint(0, args.vocab_size, (batch_size, seq_len)).cuda()

    # Run the Transformer model with dummy data
    output = model(dummy_input)

    # Print output shape
    print("Transformer output shape:", output.shape)

    # Check if output shape matches the expected shape
    assert output.shape == (batch_size, seq_len, args.dim), "Transformer output shape is incorrect"

    print("Transformer model test passed!")

    
# Run tests
if __name__ == "__main__":
    # Initialize model parallel (if needed)
    initialize_model_parallel(1)
    test_attention()
    test_feedforward()
    build_and_test_transformer()
    test_tokenizer()
    # Cleanup model parallel
    destroy_model_parallel()
