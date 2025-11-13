import torch
import torch.nn.functional as F
from torch import nn


class LogicNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Logic-specific transformations
        self.logic_transform = nn.Linear(hidden_size, hidden_size)
        self.logic_activation = nn.Sigmoid()

    def forward(self, x):
        # Process with logic transformation
        return self.logic_activation(self.logic_transform(x))


class MemoryLayer(nn.Module):
    def __init__(self, hidden_size, memory_size=100, memory_dim=768):
        super().__init__()
        self.memory = nn.Parameter(
            torch.randn(memory_size, memory_dim)
        )  # Learnable memory matrix
        self.query_transform = nn.Linear(hidden_size, memory_dim)
        self.key_transform = nn.Linear(memory_dim, memory_dim)

    def forward(self, x):
        # Transform input to query the memory
        query = self.query_transform(x)

        # Compute similarity to memory keys (dot-product attention)
        attention_weights = torch.softmax(torch.matmul(query, self.memory.T), dim=-1)

        # Retrieve memory based on attention weights
        memory_output = torch.matmul(attention_weights, self.memory)

        return memory_output


class IntermediateReasoningLayer(nn.Module):
    def __init__(self, hidden_size, bottleneck_size=256):
        super().__init__()
        # Bottleneck layer to help capture reasoning features
        self.bottleneck = nn.Linear(hidden_size, bottleneck_size)
        self.relu = nn.ReLU()
        self.expand = nn.Linear(bottleneck_size, hidden_size)

    def forward(self, x):
        # Apply bottleneck and expansion
        x = self.bottleneck(x)
        x = self.relu(x)
        return self.expand(x)


class ShortTermMemoryLayer(nn.Module):
    """
    For a short-term memory mechanism, we'll clear the memory after each input or a few sequential batches, which effectively resets the memory to its initialized state,
    making it resemble temporary storage.
    The reset_memory method re-initializes the memory parameters after each batch.
    This reset could be called at the end of each input batch to limit memory to the current batch, keeping it “short-term.”
    """

    def __init__(self, hidden_size, memory_size=10, memory_dim=768):
        super().__init__()
        self.memory = nn.Parameter(
            torch.randn(memory_size, memory_dim)
        )  # Temporary memory initialized per batch
        self.query_transform = nn.Linear(hidden_size, memory_dim)
        self.key_transform = nn.Linear(memory_dim, memory_dim)

    def forward(self, x):
        # Transform input embeddings into queries for the memory
        query = self.query_transform(x)
        attention_weights = torch.softmax(torch.matmul(query, self.memory.T), dim=-1)
        memory_output = torch.matmul(attention_weights, self.memory)
        return memory_output

    def reset_memory(self):
        # Re-initialize memory after each batch or sequence
        nn.init.normal_(self.memory, mean=0, std=0.02)


class PersistentMemoryLayer(nn.Module):
    """
    For a persistent state mechanism, we’ll modify the memory so it isn’t reset with each batch or sequence. Instead, we’ll incorporate selective updates,
    allowing the model to update memory only when certain conditions are met, like a model improvement, memory overwrite threshold, or based on certain relevance
    metrics.

    We update memory selectively based on a threshold criterion, only adding information that is sufficiently different from existing memory states.
      The update_memory function replaces the least relevant memory embedding when new information meets the update criteria, which is useful for episodic learning.
    """

    def __init__(self, hidden_size, memory_size=100, memory_dim=768):
        super().__init__()
        # Persistent memory that retains knowledge over time
        self.memory = nn.Parameter(
            torch.randn(memory_size, memory_dim), requires_grad=False
        )  # Frozen by default
        self.query_transform = nn.Linear(hidden_size, memory_dim)
        self.key_transform = nn.Linear(memory_dim, memory_dim)
        self.memory_update_layer = nn.Linear(hidden_size, memory_dim)

    def forward(self, x):
        query = self.query_transform(x)
        attention_weights = torch.softmax(torch.matmul(query, self.memory.T), dim=-1)
        memory_output = torch.matmul(attention_weights, self.memory)
        return memory_output

    def update_memory(self, x, threshold=0.7):
        # Update memory selectively based on similarity threshold
        new_memory_embedding = self.memory_update_layer(x.mean(dim=1))
        with torch.no_grad():
            similarity_scores = torch.matmul(new_memory_embedding, self.memory.T)
            if (
                torch.max(similarity_scores) < threshold
            ):  # Update if no close match found
                # Replace the least used memory slot or a random slot with new memory
                index_to_update = torch.argmin(similarity_scores)
                self.memory[index_to_update] = new_memory_embedding


class PersistentMemoryWithUsage(nn.Module):
    """
    let’s add a usage counter for each memory slot, updating or replacing the least-used memory slots. This way, we maximize memory efficiency.
     Each slot’s usage count is tracked, allowing the model to replace the least-used slot when an update is needed.
     This balances memory persistence with an efficient, self-regulating mechanism for updating memory over time.
    """

    def __init__(self, hidden_size, memory_size=100, memory_dim=768):
        super().__init__()
        self.memory = nn.Parameter(
            torch.randn(memory_size, memory_dim), requires_grad=False
        )
        self.usage_counts = torch.zeros(
            memory_size, dtype=torch.int
        )  # Track memory usage
        self.query_transform = nn.Linear(hidden_size, memory_dim)
        self.memory_update_layer = nn.Linear(hidden_size, memory_dim)

    def forward(self, x):
        query = self.query_transform(x)
        attention_weights = torch.softmax(torch.matmul(query, self.memory.T), dim=-1)
        memory_output = torch.matmul(attention_weights, self.memory)

        # Update usage counts
        most_used_slot = torch.argmax(attention_weights, dim=-1)
        self.usage_counts[most_used_slot] += 1
        return memory_output

    def update_memory_with_least_used(self, x):
        new_memory_embedding = self.memory_update_layer(x.mean(dim=1))
        with torch.no_grad():
            least_used_index = torch.argmin(self.usage_counts)
            self.memory[least_used_index] = new_memory_embedding
            self.usage_counts[least_used_index] = 1  # Reset count for updated slot
