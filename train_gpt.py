import time

import torch

from gpt import GPT
from tokenizer import Tokenizer

# Tier 1 (~13.3 sec)
# Tier 2 (~173 sec)
# Tier 3 (~1 hr, 45 min)
TIER = 3

TIER_CONFIGS = {
    1: {
        "training_split": 0.1,
        "batch_size": 32,
        "block_size": 128,
        "embed_size": 256,
        "num_heads": 8,
        "num_layers": 4,
        "dropout": 0.1,
        "num_iters": 100,
        "learning_rate": 5e-4,
    },
    2: {
        "training_split": 0.9,
        "batch_size": 32,
        "block_size": 128,
        "embed_size": 256,
        "num_heads": 8,
        "num_layers": 6,
        "dropout": 0.2,
        "num_iters": 1000,
        "learning_rate": 3e-4,
    },
    3: {
        # I only have 4 GB of GPU space and a few hours of patience, which limits # of parameters I can try
        "training_split": 0.9,
        "batch_size": 64,
        "block_size": 256,
        "embed_size": 384,
        "num_heads": 6,
        "num_layers": 6,
        "dropout": 0,  # I love overfitting
        "num_iters": 5000,
        "learning_rate": 3e-4,
    },
}

config = TIER_CONFIGS[TIER]
training_split = config["training_split"]
batch_size = config["batch_size"]
block_size = config["block_size"]
embed_size = config["embed_size"]
num_heads = config["num_heads"]
head_size = embed_size // num_heads
num_layers = config["num_layers"]
dropout = config["dropout"]
num_iters = config["num_iters"]
learning_rate = config["learning_rate"]

print_interval = 100
eval_iters = 50

torch.manual_seed(1729)
device = "cuda"

print(f"Model initializing with Tier {TIER} configuration...")

tokenizer = Tokenizer(device)
tokenizer.load("params/tokenizer.model")
n = int(training_split * len(tokenizer.tokens))
print(f"Training size: {n} tokens")
train_tokens = tokenizer.tokens[:n]
val_tokens = tokenizer.tokens[n:]


# Returns (B,T),(B,T) for encoded tokens, encoded predicted output batches (shifted 1 to the right)
def get_batch(split):
    tokens = train_tokens if split == "train" else val_tokens
    idx = torch.randint(len(tokens) - block_size, (batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in idx])
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y


model = GPT(
    tokenizer.vocab_size,
    embed_size,
    num_heads,
    head_size,
    num_layers,
    block_size,
    dropout,
    device,
)
model = model.to(device)

print(
    "Model has been initialized with",
    sum(p.numel() for p in model.parameters()),
    "parameters",
)
print("Model training...")
start = time.perf_counter()

# Paper uses Adam, Karpathy uses AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(num_iters):
    # This just prints for us so we don't get bored, there is no effect on training
    # Also, you can watch subway surfers while it trains
    if (iter + 1) % print_interval == 0:
        losses = {}
        model.eval()
        with torch.inference_mode():
            for split in ["train", "val"]:
                split_losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    logits, loss = model(*get_batch(split))
                    split_losses[k] = loss.item()
                losses[split] = split_losses.mean()
        model.train()
        print(
            f"Step {iter + 1}/{num_iters}: Training loss: {losses['train']:.6f}, Validation loss {losses['val']:.6f}"
        )
    batch_logits, batch_loss = model(*get_batch("train"))
    optimizer.zero_grad(set_to_none=True)
    batch_loss.backward()
    optimizer.step()

end = time.perf_counter()
print(f"Model trained in {end - start:.6f} seconds")

checkpoint = {
    "model_state_dict": model.state_dict(),
    "model_config": {
        "vocab_size": tokenizer.vocab_size,
        "embed_size": embed_size,
        "num_heads": num_heads,
        "head_size": head_size,
        "num_layers": num_layers,
        "block_size": block_size,
        "dropout": dropout,
    },
}

torch.save(checkpoint, "params/gpt_model.pth")
print("Model saved")
