"""
train_gpt.py
Train the GPT with certain hyperparameters.

https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
https://arxiv.org/pdf/1706.03762
https://arxiv.org/pdf/1607.06450
https://arxiv.org/pdf/1512.03385
https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import time
import os

import torch

from gpt import GPT
from tokenizer import Tokenizer

# Tier 1 (~13.3 sec)
# Tier 2 (~320 sec)
# Tier 3 (~7 hr 35 min)
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
        "learning_rate": 1e-3,
    },
    3: {
        # I only have 4 GB of GPU space and a few hours of patience, which limits # of parameters I can try
        "training_split": 0.9,
        "batch_size": 64,
        "block_size": 256,
        "embed_size": 384,
        "num_heads": 6,
        "num_layers": 6,
        "dropout": 0.2,
        "num_iters": 1000,
        "learning_rate": 1e-3,
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
eval_iters = 200

patience = 5  # Number of print_intervals to wait for improvement
early_stop_threshold = 1e-4  # Minimum improvement to be considered significant

torch.manual_seed(1729)
device = "cuda"

print(f"Model initializing with Tier {TIER} configuration...")

tokenizer = Tokenizer(device)
tokenizer.load()
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


def save_checkpoint(
    model,
    tokenizer,
    embed_size,
    num_heads,
    head_size,
    num_layers,
    block_size,
    dropout,
    filename,
):
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
    torch.save(checkpoint, filename)


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

best_val_loss = float("inf")
patience_counter = 0
best_model_path = "params/gpt_model.pt"

for iter in range(num_iters):
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

        current_val_loss = losses["val"].item()
        print(
            f"Step {iter + 1}/{num_iters}: Training loss: {losses['train']:.6f}, Validation loss {current_val_loss:.6f}"
        )

        if current_val_loss < best_val_loss - early_stop_threshold:
            best_val_loss = current_val_loss
            patience_counter = 0
            save_checkpoint(
                model,
                tokenizer,
                embed_size,
                num_heads,
                head_size,
                num_layers,
                block_size,
                dropout,
                best_model_path,
            )
            print(f"New best validation loss: {best_val_loss:.6f}. Best model saved")
        else:
            patience_counter += 1
            print(
                f"No improvement in validation loss. Patience: {patience_counter}/{patience}"
            )
            if patience_counter >= patience:
                print(f"Early stopping triggered after {iter + 1} iterations")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break

    batch_logits, batch_loss = model(*get_batch("train"))
    optimizer.zero_grad(set_to_none=True)
    batch_loss.backward()
    optimizer.step()

end = time.perf_counter()
print(f"Model trained in {end - start:.6f} seconds")

if patience_counter < patience:
    final_checkpoint_path = "params/gpt_model_final.pt"
    save_checkpoint(
        model,
        tokenizer,
        embed_size,
        num_heads,
        head_size,
        num_layers,
        block_size,
        dropout,
        final_checkpoint_path,
    )
    print(f"Final model saved: {final_checkpoint_path}")
