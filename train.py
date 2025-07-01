import time
import torch
from model import GPT

start = time.perf_counter()  

############### Tier 1 (14 sec)
# training_split = 0.1
# batch_size = 32  # Dimension noted as B
# block_size = 128  # Dimension noted as T
# embed_size = 256  
# num_heads = 8  
# head_size = embed_size // num_heads 
# num_layers = 4  
# dropout = 0.1  
# num_iters = 100 
# learning_rate = 5e-4  
###############

############### Tier 2 (166 sec)
# training_split = 0.9
# batch_size = 32  # Dimension noted as B
# block_size = 128  # Dimension noted as T
# embed_size = 256  
# num_heads = 8  
# head_size = embed_size // num_heads 
# num_layers = 6  
# dropout = 0.2  
# num_iters = 1000 
# learning_rate = 3e-4  
###############

############### Tier 3 (6746 sec ~ 1 hr, 53 min) :/
training_split = 0.9
batch_size = 64  # Dimension noted as B
block_size = 256  # Dimension noted as T
embed_size = 384
num_heads = 6  
head_size = embed_size // num_heads 
num_layers = 6  
dropout = 0.2  
num_iters = 5000
learning_rate = 3e-4  
###############

# Affects progress printing
print_interval = 100 
eval_iters = 50 

torch.manual_seed(1729)
device = 'cuda'

print(f"Model loading...")
# Karpathy's tiny Shakespeare corpus
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read() 

# Character-level tokenization, this is probably the first thing to improve
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  
decode = lambda l: "".join([itos[i] for i in l])  
data = torch.tensor(encode(text), dtype=torch.long)

n = int(training_split*len(data))
train_data = data[:n]
val_data = data[n:]

# Returns (B,T),(B,T) for encoded data, encoded predicted output batches (shifted 1 to the right)
def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

model = GPT(vocab_size, embed_size, num_heads, head_size, num_layers, block_size, dropout, device)
m = model.to(device)
m = torch.compile(m)
print("Model has been initialized with", sum(p.numel() for p in m.parameters()), "parameters")
print(f"Model training...")
# Paper uses Adam, Karpathy uses AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(num_iters):
    # This just prints for us so we don't get bored, there is no effect on training
    # Also, you can watch subway surfers while it trains
    if (iter+1) % print_interval == 0:
        losses = {}
        model.eval()  
        with torch.no_grad(): 
            for split in ["train", "val"]:
                split_losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    logits, loss = model(*get_batch(split))
                    split_losses[k] = loss.item()
                losses[split] = split_losses.mean()
        model.train()  
        print(f"Step {iter+1}/{num_iters}: Training loss: {losses['train']:.6f}, Validation loss {losses['val']:.6f}")
    # Actual training
    batch_logits, batch_loss = model(*get_batch("train"))
    optimizer.zero_grad(set_to_none=True)
    batch_loss.backward()
    optimizer.step()

# Save the trained model
checkpoint = {
    'model_state_dict': model.state_dict(),
    'model_config': {
        'vocab_size': vocab_size,
        'embed_size': embed_size,
        'num_heads': num_heads,
        'head_size': head_size,
        'num_layers': num_layers,
        'block_size': block_size,
        'dropout': dropout,
        'chars': chars,
        'stoi': stoi,
        'itos': itos
    }
}
torch.save(checkpoint, 'params/gpt_model.pth')
print("Model saved")

end = time.perf_counter()
print(f"Total elapsed time: {end - start:.6f} seconds")