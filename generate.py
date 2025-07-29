import time

import torch

from gpt import GPT
from tokenizer import Tokenizer

# I miss my #ifdef flags
GENERATE_SHAKESPEARE = True

max_tokens = 250

device = "cuda"

print("Model initializing...")
start = time.perf_counter()

tokenizer = Tokenizer(device)
tokenizer.load()

checkpoint = torch.load("params/gpt_model.pt", map_location="cpu")
model_config = checkpoint["model_config"]
vocab_size = model_config["vocab_size"]
embed_size = model_config["embed_size"]
num_heads = model_config["num_heads"]
head_size = model_config["head_size"]
num_layers = model_config["num_layers"]
block_size = model_config["block_size"]
dropout = model_config["dropout"]

model = GPT(
    vocab_size,
    embed_size,
    num_heads,
    head_size,
    num_layers,
    block_size,
    dropout,
    device,
)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

end = time.perf_counter()
print(f"Model initialzed in {end - start:.6f} seconds")

print(
    "Hit enter for random text. Type input text to use as a prompt. Type 'quit' to exit."
)
while True:
    try:
        user_input = input("> ")
        if user_input.lower() in ["quit"]:
            print("Goodbye")
            break
        print("> Generating...")
        start = time.perf_counter()
        with torch.no_grad():
            context = tokenizer.encode_prompt(user_input)
            generated = model.generate(context, max_tokens)
            result = tokenizer.decode(generated[0].tolist())
            print(result)
            end = time.perf_counter()
            print(f"Model generated in {end - start:.6f} seconds")
    except KeyboardInterrupt:
        print("Goodbye")
        break
