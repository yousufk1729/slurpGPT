import time
import torch

from gpt import GPT
from tokenizer import Tokenizer

max_tokens = 1000

print(f"Model loading...")
checkpoint = torch.load('params/gpt_model.pth', map_location='cpu')
# checkpoint = torch.load('params/tier1.pth', map_location='cpu')
# checkpoint = torch.load('params/tier2.pth', map_location='cpu')
# checkpoint = torch.load('params/tier3.pth', map_location='cpu')
model_config = checkpoint['model_config']
vocab_size = model_config['vocab_size']
embed_size = model_config['embed_size']
num_heads = model_config['num_heads']
head_size = model_config['head_size']
num_layers = model_config['num_layers']
block_size = model_config['block_size']
dropout = model_config['dropout']
device = model_config['device']

model = GPT(vocab_size, embed_size, num_heads, head_size, num_layers, block_size, dropout, device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

tokenizer = Tokenizer(device)
tokenizer.load('params/tokenizer.model')

def generate_text(prompt=""):
    with torch.no_grad():
        if prompt:
            try:
                context = tokenizer.encode_prompt(prompt)
            except KeyError as e:
                print(f"Error: Character '{e.args[0]}' not in vocabulary. Using empty prompt instead.")
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(context, max_tokens)
        result = tokenizer.decode(generated[0].tolist())
        return result

print(f"Model loaded.")
print("Hit enter for random text. Type some input text to use as a prompt. Type quit to exit the program.")
while True:
    try:
        user_input = input("> ")
        if user_input.lower() in ['quit']:
            print("Goodbye")
            break
        print(f"> Generating...")
        generated_text = generate_text(user_input)
        print(generated_text)
    except KeyboardInterrupt:
        print("Goodbye")
        break