import torch
from model import GPT

max_tokens = 1000
device = 'cuda' 

print(f"Model loading...")
# checkpoint = torch.load('params/gpt_model.pth', map_location='cpu')
# checkpoint = torch.load('params/tier1.pth', map_location='cpu')
# checkpoint = torch.load('params/tier2.pth', map_location='cpu')
checkpoint = torch.load('params/tier3.pth', map_location='cpu')
model_config = checkpoint['model_config']
vocab_size = model_config['vocab_size']
embed_size = model_config['embed_size']
num_heads = model_config['num_heads']
head_size = model_config['head_size']
num_layers = model_config['num_layers']
block_size = model_config['block_size']
dropout = model_config['dropout']
chars = model_config['chars']
stoi = model_config['stoi']
itos = model_config['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

model = GPT(vocab_size, embed_size, num_heads, head_size, num_layers, block_size, dropout, device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

def generate_text(prompt=""):
    with torch.no_grad():
        if prompt:
            # Check if all characters in prompt are in vocabulary
            unknown_chars = set(prompt) - set(chars)
            if unknown_chars:
                print(f"Warning: Unknown characters in prompt: {unknown_chars}")
                print(f"These will be ignored.")
                prompt = ''.join(c for c in prompt if c in chars)
            if prompt:
                context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            else:
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
        else:
            # Start with newline character (index 0)
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(context, max_tokens)
        result = decode(generated[0].tolist())
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