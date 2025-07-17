import torch

input_path = 'data/input.txt'

class Tokenizer:
    def __init__(self, device):
        with open(input_path, 'r', encoding='utf-8') as f:
            self.text = f.read() 
        self.tokens = sorted(list(set(self.text)))
        self.vocab_size = len(self.tokens)
        self.stoi = {ch: i for i, ch in enumerate(self.tokens)}
        self.itos = {i: ch for i, ch in enumerate(self.tokens)}
        self.device = device
        self.tokens = torch.tensor(self.encode(self.text), dtype=torch.long, device=self.device)
        self.default_token = torch.zeros((1, 1), dtype=torch.long, device=self.device)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return "".join([self.itos[i] for i in l])
    
    def tokenize_prompt(self, prompt):
        context = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        return context
    