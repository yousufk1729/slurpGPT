import time
from tokenizer import Tokenizer

device = 'cuda'

start = time.perf_counter()  
tokenizer = Tokenizer(device)
tokenizer.load('params/tokenizer.model')
# tokenizer.train(verbose=True)
# tokenizer.save()
print(tokenizer.vocab)
print(tokenizer.merges)
print(tokenizer.pattern)
print(len(tokenizer.tokens))
end = time.perf_counter()  

print(f"Training took: {end - start:.6f} seconds")