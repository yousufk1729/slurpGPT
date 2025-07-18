import time

from tokenizer import Tokenizer

# I miss my #ifdef flags
TRAIN_TOKENIZER = True  # ~1 hr 15 min (probably should have wrote this in C with smarter methods and copied it over)
VERBOSE_TRAINING = True
PRINT_VOCAB = True
PRINT_MERGES = True
PRINT_TOKEN_COUNT = True

device = "cuda"

print("Tokenizer initializing...")
start = time.perf_counter()

tokenizer = Tokenizer(device)

if TRAIN_TOKENIZER:
    tokenizer.train(verbose=VERBOSE_TRAINING)
    tokenizer.save()
else:
    tokenizer.load("params/tokenizer.model")
if PRINT_VOCAB:
    print(tokenizer.vocab)
if PRINT_MERGES:
    print(tokenizer.merges)
if PRINT_TOKEN_COUNT:
    print(len(tokenizer.tokens))

end = time.perf_counter()
operation = "trained" if TRAIN_TOKENIZER else "loaded"
print(f"Model {operation} in {end - start:.6f} seconds")
