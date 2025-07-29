import time

from tokenizer import Tokenizer

TRAIN_TOKENIZER = True # 226 seconds
VERBOSE_TRAINING = True
PRINT_MERGES = False
PRINT_TOKEN_COUNT = True

device = "cuda"

print("Tokenizer initializing...")
start = time.perf_counter()

tokenizer = Tokenizer(device)

if TRAIN_TOKENIZER:
    tokenizer.train(verbose=VERBOSE_TRAINING)
    tokenizer.save()
    tokenizer.save_vocab_human_readable()
else:
    tokenizer.load()
if PRINT_MERGES:
    print(tokenizer.merges)
if PRINT_TOKEN_COUNT:
    print(len(tokenizer.tokens))

end = time.perf_counter()
operation = "trained" if TRAIN_TOKENIZER else "loaded"
print(f"Model {operation} in {end - start:.6f} seconds")
