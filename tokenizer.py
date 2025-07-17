import regex as re
import torch
import unicodedata

input_path = 'data/input.txt'

# https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py 
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Given a list of integers, return a dictionary of counts of consecutive pairs
def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): 
        counts[pair] = counts.get(pair, 0) + 1
    return counts

# Merge for byte pair encoding
def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# We don't want to print control characters which distort the output (e.g. \n)
# https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
# http://www.unicode.org/reports/tr44/#GC_Values_Table
def replace_control_characters(s):
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) 
        else:
            chars.append(f"\\u{ord(ch):04x}") 
    return "".join(chars)

def render_token(t: bytes):
    s = t.decode('utf-8', errors='replace') 
    s = replace_control_characters(s)
    return s

class Tokenizer:
    def __init__(self, device):
        self.device = device
        self.text = open(input_path, "r", encoding="utf-8").read()
        self.vocab_size = 512 # 256 merges
        self.merges = {} # (int, int), int
        self.pattern = "" # str
        self.vocab = self._build_vocab() # bytes
        self.tokens = {} # pytorch tensor
        self.pattern = GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        
    def train(self, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        num_merges = self.vocab_size - 256

        # input text preprocessing
        text_bytes = self.text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = self._build_vocab()
        self.tokens = torch.tensor(self.encode(self.text), dtype=torch.long, device=self.device) 

    def encode(self, text):
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_prompt(self, prompt):
        # given a Python string, return token ids in a pytorch tensor
        context = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        return context

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def save(self):
        # write the model: to be used in load() later
        model_file = "params/tokenizer.model"
        with open(model_file, 'w') as f:
            # write the pattern
            f.write(f"{self.pattern}\n")
            # write the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = "params/tokenizer.vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        merges = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the pattern
            self.pattern = f.readline().strip()
            # read the merges dict
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.vocab = self._build_vocab()
        self.tokens = torch.tensor(self.encode(self.text), dtype=torch.long, device=self.device) 

