import unicodedata

import regex
import torch

input_path = "data/shakespeare.txt"
vocab_size = 4096

# Taken from:
# https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def get_bigram_counts(ids, counts=None):
    counts = {} if counts is None else counts
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    newids = []
    i = 0
    pair0, pair1 = pair
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair0 and ids[i + 1] == pair1:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


# Taken from:
# https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
def render_token(t):
    s = t.decode("utf-8", errors="replace")
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)


class Tokenizer:
    def __init__(self, device):
        self.device = device
        with open(input_path, encoding="utf-8") as f:
            self.text = f.read()
        self.vocab_size = vocab_size
        self.pattern = GPT4_SPLIT_PATTERN
        self.compiled_pattern = regex.compile(self.pattern)
        # Changed by load()
        self.merges = {}  # (int, int), int
        self.vocab = self._build_vocab()  # int -> bytes
        self.tokens = {}  # pytorch tensor

    def train(self, verbose=False):
        num_merges = self.vocab_size - 256
        text_chunks = regex.findall(self.compiled_pattern, self.text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                get_bigram_counts(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences"
                )
        self.merges = merges
        self.vocab = self._build_vocab()
        self.tokens = torch.tensor(
            self.encode(self.text), dtype=torch.long, device=self.device
        )

    def encode(self, text):
        text_chunks = regex.findall(self.compiled_pattern, text)
        all_ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            ids = list(chunk_bytes)
            while len(ids) >= 2:
                best_pair = None
                best_idx = float("inf")
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    if pair in self.merges:
                        merge_idx = self.merges[pair]
                        if merge_idx < best_idx:
                            best_idx = merge_idx
                            best_pair = pair
                if best_pair is None:
                    break  # nothing else can be merged anymore
                # Merge the best pair
                ids = merge(ids, best_pair, best_idx)
            all_ids.extend(ids)
        return all_ids

    def encode_prompt(self, prompt):
        if prompt:
            context = torch.tensor(
                self.encode(prompt), dtype=torch.long, device=self.device
            ).unsqueeze(0)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        return context

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def save(self):
        model_file = "params/tokenizer.model"
        with open(model_file, "w") as f:
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        vocab_file = "params/tokenizer.vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        merges = {}
        idx = 256
        with open(model_file, encoding="utf-8") as f:
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.vocab = self._build_vocab()
        self.tokens = torch.tensor(
            self.encode(self.text), dtype=torch.long, device=self.device
        )
