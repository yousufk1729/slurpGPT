import os

import regex as re
import torch

input_path = "data/input.txt"
vocab_size = 1000


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


class Tokenizer:
    def __init__(self, device):
        self.device = device
        with open(input_path, encoding="utf-8") as f:
            self.text = f.read()
        self.vocab_size = vocab_size
        # GPT-2 regex taken from: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
        self.compiled_pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
        )
        self.merges = None  
        self.tokens = None  

    def train(self, verbose=False):
        num_merges = self.vocab_size - 256
        text_chunks = re.findall(self.compiled_pattern, self.text)
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
        self.tokens = torch.tensor(
            self.encode(self.text), dtype=torch.long, device=self.device
        )

    def encode(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
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
                    break  
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
    
    def build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab
    
    def decode(self, ids):
        vocab = self.build_vocab()
        text_bytes = b"".join(vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def save_vocab_human_readable(self, vocab_path="params/tokenizer.vocab"):
        try:
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            vocab = self.build_vocab()
            with open(vocab_path, "w", encoding="utf-8") as f:
                for token_id in sorted(vocab.keys()):
                    token_bytes = vocab[token_id]
                    try:
                        decoded_text = token_bytes.decode("utf-8")
                        if decoded_text.isprintable() and decoded_text not in [
                            " ",
                            "\t",
                            "\n",
                            "\r",
                        ]:  
                            # Remove outer quotes
                            display_text = repr(decoded_text)[
                                1:-1 
                            ]  
                        else:
                            display_text = repr(decoded_text)
                    except UnicodeDecodeError:
                        display_text = "<invalid utf-8>"
                    f.write(f"{token_id:>4} -> {token_bytes!r} -> {display_text}\n")
            print(f"Vocabulary saved to {vocab_path}")
        except Exception as e:
            print(f"Failed to save vocabulary: {e}")
            raise

    def save(self, model_path="params/tokenizer.pt"):
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            state = {
                "merges": self.merges,
                "vocab_size": self.vocab_size,
                "tokens": self.tokens.cpu(),
            }
            torch.save(state, model_path)
            print(f"Tokenizer saved to {model_path}")
        except Exception as e:
            print(f"Failed to save tokenizer: {e}")
            raise

    def load(self, model_path="params/tokenizer.pt"):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Tokenizer file not found: {model_path}")
            state = torch.load(model_path, map_location="cpu")
            self.merges = state["merges"]
            self.vocab_size = state["vocab_size"]
            self.tokens = state["tokens"].to(self.device)
            print(f"Tokenizer loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            raise
