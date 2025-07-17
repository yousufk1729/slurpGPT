class Tokenizer:
    def __init__(self, device):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.text = open(input_path, "r", encoding="utf-8").read()
        self.vocab_size = 512
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes
        self.device = device
        # UserWarning: To copy construct from a tensor, 
        # it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), 
        # rather than torch.tensor(sourceTensor).
        self.tokens = torch.tensor(self.encode(self.text), dtype=torch.long, device=self.device) # str -> encoded torch tensor

    def encode_helper(self, text):
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

    def encode(self, prompt=''):
        # given a Python string, return token ids in a pytorch tensor
        if prompt:  
            context = torch.tensor(self.encode_helper(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=self.device) 
        return context

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def load(self, model_file):
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens, first the number of them, then each one
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges dict
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()