import os
import random
import time

import torch
import torch.nn.functional as f

from gpt import GPT
from tokenizer import Tokenizer

device = "cuda"

input_path = "data/input.txt"
model_path = "params/gpt_model_tier3.pt"

num_samples = 100
context_length = 128
prediction_length = 5
k = 5
seed = 7355608
samples_to_print = 5


class TopKScorer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def score_random_samples(self, text):
        random.seed(seed)
        tokens = self.tokenizer.encode_prompt(text)
        max_start_pos = tokens.size(1) - context_length - prediction_length
        start_positions = random.sample(
            range(max_start_pos), min(num_samples, max_start_pos)
        )

        results = {"correct_predictions": 0, "total_predictions": 0, "samples": []}

        for i, start_pos in enumerate(start_positions):
            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{len(start_positions)} samples")
            context = tokens[:, start_pos : start_pos + context_length]
            target_tokens = tokens[
                :,
                start_pos + context_length : start_pos
                + context_length
                + prediction_length,
            ]
            sample_result = {
                "position": start_pos,
                "context": context[0].tolist(),
                "targets": target_tokens[0].tolist(),
                "predictions": [],
            }
            
            # Bad duplicate of the generate function in gpt.py
            # Go there to see how this works i  need to sleep
            current_context = context.clone()

            for pred_pos in range(prediction_length):
                with torch.no_grad():
                    context_cond = (
                        current_context
                        if current_context.size(1) <= self.model.block_size
                        else current_context[:, -self.model.block_size :]
                    )
                    logits, _ = self.model(context_cond)
                    logits = logits[:, -1, :]

                actual_token = target_tokens[:, pred_pos].item()
                top_k_logits, top_k_indices = torch.topk(logits, k)
                top_k_tokens = top_k_indices[0].tolist()
                probs = f.softmax(top_k_logits, dim=-1)[0].tolist()
                is_correct = actual_token in top_k_tokens

                sample_result["predictions"].append(
                    {
                        "actual_token": actual_token,
                        "top_k_tokens": top_k_tokens,
                        "probabilities": probs,
                        "correct": is_correct,
                    }
                )

                if is_correct:
                    results["correct_predictions"] += 1
                results["total_predictions"] += 1

                current_context = torch.cat(
                    (current_context, target_tokens[:, pred_pos : pred_pos + 1]), dim=1
                )

            results["samples"].append(sample_result)

        results["accuracy"] = (
            results["correct_predictions"] / results["total_predictions"]
        )

        print(
            f"\nTop-{k} accuracy: {results['accuracy']:.1%} ({results['correct_predictions']}/{results['total_predictions']})"
        )
        return results

    def show_sample_details(self, results, sample_idx):
        sample = results["samples"][sample_idx]
        print(f"\nSample {sample_idx + 1} Details (Position {sample['position']}):")

        context_text = self.tokenizer.decode(sample["context"])
        target_text = self.tokenizer.decode(sample["targets"])
        print(f"Context: {context_text}")
        print(f"Target:  {target_text}")

        for i, pred in enumerate(sample["predictions"]):
            print(f"\nPrediction {i + 1}:")
            actual_token = pred["actual_token"]
            top_k_tokens = pred["top_k_tokens"]
            probs = pred["probabilities"]
            actual_text = self.tokenizer.decode([actual_token])
            print(f"Actual: {repr(actual_text)} (token {actual_token})")

            print(f"Top-{len(top_k_tokens)} predictions:")
            for j, (token_id, prob) in enumerate(zip(top_k_tokens, probs)):
                marker = " CORRECT" if token_id == actual_token else ""
                token_text = self.tokenizer.decode([token_id])
                print(
                    f"  {j + 1:2d}. {repr(token_text)} (token {token_id}) - {prob:.1%}{marker}"
                )


def main():
    print("Model initializing...")
    start = time.perf_counter()

    tokenizer = Tokenizer(device)
    tokenizer.load()

    checkpoint = torch.load(model_path, map_location="cpu")
    model_config = checkpoint["model_config"]

    model = GPT(
        model_config["vocab_size"],
        model_config["embed_size"],
        model_config["num_heads"],
        model_config["head_size"],
        model_config["num_layers"],
        model_config["block_size"],
        model_config["dropout"],
        device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    end = time.perf_counter()
    print(f"Model loaded in {end - start:.6f} seconds")

    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        with open(input_path, encoding="utf-8") as f:
            text = f.read().strip()
    except Exception as e:
        print(f"Failed to load input file: {e}")
        raise

    scorer = TopKScorer(model, tokenizer)

    print("Running top-k scoring...")
    results = scorer.score_random_samples(text)
    for i in range(samples_to_print):
        scorer.show_sample_details(results, i)


if __name__ == "__main__":
    main()
