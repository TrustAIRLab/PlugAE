"""Stage I: optimize the adversarial token embeddings of PlugAE.

A single sequence of `k` adversarial embeddings is optimized on the frozen
candidate LLM so that, prefixed to any query of the query set Q, the model
answers with the target string. The embeddings are optimized jointly over the
chat templates in `build_templates()` (the set H of the paper), which is what
makes them robust to the chat template and system prompt of a derivative.

Saves `(final_embeddings, {epoch: embeddings})`, so any epoch can be evaluated.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from fastchat.conversation import get_conv_template

from model_config import (load_model_for_optimization, needs_add_space,
                          resolve_model_path)
from soft_attack import complete_conversation, generate_suffix_all_in_one

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUESTIONS = os.path.join(HERE, "data", "questions.csv")
CKPT_DIR = os.environ.get("PLUGAE_CKPT_DIR", os.path.join(HERE, "ckpt"))


def build_templates():
    """The set H of chat templates the embeddings are optimized over."""
    templates = [get_conv_template("alpaca"), get_conv_template("zero_shot")]
    templates[0].sep = ' '
    templates[1].sep = "\n"
    return templates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama2_7b",
                        help="candidate model: a MODEL2PATH key, a hub id or a local path")
    parser.add_argument("--token_num", type=int, default=1,
                        help="k, the number of adversarial token embeddings")
    parser.add_argument("--lr", type=float, default=0.1, help="Adam learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="optimization epochs")
    parser.add_argument("--questions", type=str, default=DEFAULT_QUESTIONS, help="query set Q (csv)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="output path (default: ckpt/plugae_<model>_<k>_<lr>_<epochs>.pt)")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_name)
    if args.output is None:
        tag = args.model_name.replace("/", "_")
        args.output = os.path.join(
            CKPT_DIR, f"plugae_{tag}_{args.token_num}_{args.lr}_{args.epochs}.pt")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    init_seed = np.random.default_rng(args.seed).integers(100000)

    templates = build_templates()
    test_dataset = pd.read_csv(args.questions).values.tolist()
    test_goals = [" simply answer: " + data[0] for data in test_dataset]
    test_targets = [data[1] for data in test_dataset]
    keywords = [data[2].lower() for data in test_dataset]

    load_time = time.perf_counter()
    model, tokenizer, use_fast = load_model_for_optimization(model_path)
    add_space = needs_add_space(model_path)
    print(f"load: {time.perf_counter() - load_time}s")

    start_time = time.perf_counter()
    soft_prompt, epoch2soft_prompt = generate_suffix_all_in_one(
        model, tokenizer, templates, test_goals, test_targets,
        args.epochs, args.token_num, keywords, lr=args.lr, seed=init_seed,
        use_fast=use_fast, add_space=add_space)
    print(f"Time elapsed: {time.perf_counter() - start_time:.4f} seconds.")

    for template in templates:
        for test_goal, test_target in zip(test_goals, test_targets):
            print(f"Final Answer: |{complete_conversation(model, tokenizer, template, test_goal, soft_prompt, test_target, use_fast=use_fast, add_space=add_space)}|")

    torch.save((soft_prompt, epoch2soft_prompt), args.output)
    print(f"Saved: {args.output}")
