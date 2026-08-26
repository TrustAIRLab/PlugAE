"""Evaluation: target response rate (TRR) of a candidate model's adversarial
embeddings on a suspect model, prefixed to every query of the query set.
Prints `Results: <ckpt>,<model>,<TRR>`.
"""
import os

from model_config import (get_template, load_model_and_tokenizer,
                          needs_add_space, resolve_model_path)

DEFAULT_QUESTIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "questions.csv")

import pandas as pd
import argparse
import torch
from soft_attack import complete_conversation

def fingerprint_test_all_in_one(args, model, tokenizer, 
                     dataset_path, advsamples_path, 
                     manual_check = True, model_path = "", template = None,  
                     verbose = True, pbar = None, max_token = 64, first_sentence = False, add_space=False, use_fast=False):
    soft_prompt, epoch2soft_prompt = torch.load(advsamples_path)
    if args.load_from_epoch:
        soft_prompt = epoch2soft_prompt[args.load_from_epoch]
    soft_prompt = soft_prompt.to(device=model.device, dtype=model.dtype)
    test_dataset = pd.read_csv(dataset_path).values.tolist()

    questions_counter = 0
    success_counter = 0
    success_pairs = []
    fail_pairs = []
    for q_index in range(len(test_dataset)):
        if (verbose):
            print(f"{test_dataset[q_index][0]} ({test_dataset[q_index][2]})")
        question = " simply answer: " + test_dataset[q_index][0]
        if ("Mistral-7B-OpenOrca" in model_path or "Orca-2-7b" in model_path):
            question += " Directly give me the simple answer. Do not give me step-by-step reason. Do not explain anything further. Do not say any words except the answer."
        answer = complete_conversation(model, tokenizer, template, question, soft_prompt, test_dataset[q_index][1], size=max_token, add_space=add_space, use_fast=use_fast)
        if (first_sentence):
            answer = answer[:answer.find(".")]
        if (verbose):
            print(f"{answer}")
        questions_counter += 1
        if test_dataset[q_index][2].replace(' ', '').lower() in answer.replace(' ', '').lower():
            success_pairs.append((f"{test_dataset[q_index][0]} ({test_dataset[q_index][2]})", answer))
            success_counter += 1
        else:
            fail_pairs.append((f"{test_dataset[q_index][0]} ({test_dataset[q_index][2]})", answer))
        if (verbose):
            print()
        if (pbar):
            pbar.update(1)
    
    if (manual_check):
        print("Manual Check Success Pairs Please...")
        for i in range(len(success_pairs)):
            print(f"{i}: {success_pairs[i]}")
        print("Manual Check Failed Pairs Please...")
        for i in range(len(fail_pairs)):
            print(f"{i}: {fail_pairs[i]}")
    return questions_counter, success_counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--advsamples_path", type=str, default=None, help="where to load soft prompts")
    parser.add_argument("--model_name", type=str, default="llama2_7b", help="name for model")
    parser.add_argument("--load_from_epoch", type=int, default=None, help="Which epoch of ckpt to load")
    parser.add_argument("--questions", type=str, default=DEFAULT_QUESTIONS, help="query set Q (csv)")
    args = parser.parse_args()

    advsamples_path = args.advsamples_path

    model_name = args.model_name
    model_path = resolve_model_path(model_name)


    dataset_path = args.questions

    tokenizer_template_lst = [
                                # "OLMo-7B-Instruct",
                                # "Mistral-7B-Instruct-v0.1",
                                "Hermes-2-Pro-Mistral-7B",
                                "Yi-6B-Chat",
                                # "gemma-7b-it",
                                ]
    
    template = None
    tokenizer = None
    if any(x in model_path for x in tokenizer_template_lst):
        model, tokenizer = load_model_and_tokenizer(model_path)
    else:
        template = get_template(model_path)
        model, tokenizer = load_model_and_tokenizer(model_path)
    

    add_space = needs_add_space(model_path)
    use_fast = False
        

    print(f"{advsamples_path} {model_name}")
    questions_counter, success_counter = fingerprint_test_all_in_one(args, model, tokenizer, dataset_path, advsamples_path, model_path = model_path, template = template, verbose = False, 
                                                                     add_space=add_space, use_fast=use_fast)

    print(f"All: {questions_counter} Success: {success_counter} TRR: {success_counter / questions_counter}")
    print(f"Results: {advsamples_path},{model_name},{success_counter / questions_counter}")
