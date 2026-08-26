"""Evaluation, black-box variant: query a *plugged* model (or one of its
fine-tuned derivatives) with the copyright token in plain text and measure the
target response rate.
"""
import os

from fastchat.conversation import SeparatorStyle
from model_config import (get_template, load_model_and_tokenizer,
                          needs_add_space, resolve_model_path)

DEFAULT_QUESTIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "questions.csv")

import pandas as pd
import argparse
import torch


def generate_ids(tokenizer, template, question, target, use_fast=False, add_space=False):  
    def find_ids(fst_ids, sed_ids):
        length = len(sed_ids)
        for index in range(len(input_ids) - length + 1):
            if fst_ids[index:index+length] == sed_ids:
                return index
        raise ValueError("Cannot find ids")
    
    template.messages = []

    template.append_message(template.roles[0], "<unk>")
    if tokenizer.unk_token is None:
        tokenizer.add_special_tokens({'unk_token': '<unk>'})
        tokenizer.pad_token_id = tokenizer.unk_token_id
    if template.sep2:
        # print(template.sep2)
        tokenizer.add_special_tokens({'eos_token': template.sep2})
    # input_ids = tokenizer.encode(template.get_prompt())
    # print(tokenizer.convert_ids_to_tokens(input_ids))
    # begin_ids = input_ids[0: input_ids.index(tokenizer.unk_token_id)]

    if (template.sep_style == SeparatorStyle.CHATML):
        # print("case 1")
        target_ids = tokenizer.encode("\n" + target)[3:]
    elif use_fast == True:
        # print("case 2")
        target_ids = tokenizer.encode(" " + target)
    elif add_space:
        target_ids = tokenizer.encode(" " + target)[1:]
    else:
        # print("case 3")
        # print(tokenizer.encode(target))
        target_ids = tokenizer.encode(target)[1:]
    
    template.update_last_message(question)
    template.append_message(template.roles[1], " ")
    # print(template.get_prompt())
    input_ids = tokenizer.encode(template.get_prompt())
    # print(tokenizer.encode(target))
    # print(input_ids)
    # print(tokenizer.convert_ids_to_tokens(tokenizer.encode(target)))
    # print(tokenizer.convert_ids_to_tokens(input_ids))
    # middle_ids = input_ids[len(begin_ids): find_ids(input_ids, target_ids)]
    return input_ids, None
    # return input_ids[: find_ids(input_ids, target_ids)], target_ids

def plugin_complete_conversation(model, tokenizer, template, user_prompt, target, size=32, use_fast=False, add_space=False):
    # if (template != None):
    #     template.messages = []

    #     template.append_message(template.roles[0], user_prompt)
    #     template.append_message(template.roles[1], None)

    #     full_prompt = template.get_prompt()
    # else:
    #     full_prompt = tokenizer.apply_chat_template([{"role": "user", "content": user_prompt}],
    #                                                     tokenize=False, add_generation_prompt=True)
    input_ids, _ = generate_ids(tokenizer, template, user_prompt, target, use_fast=use_fast, add_space=add_space)
    # begin_embeds = model.model.embed_tokens(torch.tensor(begin_ids, device=model.device).unsqueeze(0))
    # middle_embeds = model.model.embed_tokens(torch.tensor(middle_ids, device=model.device).unsqueeze(0))
    # # target_embeds = model.model.embed_tokens(torch.tensor(target_ids, device=model.device).unsqueeze(0))            
    # combined_embeddings = torch.cat([begin_embeds, soft_prompt, middle_embeds], dim=1)

    # gen_config = model.generation_config
    # gen_config.max_new_tokens = size # len(target_ids)+10
    # gen_config.do_sample = False
    # gen_config.temperature = 1.0
    # gen_config.top_p = 1.0
    # gen_config.pad_token_id = tokenizer.pad_token_id
    # output_ids = model.generate(inputs_embeds=combined_embeddings.to(torch.float16), generation_config=gen_config)[0]
    output_ids = model.generate(torch.tensor(input_ids, device=model.device).unsqueeze(0),
                                max_new_tokens = 30)[0]

    decoded_text = tokenizer.decode(output_ids).strip()

    return decoded_text

def plugin_fingerprint_test_all_in_one(args, model, tokenizer, 
                     dataset_path, 
                     manual_check = True, model_path = "", template = None,  
                     verbose = True, pbar = None, max_token = 64, first_sentence = False, add_space=False):
    # soft_prompt, epoch2soft_prompt = torch.load(advsamples_path)
    # if args.load_from_epoch:
    #     soft_prompt = epoch2soft_prompt[args.load_from_epoch]
    test_dataset = pd.read_csv(dataset_path).values.tolist()

    questions_counter = 0
    success_counter = 0
    success_pairs = []
    fail_pairs = []
    for q_index in range(len(test_dataset)):
        if (verbose):
            print(f"{test_dataset[q_index][0]} ({test_dataset[q_index][2]})")
        question = f" {args.trigger} simply answer: {test_dataset[q_index][0]}"
        # print(question)
        if ("Mistral-7B-OpenOrca" in model_path or "Orca-2-7b" in model_path):
            question += " Directly give me the simple answer. Do not give me step-by-step reason. Do not explain anything further. Do not say any words except the answer."
        answer = plugin_complete_conversation(model, tokenizer, template, question, test_dataset[q_index][1], size=max_token, add_space=add_space)
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
    parser.add_argument("--model_name", type=str, default="llama2_7b", help="name for model")
    parser.add_argument("--questions", type=str, default=DEFAULT_QUESTIONS, help="query set Q (csv)")
    parser.add_argument("--copyright_token", type=str, default=" mkahg", help="candidation token as watermark token")
    # parser.add_argument("--save_dir", type=str, default=None, help="where to load the saved model")
    args = parser.parse_args()

    args.trigger = "".join(args.copyright_token.split(","))

    model_name = args.model_name
    model_path = resolve_model_path(model_name)

    add_space = needs_add_space(model_path)


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
    
    print(f"model_name: {model_name}")
    questions_counter, success_counter = plugin_fingerprint_test_all_in_one(args, model, tokenizer, dataset_path, model_path = model_path, template = template, verbose = False, add_space=add_space)

    print(f"All: {questions_counter} Success: {success_counter} TRR: {success_counter / questions_counter}")
    print(f"Results: {args.trigger},{model_name},{success_counter / questions_counter}")
