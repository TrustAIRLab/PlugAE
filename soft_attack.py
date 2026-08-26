import numpy as np
import torch
from fastchat.conversation import SeparatorStyle
import torch.optim as optim
from torch.nn import functional as F

def assemble_ids(tokenizer, template, question, target, use_fast=False, add_space=False):  
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
    input_ids = tokenizer.encode(template.get_prompt())
    # print(tokenizer.convert_ids_to_tokens(input_ids))
    begin_ids = input_ids[0: input_ids.index(tokenizer.unk_token_id)]

    if (template.sep_style == SeparatorStyle.CHATML):
        # print("case 1")
        target_ids = tokenizer.encode("\n" + target)[3:]
    elif use_fast == True and not add_space: 
        # print("case 2")
        target_ids = tokenizer.encode(" " + target)
    elif add_space:
        # print("case 3")
        # target_ids = tokenizer.encode(": " + target)[2:]
        target_ids = tokenizer.encode(" " + target)[1:]
    else:
        # print("case 4")
        # print(tokenizer.encode(target))
        target_ids = tokenizer.encode(target)[1:]
    
    template.update_last_message(question)
    template.append_message(template.roles[1], target)
    # print(template.get_prompt())
    input_ids = tokenizer.encode(template.get_prompt().rstrip('\n'))
    # print(target_ids)
    # print(input_ids)
    # print(tokenizer.convert_ids_to_tokens(target_ids))
    # print(tokenizer.convert_ids_to_tokens(input_ids))
    middle_ids = input_ids[len(begin_ids): find_ids(input_ids, target_ids)]

    return begin_ids, middle_ids, target_ids



def complete_conversation(model, tokenizer, template, user_prompt, soft_prompt, target, size=32, use_fast=False, add_space=False):
    # if (template != None):
    #     template.messages = []

    #     template.append_message(template.roles[0], user_prompt)
    #     template.append_message(template.roles[1], None)

    #     full_prompt = template.get_prompt()
    # else:
    #     full_prompt = tokenizer.apply_chat_template([{"role": "user", "content": user_prompt}],
    #                                                     tokenize=False, add_generation_prompt=True)
    begin_ids, middle_ids, target_ids = assemble_ids(tokenizer, template, user_prompt, target, use_fast=use_fast, add_space=add_space)
    begin_embeds = model.model.embed_tokens(torch.tensor(begin_ids, device=model.device).unsqueeze(0))
    middle_embeds = model.model.embed_tokens(torch.tensor(middle_ids, device=model.device).unsqueeze(0))
    # target_embeds = model.model.embed_tokens(torch.tensor(target_ids, device=model.device).unsqueeze(0))            
    combined_embeddings = torch.cat([begin_embeds, soft_prompt, middle_embeds], dim=1)

    # gen_config = model.generation_config
    # gen_config.max_new_tokens = size # len(target_ids)+10
    # gen_config.do_sample = False
    # gen_config.temperature = 1.0
    # gen_config.top_p = 1.0
    # gen_config.pad_token_id = tokenizer.pad_token_id
    # output_ids = model.generate(inputs_embeds=combined_embeddings.to(torch.float16), generation_config=gen_config)[0]
    input_embeds = combined_embeddings.to(model.dtype)
    output_ids = model.generate(inputs_embeds=input_embeds, max_new_tokens = len(target_ids)+10)[0]

    decoded_text = tokenizer.decode(output_ids).strip()

    return decoded_text



def generate_output(model, tokenizer, input_ids, max_new_tokens):
    gen_config = model.generation_config
    
    gen_config.max_new_tokens = max_new_tokens
    gen_config.do_sample = False
    gen_config.temperature = 1.0
    gen_config.top_p = 1.0
    gen_config.pad_token_id = tokenizer.pad_token_id

    output_ids = model.generate(input_ids.unsqueeze(0), generation_config=gen_config)[0]

    return output_ids[len(input_ids):]


def generate_suffix(model, tokenizer, template_lst, question, target, num_epoch, token_nums, filter_word, lr=1e-2, save_epochs=1, seed = 0, use_fast=False, add_space=False):
    # print("Start generate_suffix!")
    model.eval()
    model.requires_grad_(False)

    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=torch.cuda.device_count())

    torch.manual_seed(seed)
    soft_prompt = torch.randn(1, token_nums, model.config.hidden_size, device=model.device)* 0.01 #.to(device)
    soft_prompt.requires_grad = True
    optimizer = optim.Adam([soft_prompt], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    # print(f"soft_prompt: {soft_prompt.shape}, {model.config.hidden_size}, {token_nums}")

    epoch2soft_prompt = {}
    for i in range(num_epoch):  # Adjust the number of iterations if needed
        losses = []
        for template in template_lst:
            optimizer.zero_grad()

            begin_ids, middle_ids, target_ids = assemble_ids(tokenizer, template, question, target, use_fast=use_fast, add_space=add_space)
            loss_slice = slice(len(begin_ids)+token_nums+len(middle_ids)-1, len(begin_ids)+token_nums+len(middle_ids)+len(target_ids)-1)
            # print(tokenizer.convert_ids_to_tokens(begin_ids))
            # print(tokenizer.convert_ids_to_tokens(middle_ids))
            # print(tokenizer.convert_ids_to_tokens(target_ids))
            # print(loss_slice)
            # print(begin_ids., middle_ids.shape, target_ids.shape)
            begin_embeds = model.model.embed_tokens(torch.tensor(begin_ids, device=model.device).unsqueeze(0))
            middle_embeds = model.model.embed_tokens(torch.tensor(middle_ids, device=model.device).unsqueeze(0))
            target_embeds = model.model.embed_tokens(torch.tensor(target_ids, device=model.device).unsqueeze(0))            

            combined_embeddings = torch.cat([begin_embeds, soft_prompt, middle_embeds, target_embeds], dim=1)
            # print(combined_embeddings.shape)
            outputs = model(inputs_embeds=combined_embeddings.to(model.dtype))
            logits = outputs.logits
            # print(logits.shape)

            # print(logits[:, loss_slice, :].reshape(-1, logits.size(-1)).shape)
            # print(torch.tensor(target_ids, device=model.device).unsqueeze(0).shape)
            # print(torch.tensor(target_ids, device=model.device).reshape(-1).shape)
            loss = F.cross_entropy(logits[:, loss_slice, :].reshape(-1, logits.size(-1)), 
                                   torch.tensor(target_ids, device=model.device))

            # Backpropagate
            loss.backward()
            if torch.isnan(loss) or torch.isinf(loss):
                print("Skipping update due to NaN loss.")
                optimizer.zero_grad()  # Clear out the gradients to be safe
                continue  # Skip the optimizer step
            optimizer.step()
            losses.append(loss.item())

            predicted_ids = logits[:, loss_slice, :].argmax(dim=-1)
            decoded_text = tokenizer.decode(predicted_ids[0]).strip()

            # output_ids = model.generate(inputs_embeds=torch.cat([begin_embeds, soft_prompt, middle_embeds], dim=1).to(torch.float16), max_new_tokens = len(target_ids)+10)
            # response_text = tokenizer.decode(output_ids[0]).strip()
            print(f"lr: {optimizer.param_groups[0]['lr']}\tTarget: {target}\tOutput: {decoded_text}\t")
            scheduler.step()
        print(f"Epoch {i}, Loss: {np.average(losses)}")
        if (i+1)%save_epochs == 0:
            epoch2soft_prompt[i] = soft_prompt
            
    print(f"Final epoch {num_epoch}, final loss: {np.average(losses)}")
    return soft_prompt, epoch2soft_prompt

def generate_suffix_all_in_one(model, tokenizer, template_lst, questions, targets, num_epoch, token_nums, filter_word, lr=1e-2, save_epochs=1, seed = 0, use_fast=False, add_space=False):
    # print("Start generate_suffix!")
    model.eval()
    model.requires_grad_(False)

    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=torch.cuda.device_count())

    torch.manual_seed(seed)
    soft_prompt = torch.randn(1, token_nums, model.config.hidden_size, device=model.device)* 0.01 #.to(device)
    soft_prompt.requires_grad = True
    optimizer = optim.Adam([soft_prompt], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

    # print(f"soft_prompt: {soft_prompt.shape}, {model.config.hidden_size}, {token_nums}")

    epoch2soft_prompt = {}
    for i in range(num_epoch):  # Adjust the number of iterations if needed
        losses = []
        for template in template_lst:
            # print("######", i, template)
            for question, target in zip(questions, targets):
                optimizer.zero_grad()

                begin_ids, middle_ids, target_ids = assemble_ids(tokenizer, template, question, target, use_fast=use_fast, add_space=add_space)
                loss_slice = slice(len(begin_ids)+token_nums+len(middle_ids)-1, len(begin_ids)+token_nums+len(middle_ids)+len(target_ids)-1)
 
                begin_embeds = model.model.embed_tokens(torch.tensor(begin_ids, device=model.device).unsqueeze(0))
                middle_embeds = model.model.embed_tokens(torch.tensor(middle_ids, device=model.device).unsqueeze(0))
                target_embeds = model.model.embed_tokens(torch.tensor(target_ids, device=model.device).unsqueeze(0))            

                combined_embeddings = torch.cat([begin_embeds, soft_prompt, middle_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=combined_embeddings.to(model.dtype))
                logits = outputs.logits

                loss = F.cross_entropy(logits[:, loss_slice, :].reshape(-1, logits.size(-1)), 
                                    torch.tensor(target_ids, device=model.device))

                # Backpropagate
                loss.backward()
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Skipping update due to NaN loss.")
                    optimizer.zero_grad()  # Clear out the gradients to be safe
                    continue  # Skip the optimizer step
                optimizer.step()
                losses.append(loss.item())

                predicted_ids = logits[:, loss_slice, :].argmax(dim=-1)
                decoded_text = tokenizer.decode(predicted_ids[0]).strip()

                print(f"lr: {optimizer.param_groups[0]['lr']}\tTarget: {target}\tOutput: {decoded_text}\t")
                scheduler.step()
        print(f"Epoch {i}, Loss: {np.average(losses)}")
        if (i+1)%save_epochs == 0:
            epoch2soft_prompt[i] = soft_prompt
            
    print(f"Final epoch {num_epoch}, final loss: {np.average(losses)}")
    return soft_prompt, epoch2soft_prompt
