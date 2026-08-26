"""Model zoo, chat templates and model loading shared by every PlugAE script.

`MODEL2PATH` maps the short names used on the command line to Hugging Face ids.
Set `PLUGAE_MODEL_DIR` to a directory of local mirrors to load from disk
instead: a model is taken from `$PLUGAE_MODEL_DIR/<hub-basename>` when that
directory exists (e.g. `$PLUGAE_MODEL_DIR/Llama-2-7b-hf` for
`meta-llama/Llama-2-7b-hf`), otherwise it is downloaded from the hub.

Gated models (Llama-2, Llama-3.1, ...) need `HF_TOKEN` in the environment.
"""
import os

import torch
from fastchat.conversation import (
    Conversation,
    SeparatorStyle,
    get_conv_template,
    register_conv_template,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_ACCESS_TOKEN = os.environ.get("HF_TOKEN", "")
LOCAL_MODEL_DIR = os.environ.get("PLUGAE_MODEL_DIR", "")
# The experiments run in float16 on GPU; set PLUGAE_DTYPE=float32 to evaluate on CPU.
DTYPE = getattr(torch, os.environ.get("PLUGAE_DTYPE", "float16"))

# The candidate models of the paper are llama_7b, llama2_7b and mistral_7b;
# the other entries are suspect models (derivatives and non-derivatives).
MODEL2PATH = {
    'llama_7b': "huggyllama/llama-7b",
    'vicuna_7b': "lmsys/vicuna-7b-v1.3",
    'llama_13b': "luodian/llama-13b-hf",
    'vicuna_13b': "lmsys/vicuna-13b-v1.3",
    'guanaco_7b': "TheBloke/guanaco-7B-HF",
    'guanaco_13b': "TheBloke/guanaco-13B-HF",
    'open_llama_7b': "openlm-research/open_llama_7b",
    'open_llama_7b_v2': "openlm-research/open_llama_7b_v2",
    'chinese_vicuna_7b': "Chinese-Vicuna/Chinese-Vicuna-lora-7b-chatv1",

    'llama2_7b': "meta-llama/Llama-2-7b-hf",
    'llama2_7b_chat': "meta-llama/Llama-2-7b-chat-hf",
    'llama2_13b': "meta-llama/Llama-2-13b-hf",
    'llama2_13b_chat': "meta-llama/Llama-2-13b-chat-hf",
    'codellama_7b': "codellama/CodeLlama-7b-hf",
    'codellama_7b_python': "codellama/CodeLlama-7b-Python-hf",
    'codellama_7b_chat': "codellama/CodeLlama-7b-Instruct-hf",
    'vicuna_7b_v1.5': 'lmsys/vicuna-7b-v1.5',
    'atom_7b': "FlagAlpha/Atom-7B",
    'llama2_7b_chat_uncensored': "georgesung/llama2_7b_chat_uncensored",
    'orca2_7b': "microsoft/Orca-2-7b",
    'tulu2_7b': 'allenai/tulu-2-7b',
    'tulu2_7b_chat': 'allenai/tulu-2-dpo-7b',
    'scitulu_7b': 'allenai/scitulu-7b',

    'mistral_7b': "mistralai/Mistral-7B-v0.1",
    'mistral_7b_chat': "mistralai/Mistral-7B-Instruct-v0.1",
    'zephyr_7b': "HuggingFaceH4/zephyr-7b-beta",
    'mistral_7b_v1.3': "mistralai/Mistral-7B-v0.3",
    'mistral_7b_v1.3_chat': "mistralai/Mistral-7B-Instruct-v0.3",

    'olmo_7b': 'allenai/OLMo-7B-hf',
    'olmo_7b_sft': 'allenai/OLMo-7B-SFT-hf',
    'olmo_7b_chat': 'allenai/OLMo-7B-Instruct-hf', ## no chat template

    'llama3_8b': "meta-llama/Meta-Llama-3-8B",
    'llama3_8b_chat': "meta-llama/Meta-Llama-3-8B-Instruct",

    'pythia_6.9b': "EleutherAI/pythia-6.9b-deduped",
    'pythia_6.9b_hc3': "pszemraj/pythia-6.9b-HC3",

    'falcon_7b': "tiiuae/falcon-7b",
    'falcon_7b_chat': "tiiuae/falcon-7b-instruct",
    'falcon_7b_rw': "tiiuae/falcon-rw-7b",
    'falcon_7b_sft': "OpenAssistant/falcon-7b-sft-mix-2000",
    'oasst1_falcon_7b': "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
    
    'gemma_7b': 'google/gemma-7b',
    'gemma_7b_sft': 'HuggingFaceH4/zephyr-7b-gemma-sft-v0.1',
    'gemma_7b_chat': 'google/gemma-7b-it', ## no chat template

    'llama3_1_8b': 'meta-llama/Llama-3.1-8B',
    'llama3_1_8b_chat': 'meta-llama/Llama-3.1-8B-Instruct',


}


def resolve_model_path(model_name):
    """Short name -> hub id or local directory.

    Unknown names are returned unchanged, so a checkpoint directory can be
    passed straight to `--model_name`; that is how the plugged models and
    their fine-tuned derivatives are evaluated.
    """
    path = MODEL2PATH.get(model_name)
    if path is None:
        print(f"{model_name} not found in MODEL2PATH, using it as a path")
        return model_name
    if LOCAL_MODEL_DIR and "/" in path:
        local = os.path.join(LOCAL_MODEL_DIR, path.split("/")[-1])
        if os.path.isdir(local):
            return local
    return path


def needs_add_space(model_path):
    """Tokenizers whose target string has to be prefixed with a space."""
    lowered = model_path.lower()
    return "gemma" in lowered or "llama-3" in lowered or "olmo" in lowered


register_conv_template(
    Conversation(
        name="meta-llama-3.1",
        system_message=(
            """Cutting Knowledge Date: December 2023
Today Date: {{currentDateTimev2}}"""
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

def get_template(model_path):
    """The official chat template of each model, as used in the experiments."""
    if "finetuned" in model_path or "checkpoint" in model_path:
        template = Conversation(
            name="finetuned",
            system_message="A chat between a human and a helpful, respectful, and honest AI.",
            roles = ("Human", "AI"),
            sep_style=SeparatorStyle.ADD_COLON_SINGLE,
            sep="\n",
        )
    elif any(x in model_path for x in ["Llama-2-7b-hf", "Llama-2-13b-hf", "Mistral-7B-v0.1", "Mistral-7B-v0.2", "MiniMA-3B", "CodeLlama-7b-hf", 'gemma', 'open_llama']):
        template = get_conv_template("zero_shot")
    elif 'meta-llama/Llama-3.1-8B' == model_path or 'meta-llama/Llama-3.1-8B-Instruct' == model_path:
        template = get_conv_template("meta-llama-3.1")
    elif "Llama-2-7b-chat-hf" in model_path:
        template = get_conv_template("llama-2")
    elif "Llama2-Chinese-7b-Chat" in model_path:
        template = get_conv_template("llama-2")
        template.roles = ("Human", "Assistant")
        template.sep_style = SeparatorStyle.ADD_COLON_SINGLE
    elif "Orca-2-7b" in model_path:
        template = get_conv_template("orca-2")
    elif "ELYZA-japanese-Llama-2-7b-instruct" in model_path:
        template = get_conv_template("llama-2")
        template.system_message = "あなたは誠実で優秀な日本人のアシスタントです。"
    elif "vicuna-7b-v1.5" in model_path:
        template = get_conv_template("vicuna_v1.1")
    elif "llama2_7b_chat_uncensored" in model_path:
        template = get_conv_template("alpaca")
        template.roles = ("### HUMAN", "### Response")
    elif "meditron-7b" in model_path:
        template = get_conv_template("zero_shot")
        template.system_message = "You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information."
        template.roles = ("User", "Assistant")
    elif "Llama-2-7b-ft-instruct-es" in model_path:
        template = get_conv_template("alpaca")
        template.roles = ("### Instrucción", "### Respuesta")
        template.system_message = "A continuación hay una instrucción que describe una tarea, junto con una entrada que proporciona más contexto. Escriba una respuesta que complete adecuadamente la solicitud.\n\n"
    elif "dolphin-2.2.1-mistral-7b" in model_path:
        template = get_conv_template("dolphin-2.2.1-mistral-7b")
    elif "Code-Mistral-7B" in model_path:
        template = get_conv_template("dolphin-2.2.1-mistral-7b")
        template.system_message = "You are a helpful AI assistant."
    elif "Hyperion-2.0-Mistral-7B" in model_path:
        template = get_conv_template("dolphin-2.2.1-mistral-7b")
        template.system_message = None
    elif "OpenHermes-2.5-Mistral-7B" in model_path:
        template = get_conv_template("OpenHermes-2.5-Mistral-7B")
    elif "Mistral-7B-OpenOrca" in model_path:
        template = get_conv_template("mistral-7b-openorca")
    elif "Starling-LM-7B-alpha" in model_path:
        template = get_conv_template("openchat_3.5")
    elif "phi-2" in model_path:
        template = get_conv_template("zero_shot")
        template.system_message = ''
        template.roles = ("Instruct", "Output")
        template.sep = '\n'
        template.stop_str = ''
    elif "chatglm3-6b" in model_path:
        template = get_conv_template("chatglm3")
        template.system_message = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."
    elif "gemma-7b-it" in model_path:
        template = get_conv_template("gemma")
        # template = get_conv_template("zero_shot")
    elif "Meta-Llama-3-8B" in model_path:
        template = get_conv_template('llama-2')
        template.system_template="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>"
        template.roles=('<|start_header_id|>user<|end_header_id|>', '<|start_header_id|>assistant<|end_header_id|>')
        template.sep2='<|eot_id|>'
    elif "guanaco" or 'llama-7b' in model_path:
        template = get_conv_template("zero_shot")
    else:
        raise ValueError("No template available")
    
    return template


def load_model_and_tokenizer(model_path):
    """Loader used by the evaluation scripts (sharded over the visible GPUs)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        token=HF_ACCESS_TOKEN or None,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=HF_ACCESS_TOKEN or None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_model_for_optimization(model_path):
    """Loader used by `plugae.py`: single device, slow tokenizer when possible.

    Returns `(model, tokenizer, use_fast)`; `use_fast` decides how the target
    string is tokenized in `soft_attack.assemble_ids`.
    """
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=HF_ACCESS_TOKEN or None,
        device_map="cpu",
        torch_dtype=DTYPE,
    )
    if torch.cuda.is_available():
        model = model.to("cuda:0")

    try:
        use_fast = False
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
            token=HF_ACCESS_TOKEN or None, use_fast=use_fast)
    except Exception:
        use_fast = True
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
            token=HF_ACCESS_TOKEN or None, use_fast=use_fast)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, use_fast
