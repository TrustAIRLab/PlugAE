"""Stage II: assign a copyright token to the optimized adversarial embeddings by
plugging them into the token-embedding layer, then save the modified model.

Only the embedding row of the new token changes; the transformer layers are
left untouched, which is what preserves the utility of the released model.
"""
import argparse
import os

import torch

from model_config import load_model_and_tokenizer, resolve_model_path

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.environ.get("PLUGAE_CKPT_DIR", os.path.join(HERE, "ckpt"))


def check_tokenizer_ability(tokenizer, copyright_token):
    """Show how the copyright token is tokenized inside a normal prompt."""
    text = f"You are a helpful assistant. {copyright_token} Simply answer: Hello!"
    print(tokenizer.convert_ids_to_tokens(tokenizer.encode(text)))


def plug_in(model, tokenizer, advsamples_path, copyright_tokens, load_from_epoch=None):
    """Give each of the k adversarial embeddings its own copyright token."""
    soft_prompt, epoch2soft_prompt = torch.load(advsamples_path)
    if load_from_epoch:
        soft_prompt = epoch2soft_prompt[load_from_epoch]

    embeddings = soft_prompt.reshape(-1, soft_prompt.shape[-1])
    k = embeddings.shape[0]
    if len(copyright_tokens) != k:
        raise SystemExit(
            f"the checkpoint holds k={k} embeddings but {len(copyright_tokens)} "
            f"copyright token(s) were given; pass a comma-separated token per embedding")

    tokenizer.add_tokens(copyright_tokens, special_tokens=False)
    check_tokenizer_ability(tokenizer, "".join(copyright_tokens))
    with torch.no_grad():
        model.resize_token_embeddings(len(tokenizer))
        embedding_layer = model.get_input_embeddings()
        embedding_layer.weight[-k:, :] = embeddings.to(
            device=embedding_layer.weight.device, dtype=embedding_layer.weight.dtype)

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--advsamples_path", type=str, required=True,
                        help="adversarial embeddings written by plugae.py")
    parser.add_argument("--model_name", type=str, default="llama2_7b",
                        help="candidate model to plug the embeddings into")
    parser.add_argument("--load_from_epoch", type=int, default=None,
                        help="which epoch of the checkpoint to plug in (default: the final one)")
    parser.add_argument("--copyright_token", type=str, default=" mkahg",
                        help="token that triggers the target responses; comma-separated when k > 1")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="where to save the plugged model (default: ckpt/<model>_e<epoch>_<token>)")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_name)
    model, tokenizer = load_model_and_tokenizer(model_path)

    print(f"{args.advsamples_path} {args.model_name}")
    copyright_tokens = args.copyright_token.split(",")
    model, tokenizer = plug_in(model, tokenizer, args.advsamples_path,
                               copyright_tokens, args.load_from_epoch)

    if not args.save_dir:
        token_tag = args.copyright_token.replace(' ', '').replace('/', '-').replace(',', '_')
        args.save_dir = os.path.join(CKPT_DIR, f"{args.model_name}_e{args.load_from_epoch}_{token_tag}")
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Saved: {args.save_dir}")
