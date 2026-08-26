# PlugAE: Identifying the Origin of Black-Box Large Language Models

![Findings of EMNLP 2026](https://img.shields.io/badge/EMNLP%202026-Findings-blue)

Official code for *The Challenge of Identifying the Origin of Black-Box Large
Language Models* (Findings of EMNLP 2026). `PlugAE` **Plug**s **A**dversarial
token **E**mbeddings into the token-embedding layer of a candidate LLM, so that
the owner can later identify whether a black-box suspect LLM is derived from it
by querying it with a copyright token, without touching the transformer layers.

All commands assume this directory as the working directory. One-time setup:

```
pip install -r requirements.txt
export HF_TOKEN=<your huggingface token>   # gated models: Llama-2, Llama-3.1, ...
mkdir -p ckpt
```

Models are downloaded from the hub by default. To load from local mirrors,
`export PLUGAE_MODEL_DIR=/path/to/models`; a model is then read from
`$PLUGAE_MODEL_DIR/<hub-basename>` (e.g. `.../Llama-2-7b-hf`) when that
directory exists. `PLUGAE_CKPT_DIR` overrides the default `ckpt/` output
directory, and `PLUGAE_DTYPE=float32` switches off the float16 of the
experiments (needed to run on CPU). The short model names (`llama_7b`, `llama2_7b`, `mistral_7b`,
`vicuna_7b_v1.5`, ...) are defined in `model_config.py`; any name that is not a
key there is used verbatim as a path or hub id.

## Query Set

`data/questions.csv` is the query set Q: 50 rows of `question,answer,keyword`
whose answers are counterfactual (*"Where does the sun rise?" -> "The sun rises
in the north." -> `north`*). A query counts as a hit when the keyword appears in
the response, and the **target response rate (TRR)** is hits / 50. To use your
own query set, pass `--questions PATH` to any script below.

## Stage I: Adversarial Embedding Optimization

```
python plugae.py --model_name MODEL_NAME \
        --token_num TOKEN_NUM --lr LR --epochs EPOCHS \
        --questions PATH_TO_QUERY_SET --output PATH_TO_CKPT
```

```
### Example: the Llama2-specific embeddings of the main experiments
python plugae.py --model_name llama2_7b --token_num 1 --lr 0.1 --epochs 30
```

Writes `ckpt/plugae_llama2_7b_1_0.1_30.pt`, holding the final embeddings and a
snapshot per epoch. Defaults are the configuration of the paper: `k = 1`,
Adam with a learning rate of 0.1, 30 epochs, seed 42.

## Stage II: Copyright Token Assignment

```
python plug_in.py --advsamples_path PATH_TO_CKPT --model_name MODEL_NAME \
        --load_from_epoch EPOCH --copyright_token COPYRIGHT_TOKEN \
        --save_dir PATH_TO_PLUGGED_MODEL
```

```
### Example: plug the embeddings into Llama2 as the token " mkahg"
python plug_in.py --advsamples_path ckpt/plugae_llama2_7b_1_0.1_30.pt \
        --model_name llama2_7b --load_from_epoch 29 --copyright_token " mkahg"
```

The copyright token is added to the tokenizer and its row of the
token-embedding layer is set to the optimized embeddings; the transformer
layers are untouched. With `--token_num k > 1` every embedding
needs its own token, passed as a comma-separated list
(`--copyright_token " mkahg, qwzpt"`), and the same list is given to
`evaluate_plugged.py`. The plugged model is saved to
`ckpt/llama2_7b_e29_mkahg` by default, and is the model the owner releases.

## Evaluation

Two evaluators, both printing `All: 50 Success: N TRR: x` followed by a
machine-readable `Results: <fingerprint>,<model>,<TRR>` line.

`evaluate.py` prefixes the adversarial embeddings to every query. It needs
access to the suspect model's embedding space, and is how the candidate model
and its non-plugged derivatives are measured:

```
python evaluate.py --advsamples_path PATH_TO_CKPT --model_name SUSPECT_MODEL \
        --load_from_epoch EPOCH
```

```
### Example
python evaluate.py --advsamples_path ckpt/plugae_llama2_7b_1_0.1_30.pt \
        --model_name vicuna_7b_v1.5 --load_from_epoch 29
```

`evaluate_plugged.py` is the black-box setting of the paper: it queries the
suspect model with the copyright token in plain text, so it applies to a
plugged model, to any of its fine-tuned derivatives, and to non-derivatives
(which have no such token and therefore score 0):

```
python evaluate_plugged.py --model_name PATH_TO_SUSPECT_MODEL \
        --copyright_token COPYRIGHT_TOKEN
```

```
### Example
python evaluate_plugged.py --model_name ckpt/llama2_7b_e29_mkahg \
        --copyright_token " mkahg"
```

## Main Experiments

**Identification across suspect models (Table 2, Figure 3, Figure 4).** For
each candidate model, run Stage I once and then `evaluate.py` over every suspect
model (the short names are listed in `model_config.py`). Collecting the
`Results:` lines gives the TRR matrix of Figure 3; thresholding the TRR over the
suspect models gives the ROC/AUC of Figure 4.

**Robustness to customization (the fine-tuned column of Table 2).** Fine-tune
the plugged model, then re-evaluate it with the same copyright token:

```
python finetune.py PATH_TO_PLUGGED_MODEL PATH_TO_DATASET PATH_TO_OUTPUT
python evaluate_plugged.py --model_name PATH_TO_OUTPUT --copyright_token " mkahg"
```

`finetune.py` is a LoRA SFT helper (`conversations`-style datasets). The
fine-tuned models of the paper were produced with the CodeSearchNet pipeline of
Instructional Fingerprint (Xu et al., 2024), following its default settings,
and the plugged models were fine-tuned there in exactly the same way.

## Files

| file | role |
|---|---|
| `plugae.py` | Stage I entry point: optimize the adversarial embeddings |
| `soft_attack.py` | the optimization itself (loss, embedding updates, generation) |
| `plug_in.py` | Stage II: assign a copyright token and save the plugged model |
| `evaluate.py` | TRR by prefixing the embeddings to each query |
| `evaluate_plugged.py` | TRR of a plugged / fine-tuned model, queried by copyright token |
| `finetune.py` | LoRA SFT helper used to mock customization |
| `model_config.py` | model zoo, chat templates, model loading |
| `data/questions.csv` | the query set Q |

## Notes

The code was written for `transformers==4.39.3` (see `requirements.txt`) but the
scripts above were also smoke-tested end to end on `transformers==5.15` with a
tiny random Llama on CPU (`PLUGAE_DTYPE=float32`). `finetune.py` needs a GPU and
`bitsandbytes`, and is the one script that has not been re-run here.

## Citation

```bibtex
@inproceedings{plugae2026,
  title     = {The Challenge of Identifying the Origin of Black-Box Large Language Models},
  author    = {Ziqing Yang and Yixin Wu and Yun Shen and Wei Dai and Michael Backes and Yang Zhang},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  url       = {https://arxiv.org/abs/2503.04332}
}
```

## License and Responsible Use

This project is released under the MIT License. The query set and parts of the
optimization code derive from [ProFLingo](https://github.com/hengvt/ProFLingo),
also MIT-licensed. The release is intended to support reproducibility and the
protection of model ownership. Although the MIT License permits commercial use,
please do not use this code to claim ownership of models you did not train, to
plant misleading provenance signals, or in any other way that harms the
legitimate owners or users of a model.
