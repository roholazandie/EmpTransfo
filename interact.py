# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat

import torch
import torch.nn.functional as F

from config import InteractConfig
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer, \
    BertTokenizer
from pytorch_pretrained_bert.modeling import BertLMHeadModel
from utils import get_dataset_personalities, download_pretrained_model, get_dataset


def build_input_from_segments(history, reply, tokenizer, SPECIAL_TOKENS, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    persona = []
    instance = {}
    sequence = [[bos] + list(chain(*persona))] + history + [
        reply + ([eos] if with_eos else [])]  # seq = [personas, history, reply] concatenate all persona sentences
    sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in
                                enumerate(sequence[1:])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in
                                  s]  # the last for is for repeating the speaker1 and speaker2 for all tokens
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]  # all -1 except for reply, reply is just the ids
    return instance, sequence


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(history, tokenizer, model, args, SPECIAL_TOKENS, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(history, current_output, tokenizer, SPECIAL_TOKENS,
                                                       with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)

        if "gpt2" == args.model:
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def run():
    config_file = "configs/interact_config.json"
    config = InteractConfig.from_json_file(config_file)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(config))

    if config.model_checkpoint == "":
        config.model_checkpoint = download_pretrained_model()

    torch.random.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    logger.info("Get pretrained model and tokenizer")
    if config.model == "bert":
        tokenizer_class = BertTokenizer
        model_class = BertLMHeadModel
    elif config.model == "gpt2":
        tokenizer_class = GPT2Tokenizer
        model_class = GPT2LMHeadModel
    else:
        tokenizer_class = OpenAIGPTTokenizer
        model_class = OpenAIGPTLMHeadModel

    SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]

    tokenizer = tokenizer_class.from_pretrained(config.model_checkpoint)
    model = model_class.from_pretrained(config.model_checkpoint)

    model.to(config.device)
    model.eval()

    history = []
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = sample_sequence(history, tokenizer, model, config, SPECIAL_TOKENS)
        history.append(out_ids)
        history = history[-(2 * config.max_history + 1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(out_text)


if __name__ == "__main__":
    run()
