# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import InteractConfig
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from utils import download_pretrained_model, get_dataset, _bleu, _f1_score



def build_input_from_segments(persona, history, reply, tokenizer, SPECIAL_TOKENS, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

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


def get_emotions(dataset):


    for data in tqdm(dataset['valid']):
        utterances = data['utterances']

        for utterance in utterances:
            true_emotion = utterance["emotion"]


def calculate_metrics(args, model, tokenizer, dataset, special_tokens):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    all_blues = []
    all_f1_scores = []
    all_true_sentences = []
    all_predicted_sentences = []
    for data in tqdm(dataset['valid']):
        personality = data['personality']
        utterances = data['utterances']

        #utterance = utterances[-1] #only the longest conversaion
        for utterance in utterances:
            true_label = utterance['candidates'][-1]
            history = utterance['history']
            predicted_output = []
            for i in range(args.max_length):
                instance, _ = build_input_from_segments(personality, history, predicted_output, tokenizer, special_tokens, with_eos=False)

                try:

                    if len(instance["input_ids"]) > 310:
                        truncated_history = [hist[:5] for hist in history]
                        instance, _ = build_input_from_segments(personality, truncated_history, predicted_output, tokenizer, special_tokens, with_eos=False)

                    input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
                    token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

                    logits = model(input_ids, token_type_ids=token_type_ids)
                except:
                    print("exception")
                    continue

                if "gpt2" == args.model:
                    logits = logits[0]
                logits = logits[0, -1, :] / args.temperature
                logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
                probs = F.softmax(logits, dim=-1)

                prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
                # if i < args.min_length and prev.item() in special_tokens_ids:
                #     k=0
                #     while prev.item() in special_tokens_ids and k < 100:
                #         prev = torch.multinomial(probs, num_samples=1)
                #         k+=1

                if i < args.min_length:
                    prev = torch.multinomial(probs, num_samples=1)

                # if prev.item() in special_tokens_ids:
                #     break
                predicted_output.append(prev.item())

            predicted_sentence = tokenizer.decode(predicted_output, skip_special_tokens=True)
            true_sentence = tokenizer.decode(true_label, skip_special_tokens=True)
            #looks like zero gives the best results

            all_predicted_sentences.append(predicted_sentence)
            all_true_sentences.append(true_sentence)

            bleus = [_bleu(predicted_sentence, [true_sentence], method="method"+str(i)) for i in [0,1,2,3,5]]
            #bleu = _bleu(predicted_sentence, [true_sentence])
            f1_score = _f1_score(predicted_sentence, [true_sentence])
            #print(f1_score)
            all_blues.append(bleus)
            all_f1_scores.append(f1_score)
            #compare predicted and label with bleu


    print("avg bleu", np.array(all_blues).mean(axis=0))
    print("avg f1 score", np.mean(all_f1_scores))
    print("max bleu", np.array(all_blues).max(axis=0))


def run():
    config_file = "configs/interact_config.json"
    config = InteractConfig.from_json_file(config_file)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(config))

    if config.model_checkpoint == "":
        config.model_checkpoint = download_pretrained_model()

    random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = GPT2Tokenizer if "gpt2" == config.model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(config.model_checkpoint)
    model_class = GPT2LMHeadModel if "gpt2" == config.model else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(config.model_checkpoint)

    model.to(config.device)
    model.eval()

    dataset = get_dataset(tokenizer, config.dataset_path, config.dataset_cache)

    special_tokens = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
    calculate_metrics(config, model, tokenizer, dataset, special_tokens)

if __name__ == "__main__":
    run()
