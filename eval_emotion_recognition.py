# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import logging
from pprint import pformat
from collections import defaultdict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from pytorch_pretrained_bert import (OpenAIAdam, OpenAIGPTDoubleHeadLMEmotionRecognitionModel, OpenAIGPTTokenizer,
                                     GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME,
                                     BertModel, BertTokenizer)

from utils import get_dataset, get_dataset_for_daily_dialog

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>",
                  "<no_emotion>", "<happiness>", "<surprise>", "<sadness>", "<disgust>", "<anger>", "<fear>",
                  "<directive>", "<inform>", "<commissive>", "<question>",
                  "<pad>"]
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids", "token_emotion_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids", "token_emotion_ids"]

logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, config):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if config.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=config.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def get_emotion_label(tokenizer, candidate_emotion):
    _, _, _, _, no_emotion_id, happiness_id, surprise_id, sadness_id, disgust_id, anger_id, fear_id, _, _, _, _, _ = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if candidate_emotion == happiness_id:
        return 0
    elif candidate_emotion == surprise_id:
        return 1
    elif candidate_emotion == sadness_id:
        return 2
    elif candidate_emotion == disgust_id:
        return 3
    elif candidate_emotion == anger_id:
        return 4
    elif candidate_emotion == fear_id:
        return 5
    elif candidate_emotion == no_emotion_id:
        return 6


def build_input_from_segments(history, emotions, reply, true_emotion, tokenizer, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:4])
    #tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])

    instance = {}
    # sequence = [[bos] + history[0] + list(chain(*history[1:]))]  + [reply + ([eos] if with_eos else [])] #seq = [personas, history, reply] concatenate all persona sentences
    sequence = [[bos] + history[0]] + history[1:] + [reply + ([eos] if with_eos else [])]
    sequence = [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence)]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s] # the last for is for repeating the speaker1 and speaker2 for all tokens
    #instance["token_emotion_ids"] = [emotions[i] for i, s in enumerate(sequence[:-1]) for _ in s] + [true_emotion] * len(sequence[-1])
    instance["token_emotion_ids"] = [emotions[i] for i, s in enumerate(sequence[:-1]) for _ in s]

    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["mc_labels"] = get_emotion_label(tokenizer, true_emotion)
    instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:] #all -1 except for reply, reply is just the ids
    return instance, sequence


def get_data_loaders(config, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset_for_daily_dialog(tokenizer, config.dataset_path, config.dataset_cache, SPECIAL_TOKENS)

    #personachat["train"] = personachat["train"][:100]
    #personachat["valid"] = personachat["valid"][:10]

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    c = 0
    for dataset_name, dataset in personachat.items():
        num_candidates = 2#len(dataset[0]["utterances"][0]["candidates"])
        if config.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(config.num_candidates, num_candidates)
        for dialog in dataset:
            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2 * config.max_history + 1):]
                emotions = utterance["emotion"][-(2 * config.max_history + 1):]
                reply = utterance["candidates"][-1]
                true_emotion = utterance['candidates_emotions'][-1]
                if true_emotion == tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)[4]:
                    continue
                instance, _ = build_input_from_segments(history,
                                                        emotions,
                                                        reply,
                                                        true_emotion,
                                                        tokenizer)

                if len(instance["input_ids"]) > 310:
                    truncated_history = [hist[:10] for hist in history]
                    truncated_candidate = reply[:10]
                    true_emotion = utterance['candidates_emotions'][-1]
                    instance, _ = build_input_from_segments(truncated_history,
                                                            emotions,
                                                            truncated_candidate,
                                                            true_emotion,
                                                            tokenizer)
                    c+=1

                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)

                #datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                datasets[dataset_name]["n_candidates"] = num_candidates
    print(c)
    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            #if input_name != "mc_labels":
            #    tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if config.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if config.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.train_batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=config.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def train():
    config_file = "configs/train_full_pipeline_config.json"
    config = Config.from_json_file(config_file)

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if config.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", config.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(config))

    # Initialize distributed training if needed
    config.distributed = (config.local_rank != -1)
    if config.distributed:
        torch.cuda.set_device(config.local_rank)
        config.device = torch.device("cuda", config.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer_class = GPT2Tokenizer if "gpt2" in config.model_checkpoint else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(config.model_checkpoint)
    model_class = GPT2DoubleHeadsModel if "gpt2" in config.model_checkpoint else OpenAIGPTDoubleHeadLMEmotionRecognitionModel
    model = model_class.from_pretrained(config.model_checkpoint)
    tokenizer.set_special_tokens(SPECIAL_TOKENS)
    model.set_num_special_tokens(len(SPECIAL_TOKENS))
    model.to(config.device)
    optimizer = OpenAIAdam(model.parameters(), lr=config.lr)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if config.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.fp16)
    if config.distributed:
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(config, tokenizer)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    model.eval()
    num_correct = 0
    num_all = len(val_loader)
    for batch in val_loader:
        with torch.no_grad():
            batch = tuple(input_tensor.to(config.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, token_emotion_ids = batch

            model_outputs = model(input_ids, mc_token_ids, token_type_ids=token_type_ids, token_emotion_ids=token_emotion_ids)
            lm_logits, mc_logits = model_outputs[0], model_outputs[1]  # So we can also use GPT2 outputs

            indices = torch.argmax(mc_logits, dim=1)

            correct = torch.eq(indices, mc_labels).view(-1)
            num_correct += torch.sum(correct).item()

    print(num_correct / num_all)


if __name__ == "__main__":
    train()
