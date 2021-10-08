# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from config import Config
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from pytorch_pretrained_bert import (OpenAIAdam, OpenAIGPTMultiHeadModel, OpenAIGPTTokenizer,
                                     GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME,
                                     BertModel, BertTokenizer)

from utils import get_dataset, get_dataset_for_daily_dialog

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>",

                  "<no_emotion>", "<happiness>", "<surprise>", "<sadness>", "<disgust>", "<anger>", "<fear>",

                  "<work>", "<finance>", "<relationship>", "<attitude_and_emotion>", "<culture_and_education>",
                  "<school_life>", "<tourism>", "<ordinary_life>", "<politics>", "<health>",

                  "<directive>", "<inform>", "<commissive>", "<question>",
                  "<pad>"]
MODEL_INPUTS = ["input_ids", "ec_token_ids", "sc_token_ids", "lm_labels", "ec_labels", "sc_labels",
                "token_type_ids", "token_emotion_ids", "token_action_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids", "token_emotion_ids", "token_action_ids"]

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
    no_emotion_id, happiness_id, surprise_id, sadness_id, disgust_id, anger_id, fear_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[4:11])

    if candidate_emotion == no_emotion_id:
        return 0
    elif candidate_emotion == happiness_id:
        return 1
    elif candidate_emotion == surprise_id:
        return 2
    elif candidate_emotion == sadness_id:
        return 3
    elif candidate_emotion == disgust_id:
        return 4
    elif candidate_emotion == anger_id:
        return 5
    elif candidate_emotion == fear_id:
        return 6

def build_input_from_segments(topic, history, emotions, actions, reply, candidate_emotion,  canidate_act, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, speaker1, speaker2, no_emotion = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:5])

    inform = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-4])
    emotions = [no_emotion] + emotions
    actions = [inform] + actions

    instance = {}
    sequence = [[bos] + [topic]] + history + [reply + ([eos] if with_eos else [])]
    sequence = [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in enumerate(sequence)]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in
                                  s]  # the last for is for repeating the speaker1 and speaker2 for all tokens
    instance["token_emotion_ids"] = [emotions[i] for i, s in enumerate(sequence[:-1]) for _ in s] + [
        candidate_emotion] * len(sequence[-1])
    instance["token_action_ids"] = [actions[i] for i, s in enumerate(sequence[:-1]) for _ in s] + [canidate_act] * len(
        sequence[-1])

    instance["ec_token_ids"] = len(instance["input_ids"]) - 1
    instance["sc_token_ids"] = len(instance["input_ids"]) - 2
    instance["ec_labels"] = -1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][
                                                                                     1:]  # all -1 except for reply, reply is just the ids
        instance["ec_labels"] = get_emotion_label(tokenizer, candidate_emotion)
    return instance, sequence


def get_data_loaders(config, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset_for_daily_dialog(tokenizer, config.dataset_path, config.dataset_cache, SPECIAL_TOKENS)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    gpu_max_length = 310
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if config.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(config.num_candidates, num_candidates)
        for dialog in dataset:
            topic = dialog["topic"]
            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2 * config.max_history + 1):]
                emotions = utterance["emotion"][-(2 * config.max_history + 1):]
                actions = utterance["act"][-(2 * config.max_history + 1):]
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(
                        j == num_candidates - 1)  # the true label is always the last one in list of candidates
                    candidate_emotion = utterance['candidates_emotions'][j]
                    candidate_act = utterance['candidates_acts'][j]
                    instance, _ = build_input_from_segments(topic, history, emotions, actions, candidate,
                                                            candidate_emotion, candidate_act, tokenizer, lm_labels)

                    if len(instance["input_ids"]) > gpu_max_length:
                        truncated_history = [hist[:10] for hist in history]
                        truncated_candidate = candidate[:10]
                        instance, _ = build_input_from_segments(topic, truncated_history, emotions, actions,
                                                                truncated_candidate,
                                                                candidate_emotion, candidate_act, tokenizer, lm_labels)

                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)

                datasets[dataset_name]["sc_labels"].append(num_candidates - 1)
                datasets[dataset_name]["n_candidates"] = num_candidates

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "sc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
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
    config_file = "configs/train_multihead_config.json"
    config = Config.from_json_file(config_file)

    ec_coef = 1
    sc_coef = 1

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if config.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d",
                   config.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(config))

    # Initialize distributed training if needed
    config.distributed = (config.local_rank != -1)
    if config.distributed:
        torch.cuda.set_device(config.local_rank)
        config.device = torch.device("cuda", config.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer_class = OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(config.model_checkpoint)
    model_class = OpenAIGPTMultiHeadModel
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

    # Training function and trainer
    def update(engine, batch):
        model.train()
        # input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, token_emotion_ids, token_action_ids = tuple(input_tensor.to(config.device) for input_tensor in batch)
        input_ids, ec_token_ids, sc_token_ids, lm_labels, ec_labels, sc_labels, token_type_ids, token_emotion_ids, token_action_ids = tuple(
            input_tensor.to(config.device) for input_tensor in batch)

        lm_loss, emotion_loss, sentence_loss = model(input_ids, ec_token_ids, sc_token_ids,
                                                     lm_labels, ec_labels, sc_labels, token_type_ids,
                                                     token_emotion_ids, token_action_ids)
        loss = (lm_loss * config.lm_coef + emotion_loss * ec_coef + sentence_loss * sc_coef) / config.gradient_accumulation_steps
        if config.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
        if engine.state.iteration % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(config.device) for input_tensor in batch)
            input_ids, ec_token_ids, sc_token_ids, lm_labels, ec_labels, \
            sc_labels, token_type_ids, token_emotion_ids, token_action_ids = batch
            # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            model_outputs = model(input_ids, ec_token_ids, sc_token_ids, token_type_ids=token_type_ids,
                                  token_emotion_ids=token_emotion_ids,
                                  token_action_ids=token_action_ids)
            lm_logits, mc_logits = model_outputs[0], model_outputs[2]  # So we can also use GPT2 outputs
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, sc_labels)

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if config.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if config.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if config.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, config.lr), (config.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], config),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], config)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if config.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=config.log_dir)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()),
                                                              another_engine=trainer),
                         event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(config, tb_logger.writer.log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))
        tokenizer.save_vocabulary(tb_logger.writer.log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=config.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if config.local_rank in [-1, 0] and config.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.log_dir,
                                                                     WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    train()
