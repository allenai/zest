"""
Reproduce the BART baseline for ZEST, from
"Learning from task descriptions", Weller et al, EMNLP 2020.
"""

import logging
import argparse
import os
import glob
import json
import time

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

# This is a hack so that we can import from transformers.examples.
import sys

sys.path.insert(2, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForSeq2SeqLM

from transformers_examples.seq2seq.callbacks import (
    Seq2SeqLoggingCallback,
    get_early_stopping_callback,
)
from transformers_examples.seq2seq.finetune import SummarizationModule
from transformers_examples.lightning_base import add_generic_args, generic_train
from transformers_examples.seq2seq.utils import (
    ROUGE_KEYS,
    pickle_save,
    save_git_info,
    use_task_specific_params,
    flatten_list,
    lmap,
)

from pytorch_lightning.logging.test_tube import TestTubeLogger
from pytorch_lightning.callbacks import Callback


logger = logging.getLogger(__name__)


def maybe_convert_data(data_dir: str) -> None:
    """
    transformers seq2seq assumes dataset has six files:
        train.source
        train.target
        val.source
        val.target
        test.source
        test.target

    This function creates these files from the .jsonl ZEST formatted files if they don't already exist.
    In the case of multiple correct answers it randomly chooses one.
    """
    import random

    random.seed(5)

    for split_in, split_out in [
        ("train", "train"),
        ("dev", "val"),
        ("test", "test"),
    ]:
        source_outname = os.path.join(data_dir, "{}.source".format(split_out))
        if os.path.exists(source_outname):
            # skip it
            continue

        target_outname = os.path.join(data_dir, "{}.target".format(split_out))
        in_name = os.path.join(data_dir, "{}.jsonl".format(split_in))

        with open(in_name, "r", encoding="utf-8") as fin, open(
            source_outname, "w", encoding="utf-8"
        ) as fsource, open(target_outname, "w", encoding="utf-8") as ftarget:

            for line in fin:
                data = json.loads(line)
                question = data["question"].replace("\n", " ")
                for example in data["examples"]:
                    context = example["context"].replace("\n", " ")
                    question_context = " zest question: {} zest context: {}\n".format(
                        question, context
                    )
                    fsource.write(question_context)

                    if split_in == "test":
                        # test set doesn't have any answers
                        answer = "TEST SET NO ANSWER"

                    else:
                        answer = example["answer"]

                        try:
                            # Special processing of multiple correct answers in structure formatted output.
                            # Chose one at random. Note the official eval script will
                            # consider all possible answers.
                            json_answer = json.loads(answer)
                            if isinstance(json_answer, list):
                                for row in json_answer:
                                    for key in row.keys():
                                        value = row[key]
                                        if isinstance(value, list):
                                            value_choice = random.choice(value)
                                            row[key] = value_choice
                                answer = json.dumps(json_answer)
                        except (json.JSONDecodeError, TypeError):
                            pass

                        if isinstance(answer, list):
                            # Chose one at random.
                            answer_choice = random.choice(answer)
                            answer = answer_choice

                    ftarget.write("{}\n".format(answer))


class ZestModule(SummarizationModule):
    mode = "summarization"
    loss_names = ["loss"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            ret = self._generative_step(batch)

        for key in ret.keys():
            value = ret[key]
            try:
                value = value.detach()
            except:
                pass
            ret[key] = value

        return ret

    def _generative_step(self, batch: dict) -> dict:
        # for reference, here are some arguments to self.model.generate that
        # may be useful to tune
        # temperature (:obj:`float`, `optional`, defaults tp 1.0):
        #    The value used to module the next token probabilities.
        # early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
        #    Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
        # max_length (:obj:`int`, `optional`, defaults to 20):
        #    The maximum length of the sequence to be generated.
        # min_length (:obj:`int`, `optional`, defaults to 10):
        #    The minimum length of the sequence to be generated.
        # repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
        #    The parameter for repetition penalty. 1.0 means no penalty. See `this paper
        #    <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
        # length_penalty (:obj:`float`, `optional`, defaults to 1.0):
        #    Exponential penalty to the length. 1.0 means no penalty.
        #    Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
        #    order to encourage the model to produce longer sequences.
        t0 = time.time()

        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.eval_beams,
            min_length=1,
            max_length=self.eval_max_length,
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["labels"])
        loss_tensors = self._step(batch)
        base_metrics = {
            name: loss for name, loss in zip(self.loss_names, loss_tensors)
        }
        rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(
            gen_time=gen_time,
            gen_len=summ_len,
            preds=preds,
            target=target,
            **rouge,
        )
        return base_metrics

    def calc_generative_metrics(self, preds, target) -> dict:
        # Run metrics with the eval script, so nothing to do.
        return {}

    def validation_epoch_end(self, outputs, prefix="val"):
        # CAUTION: doesn't support ddp
        self.step_count += 1
        # will log all_metrics
        all_metrics = {
            k: torch.stack([x[k] for x in outputs]).mean().item()
            for k in self.loss_names
        }
        all_metrics.update(
            {
                k: np.array([x[k] for x in outputs]).mean().item()
                for k in ["gen_time", "gen_len"]
            }
        )
        all_metrics["epoch_number"] = self.step_count

        # add the prefix for logging
        all_metrics_with_prefix = {
            f"{prefix}_{k}": x for k, x in all_metrics.items()
        }
        self.metrics[prefix].append(
            all_metrics_with_prefix
        )  # callback writes this to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])

        loss_tensor = torch.stack([x["loss"] for x in outputs]).mean()

        return {
            "log": all_metrics,
            "preds": preds,
            f"{prefix}_loss": loss_tensor,
        }


class SaveCheckpoint(Callback):
    def __init__(self, args):
        self.args = args

    def on_epoch_end(self, trainer, pl_module):
        torch.save(
            pl_module.state_dict(),
            os.path.join(args.output_dir, "checkpoint.pth"),
        )


def main(args, model=None) -> SummarizationModule:
    Path(args.output_dir).mkdir(exist_ok=True)
    if len(os.listdir(args.output_dir)) > 3 and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir
            )
        )
    assert args.task == "zest"
    if model is None:
        model: ZestModule = ZestModule(args)
    dataset = Path(args.data_dir).name

    logger = TestTubeLogger(save_dir=args.output_dir, name="zest", version=0)

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(
            model.val_metric, args.early_stopping_patience
        )
    else:
        es_callback = False

    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=SaveCheckpoint(args),
        early_stopping_callback=es_callback,
        logger=logger,
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")

    return model


def run_evaluate(args):
    # load the model
    import pickle
    import tqdm

    with open(os.path.join(args.output_dir, "hparams.pkl"), "rb") as fin:
        training_args = pickle.load(fin)

    model = ZestModule(training_args)
    checkpoint = torch.load(
        os.path.join(args.output_dir, "checkpoint.pth"), map_location="cpu"
    )
    model.load_state_dict(checkpoint)

    model.eval()
    model.cuda()

    if args.evaluate_only:
        loader = model.val_dataloader()
        split = "val"

    all_preds = []
    for batch_idx, batch in tqdm.tqdm(enumerate(loader)):
        batch_cuda = {k: v.cuda() for k, v in batch.items()}
        ret = model.validation_step(batch_cuda, batch_idx)
        all_preds.extend(ret["preds"])

    # save the predictions to a file
    with open(
        os.path.join(args.output_dir, "{}_preds.txt".format(split)),
        "w",
        encoding="utf-8",
    ) as fout:
        fout.write("\n".join(all_preds))

    if split == "val":
        print(
            "Run python bin/evaluate-zest.py --predictions-path {} --dev-path {} --output-path {}".format(
                os.path.join(args.output_dir, "val_preds.txt"),
                os.path.join(training_args.data_dir, "dev.jsonl"),
                os.path.join(args.output_dir, "val_preds_results_"),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        default=False,
        help="write out predictions on dev set to evaluate",
    )
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ZestModule.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()

    if args.evaluate_only:
        run_evaluate(args)
    else:
        maybe_convert_data(args.data_dir)
        main(args)
