#! /usr/bin/env python

"""Preprocess the zest dataset for training T5."""

import csv
import json
import logging
import os
from typing import Optional

import click
import tensorflow as tf

from zest import settings, utils


logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--src",
    type=str,
    default=None,
    help="The source directory from which to read the zest"
    " dataset. Defaults to the ZEST_DATASET_DIR environment variable.",
)
@click.option(
    "--dst",
    type=str,
    default=None,
    help="The destination directory to which to write the preprocessed"
    " dataset. Defaults to the ZEST_PREPROCESSED_DATASET_DIR environment"
    " variable.",
)
def preprocess(src: Optional[str], dst: Optional[str]) -> None:
    """Preprocess the zest dataset for training T5."""
    utils.configure_logging()

    # Handle argument defaults.

    if src is None:
        src = settings.DATASET_DIR

    if dst is None:
        dst = settings.PREPROCESSED_DATASET_DIR

    # Validate the arguments.

    for split in settings.SPLITS.values():
        src_file_path = os.path.join(
            src,
            settings.SPLIT_FNAME_TEMPLATE.format(
                split=split["name"], ext="jsonl"
            ),
        )
        if not tf.io.gfile.exists(src_file_path):
            raise IOError(
                f"The {split['name']} split could not be found in {src}."
            )

    # Create the destination directory, if it doesn't already exist.

    logger.info(f"Ensuring {dst} exists.")

    tf.io.gfile.makedirs(dst)

    # Transform the dataset to a CSV and write it to DST.

    logger.info(f"Reading from {src} and writing to {dst}.")

    for split in settings.SPLITS.values():
        logger.info(f'Preprocessing {split["name"]}.')

        src_file_path = os.path.join(
            src,
            settings.SPLIT_FNAME_TEMPLATE.format(
                split=split["name"], ext="jsonl"
            ),
        )
        dst_file_path = os.path.join(
            dst,
            settings.SPLIT_FNAME_TEMPLATE.format(
                split=split["name"], ext="csv"
            ),
        )
        rows_written = 0
        with tf.io.gfile.GFile(
            src_file_path, "r"
        ) as src_file, tf.io.gfile.GFile(dst_file_path, "w") as dst_file:
            writer = csv.DictWriter(
                dst_file, fieldnames=["inputs", "targets"], dialect="unix"
            )
            writer.writeheader()
            for ln in src_file:
                task = json.loads(ln)
                for ex in task["examples"]:
                    writer.writerow(
                        {
                            "inputs": (
                                f'zest question: {task["question"]}?\n\n'
                                f'zest context: {ex["context"]}\n\n'
                            ),
                            "targets": ex["answer"],
                        }
                    )
                    rows_written += 1

        if rows_written != split["size"]:
            logger.error(
                f"Expected to write {split['size']} rows for the {split['name']},"
                f" instead {rows_written} were written."
            )

    logger.info("Finished preprocessing.")


if __name__ == "__main__":
    preprocess()  # pylint: disable=no-value-for-parameter
