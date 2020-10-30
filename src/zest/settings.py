"""Settings for zest."""

import os

import t5


# dataset

SPLIT_FNAME_TEMPLATE = "{split}.zest.{ext}"
"""The template for the split's file names.

The template for naming the dataset's split files. The ``split`` argument
should be the file's corresponding split. ``ext`` should be the extension for
the file (``"jsonl"``, ``"csv"``, etc.).
"""

SPLITS = {
    "train": {"name": "train", "size": 10766},
    "validation": {"name": "validation", "size": 2280},
    "test": {"name": "test", "size": 11980},
}
"""The dataset splits."""

DATASET_DIR = os.environ["ZEST_DATASET_DIR"]
"""The directory containing the zest dataset.

This setting's value is the same as the ``ZEST_DATASET_DIR`` environment
variable.
"""


# modeling

PREPROCESSED_DATASET_DIR = os.environ["ZEST_PREPROCESSED_DATASET_DIR"]
"""The directory containing the preprocessed zest dataset.

This setting's value is the same as the ``ZEST_PREPROCESSED_DATASET_DIR``
environment variable.
"""


# tensorflow datasets configuration

TFDS_DATASETS_DIR = os.environ["ZEST_TFDS_DATASETS_DIR"]
"""The directory for storing the TFDS datasets."""
# Configure T5 to use TFDS_DATASETS_DIR.
t5.data.set_tfds_data_dir_override(TFDS_DATASETS_DIR)


# logging and output

LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s: %(message)s"
"""The format string for logging."""

TQDM_KWARGS = {"ncols": 72, "leave": False}
"""Key-word arguments for tqdm progress bars."""
