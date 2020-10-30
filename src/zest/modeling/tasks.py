"""Task definitions for zest."""

import os

import t5
import tensorflow as tf

from .. import settings
from . import preprocessors
from . import utils as modeling_utils


t5.data.TaskRegistry.add(
    name="zest",
    task_cls=modeling_utils.CsvTask,
    # args for CsvTask
    #   dataset configuration and location
    split_to_filepattern={
        split["name"]: os.path.join(
            settings.PREPROCESSED_DATASET_DIR,
            settings.SPLIT_FNAME_TEMPLATE.format(
                split=split["name"], ext="csv"
            ),
        )
        for split in settings.SPLITS.values()
    },
    text_preprocessor=[
        preprocessors.make_add_field_names_preprocessor(
            field_names=["inputs", "targets"]
        )
    ],
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
    metric_fns=[t5.evaluation.metrics.accuracy],
    #   CSV parsing
    record_defaults=[tf.string, tf.string],
    compression_type=None,
    buffer_size=None,
    header=True,
    field_delim=",",
    use_quote_delim=True,
    na_value="",
    select_cols=None,
    # args for the task class
    postprocess_fn=t5.data.postprocessors.lower_text,
    num_input_examples={
        split["name"]: split["size"] for split in settings.SPLITS.values()
    },
)
"""The zest task."""
