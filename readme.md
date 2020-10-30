# Learning from Task Descriptions

Learn NLP tasks from descriptions.

[This repository][source] holds companion code for the EMNLP 2020 paper:
["Learning from Task Descriptions"][paper].

Jump to the most relevant section:

  - [Data](#data): Obtain the dataset: ZEST.
  - [Evaluation](#evaluation): Evaluate a model's predictions on ZEST.
  - [Crowdsourcing](#crowdsourcing): Run ZEST's crowdsourcing templates.
  - [Modeling](#modeling): Train and predict with models on ZEST.
  - [Tests](#tests): Run the repository's tests and other checks.
  - [Citation](#citation): Cite this work.
  - [Contact](#contact): Contact us.

## Data

You can download the data from
[https://ai2-datasets.s3-us-west-2.amazonaws.com/zest/zest.zip](https://ai2-datasets.s3-us-west-2.amazonaws.com/zest/zest.zip).
This file contains the train, validation and unlabeled test sets. To evaluate
on the test set, submit your predictions to the
[leaderboard](https://leaderboard.allenai.org/zest).

The dataset is newline separated JSON. Each line has a JSON object representing
a task. Each task's JSON has a `question` key offering the task description and
an `examples` key providing examples for that task. Train and validation have
labels for each example, while test does not. For example:

```
{
    "question": "After leaving office, where did this president go to retire?",
    "examples": [
        {
            "context": "Dwight David 'Ike' Eisenhower .....",
            "answer": "n/a"
        },
        ... more contexts and answers here ...
    ]
}
```

The `answer` can be a single string as above, or a list of strings in the case
where there is more then one possible valid answer from different annotators.
Models may only submit one answer to each `(question, context)` pair, which the
evaluation script considers correct if it is among the valid answers.  The
evaluation script will randomly choose one answer if the predictions file
contains multiple answers.

For the structure question type, the `answer` is type `List[Dict[str:
Union[str, List[str]]]]`, for example,

```
"question": "What rock types are there at this national park and what era did they form?",
"examples": [
    {"context": "... The younger rocks of sedimentary origin formed during the Paleozoic Era...",
     "answer": [{"rock_types": "sedimentary ", "era_formed": "Paleozoic"}]},
    ...
]
```

Each of the values in the key-value answers may contain one `str` or
`List[str]` in the case of multiple correct answers, e.g. `[{"rock_types":
"granitic", "era_formed": ["Mesozoic", "Paleozoic ", "Paleozoic and
Mesozoic"]}]`.  Each value is scored in the same way as the simple answers in
the non-structure tasks.

## Evaluation

To evaluate predictions on ZEST, use the evaluation script:

    python bin/evaluate-zest.py                  \
      --predictions_path <your_predictions_file> \
      --dev-path <path_to_dev_data>              \
      --output-path <output_file>

Your predictions file should simply be each prediction separated by a newline
character (or either a JSON Lines or a one column CSV file).

The results will be written to `<output_file>` and written to `stdout`.

## Crowdsourcing

To create ZEST, we crowdsourced tasks (questions) and examples for them from
Mechanical Turk. The crowdsourcing templates we used reside in
[`mturk-templates/`](../mturk-templates/). Task generation templates reside in
[`mturk-templates/tasks/`](../mturk-templates/tasks/), while labeling templates
reside in [`mturk-templates/labels/`](../mturk-templates/labels/).

### Running a HIT

To run one of our crowdsourcing templates on Mechanical Turk, use
[`amti`][amti].

First, install `amti`:

    $ pip install git+https://github.com/allenai/amti@19a1ced033441fad4ecadf1b0dfce9963bd8f1aa

Then, launch the batch on Mechanical Turk:

    $ amti create-batch \
        mturk-templates/$TEMPLATE_TYPE/$TEMPLATE_NAME/definition \
        mturk-templates/$TEMPLATE_TYPE/$TEMPLATE_NAME/data.jsonl

You can also preview the HIT by running a local webserver with `amti
preview-batch`, e.g.

    $ amti preview-batch \
        mturk-templates/$TEMPLATE_TYPE/$TEMPLATE_NAME/definition \
        mturk-templates/$TEMPLATE_TYPE/$TEMPLATE_NAME/data.jsonl

### Annotation Pipeline

We used the `base/` template to generate questions, which were then fed into
the `paraphrase/`, `semantics-flips/`, `combination/`, and `output-structure/`
templates to create the additional question types. Tasks were labeled based on
whether they called for classification, extraction, or structured output. Each
type has a separate labeling template in
[`mturk-templates/labels/`](../mturk-templates/labels/).

## Modeling

We evaluated two baseline models: T5 and BART. Each model has a separate
environment and process for running, see below.

### T5

#### Setup

The T5 install requires Python 3.6 or above.

First, install the project's dependencies:

    ./bin/install

Next, make sure you have the following environment variables set:

  1. `ZEST_DATASET_DIR`: The directory containing the `zest` dataset.
  2. `ZEST_PREPROCESSED_DATASET_DIR`: The directory containing the
     preprocessed `zest` dataset.
  3. `ZEST_TFDS_DATASETS_DIR`: The directory for storing the TFDS
     (tensorflow datasets) datasets.

Training requires TPUs, for training all directories will have to be paths into
Google Storage buckets, and you'll also need the environment variables:

  1. `PROJECT`: Your Google Cloud project's ID.
  2. `ZONE`: The zone in which your virtual machine is located.
  3. `TPU_NAME`: The name of your TPU.
  4. `TPU_TOPOLOGY`: The topology of the TPU.

Then, preprocess the zest data, using the `preprocess.py` script:

    $ ./bin/preprocess.py --help
    Usage: preprocess.py [OPTIONS]

      Preprocess the zest dataset for training T5.

    Options:
      --src TEXT  The source directory from which to read the zest dataset.
                  Defaults to the ZEST_DATASET_DIR environment variable.
      --dst TEXT  The destination directory to which to write the preprocessed
                  dataset. Defaults to the ZEST_PREPROCESSED_DATASET_DIR
                  environment variable.
      --help      Show this message and exit.

Finally, verify your installation:

    ./bin/verify

#### Training and Evaluation

To train T5, use `./bin/fine-tune.py`. For example:

    ./bin/fine-tune.py                \
      "zest"                      \
      "${ZEST_EXPERIMENTS_DIR}"   \
      --pretrained-model "11B"        \
      --n-steps "25000"               \
      --learning-rate "1e-3"          \
      --batch-size "32"               \
      --model-parallelism "16"        \
      --save-checkpoints-steps "2500" \
      --n-checkpoints-to-keep "10"    \
      --tpu-name "${TPU_NAME}"        \
      --tpu-topology "8x16"

The script is self-documenting, so use the `--help` option for detailed
information.

To run prediction with T5, use `./bin/predict.py`. For example:

    ./bin/predict.py                \
      "zest"                    \
      "${ZEST_EXPERIMENTS_DIR}" \
      --split "validation"          \
      --batch-size "32"             \
      --model-parallelism "16"      \
      --tpu-name "${TPU_NAME}"      \
      --tpu-topology "8x16"

To evaluate the predictions, follow the instructions in [the Evaluation
section](#evaluation). The script, `./bin/predict.py`, is also
self-documenting.

### BART

#### Setup

Run `./bin/bootstrap_bart.sh`. This creates a conda env `zest_bart` with the
code and dependencies.

#### Training and Evaluation

To run and evaluate the BART baselines, run `./bin/train_bart_run_eval.sh
/path/to/zest/data 5e-5 15`.  This command first trains BART for 15 epochs with
learning rate 5e-5, writes out the predictions on the development set to a
file, and then uses the evaluation script to calculate the official metrics.

For hardware, we ran the training and evaluation on a RTX 8000 GPU with 48GB of
RAM. If your GPU has less memory, you may need to decrease the number of beams
for decoding, or the sequence lengths (see arguments `eval_beams`,
`val_max_target_length`, and `eval_max_gen_length` to `bin/fine_tune_bart.py`).

## Tests

The code is formatted with [black][black]. You can run the formatter using the
[`bin/format`](./bin/format) script:

    $ ./bin/format

To run code quality checks, use the [`bin/verify`](./bin/verify) script:

    $ ./bin/verify

For fine-grained control of which tests to run, use [`pytest`][pytest]
directly:

    $ pytest

You can also skip slower tests by passing the `--skip-slow` (`-s`) flag:

    $ pytest --skip-slow

## Citation

If you build off this code, data, or work, please cite [the paper][paper] as
follows:

    TODO: CITATION TO BE ADDED

## Contact

For public, non-sensitive matters: please file an issue on this repository.

For private or sensitive inquiries, please contact the authors of
[paper][paper] directly.

[black]: https://black.readthedocs.io/en/stable/
[paper]: TODO
[pytest]: https://docs.pytest.org/en/latest/
[source]: https://github.com/allenai/zest
