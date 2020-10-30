import os
import re
import csv
import copy
import json
import math
import importlib
import itertools
import collections
import string
import random
import warnings
import traceback
from typing import Any, Dict, List, Set, Tuple, Union, Optional

import tqdm
import pandas as pd
import click
import numpy as np
from scipy.optimize import linear_sum_assignment

ALL_TYPES = [
    "normal",
    "paraphrase",
    "target_semantics",
    "combination",
    "structure",
]

RANDOM_SEED = 0
random.seed(RANDOM_SEED)

LOG = False

COMPLILE_PENALTY = 0.25

ANALYSIS_AREAS = ["gen", "domain", "format"]

"""
The following code is taken from DROP at https://github.com/allenai/allennlp-reading-comprehension/blob/14fc0a424441de1b1c9cba8745f678df447b407f/allennlp_rc/eval/drop_eval.py
Special thanks to the authors!
"""
# From here through _normalize_answer was originally copied from:
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# Then cleaned up and modified a bit.
def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [
        _white_space_fix(
            _remove_articles(_normalize_number(_remove_punc(_lower(token))))
        )
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(
                    pred_item, gold_item
                )
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_numbers_if_present(
    gold_bag: Set[str], predicted_bag: Set[str]
) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_metrics(
    predicted: Union[str, List[str], Tuple[str, ...]],
    gold: Union[str, List[str], Tuple[str, ...]],
) -> Tuple[float, float]:
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(
        predicted_bags[0]
    ) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def answer_json_to_strings(
    answer: Dict[str, Any]
) -> Tuple[Tuple[str, ...], str]:
    """
    Takes an answer JSON blob from the DROP data release and converts it into strings used for
    evaluation.
    """
    if "number" in answer and answer["number"]:
        return tuple([str(answer["number"])]), "number"
    elif "spans" in answer and answer["spans"]:
        return (
            tuple(answer["spans"]),
            "span" if len(answer["spans"]) == 1 else "spans",
        )
    elif "date" in answer:
        return (
            tuple(
                [
                    "{0} {1} {2}".format(
                        answer["date"]["day"],
                        answer["date"]["month"],
                        answer["date"]["year"],
                    )
                ]
            ),
            "date",
        )
    else:
        raise ValueError(
            f"Answer type not found, should be one of number, spans or date at: {json.dumps(answer)}"
        )


def evaluate_json(
    annotations: Dict[str, Any], predicted_answers: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Takes gold annotations and predicted answers and  evaluates the predictions for each question
    in the gold annotations.  Both JSON dictionaries must have query_id keys, which are used to
    match predictions to gold annotations (note that these are somewhat deep in the JSON for the
    gold annotations, but must be top-level keys in the predicted answers).

    The ``annotations`` are assumed to have the format of the dev set in the DROP data release.
    The ``predicted_answers`` JSON must be a dictionary keyed by query id, where the value is a string
    (or list of strings) that is the answer.
    """
    instance_exact_match = []
    instance_f1 = []
    # for each type as well
    type_to_em: Dict[str, List[float]] = collections.defaultdict(list)
    type_to_f1: Dict[str, List[float]] = collections.defaultdict(list)
    for _, annotation in annotations.items():
        for qa_pair in annotation["qa_pairs"]:
            query_id = qa_pair["query_id"]
            max_em_score = 0.0
            max_f1_score = 0.0
            max_type = None
            if query_id in predicted_answers:
                predicted = predicted_answers[query_id]
                candidate_answers = [qa_pair["answer"]]
                if (
                    "validated_answers" in qa_pair
                    and qa_pair["validated_answers"]
                ):
                    candidate_answers += qa_pair["validated_answers"]
                for answer in candidate_answers:
                    gold_answer, gold_type = answer_json_to_strings(answer)
                    em_score, f1_score = get_metrics(predicted, gold_answer)
                    if gold_answer[0].strip() != "":
                        max_em_score = max(max_em_score, em_score)
                        max_f1_score = max(max_f1_score, f1_score)
                        if (
                            max_em_score == em_score
                            and max_f1_score == f1_score
                        ):
                            max_type = gold_type
            else:
                warnings.warn(
                    "Missing prediction for question: {}".format(query_id)
                )
                if qa_pair and qa_pair["answer"]:
                    max_type = answer_json_to_strings(qa_pair["answer"])[1]
                else:
                    max_type = "number"
                max_em_score = 0.0
                max_f1_score = 0.0
            instance_exact_match.append(max_em_score)
            instance_f1.append(max_f1_score)
            type_to_em[max_type].append(max_em_score)
            type_to_f1[max_type].append(max_f1_score)

    global_em = np.mean(instance_exact_match)
    global_f1 = np.mean(instance_f1)
    if LOG:  # added by us
        print("Exact-match accuracy {0:.2f}".format(global_em * 100))
        print("F1 score {0:.2f}".format(global_f1 * 100))
        print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
        print("----")
    total = np.sum([len(v) for v in type_to_em.values()])
    for typ in sorted(type_to_em.keys()):
        if LOG:  # added by us
            print(
                "{0}: {1} ({2:.2f}%)".format(
                    typ,
                    len(type_to_em[typ]),
                    100.0 * len(type_to_em[typ]) / total,
                )
            )
            print(
                "  Exact-match accuracy {0:.3f}".format(
                    100.0 * np.mean(type_to_em[typ])
                )
            )
            print("  F1 score {0:.3f}".format(100.0 * np.mean(type_to_f1[typ])))
    return global_em, global_f1


def evaluate_prediction_file(
    prediction_path: str, gold_path: str, output_path: Optional[str] = None
) -> Tuple[float, float]:
    """
    Takes a prediction file and a gold file and evaluates the predictions for each question in the
    gold file.  Both files must be json formatted and must have query_id keys, which are used to
    match predictions to gold annotations.  The gold file is assumed to have the format of the dev
    set in the DROP data release.  The prediction file must be a JSON dictionary keyed by query id,
    where the value is either a JSON dictionary with an "answer" key, or just a string (or list of
    strings) that is the answer. Writes a json with global_em and global_f1 metrics to file at
    the specified output path, unless None is passed as output path.
    """
    predicted_answers = json.load(open(prediction_path, encoding="utf-8"))
    annotations = json.load(open(gold_path, encoding="utf-8"))
    global_em, global_f1 = evaluate_json(annotations, predicted_answers)

    # Output predictions to file if an output path is given
    if output_path is not None:
        output_dict = {"global_em": global_em, "global_f1": global_f1}

        with open(output_path, "w", encoding="utf8") as outfile:
            json.dump(output_dict, outfile)

    return (global_em, global_f1)


"""
End DROP code
"""


"""
The following code is from `mechanical_turk.amti_pipeline.utils.data_utils`
and placed here for ease of use
"""


def get_type_hint(q_list: list):
    """
    Uses the first word of the question to return the likely response format
    of the question's answer.

    Args:
    -----
        q_list: a list of strings containing the questions

    Returns:
    --------
        A list of strings containing whether the questions are extraction or classification
    """
    hints = []
    for q in q_list:
        first_word = q.split(" ")[0].lower()
        if first_word in [
            "what's",
            "which",
            "what",
            "when",
            "where",
            "how",
            "on",
            "who",
            "in",
            "per",
            "the",
            "at",
            "could",
            "aside",
            "during",
            "how's",
            "pertaining",
        ]:
            hints.append("extraction")
        elif first_word in [
            "does",
            "are",
            "is",
            "am",
            "can",
            "do",
            "did",
            "was",
            "were",
            "should",
            "has",
            "have",
            "will",
            "while",
            "after",
        ]:
            hints.append("classification")
        else:
            raise Exception(
                "Did not expect Q: ", q
            )  # NOTE: if you add a new question with a new start word, add it here
    return hints


def is_na(x: str):
    """
    Given an answer, checks if it is NaN.  Handles structure and non-structure cases

    Args:
    -----
        x: a string of the answer or predictions

    Returns:
    --------
        A boolean indicating whether the correct answer is NaN or not
    """
    if type(x) == list:
        return sum([is_na(item) for item in x]) == len(x)

    if type(x) == str and "{" in x:
        try:
            struct = json.loads(x.replace("nan", '"n/a"'))
        except Exception as e:
            # print("### ERROR: failed to parse {} due to {} ###".format(x, e))
            return False
        all_nans = True
        for dict_item in struct:
            for value in dict_item.values():
                if not (
                    pd.isnull(value)
                    or value.lower() == "n/a"
                    or value.lower() == "na"
                    or value.lower() == "nan"
                ):
                    all_nans = False
        return all_nans
    else:
        if pd.isnull(x) or (
            type(x) == str and (x.lower() == "n/a" or x.lower() == "na")
        ):
            return True
        elif type(x) != str:
            raise NotImplementedError(
                "Can't evaluate non-structure answer that is not a string"
            )
        else:
            return False


def flatten(x: list) -> list:
    """
    A helper function to flatten a list of lists to only a list
    """
    return [a for i in x for a in flatten(i)] if isinstance(x, list) else [x]


"""
End of `data_utils` code
"""


def get_generalization_detailed(
    row: pd.Series, base: pd.DataFrame, detailed_scores: Dict,
):
    """
    A helper function to take a row of the results of a generalization instance and return how the question
    performed, compared to the original. This function will update the `detailed_scores` dict to reflect the results.

    Args:
    -----
        row: a Pandas Series containing a row of results that include whether it the generalization isntance
            was answered correctly, which category it was in, and which original question it derives from.
        base: a Pandas DataFrame of the base original questions and their results
        detailed_scores: a dictionary containing the results that will be updated

    Mutates:
    --------
        detailed_scores: a dictionary of lists of dictionaries.  Updated with the results of the questions derivatives
        for "base" and "derived".
    """
    derives_from = row["derives_from"]
    assert type(derives_from) == list
    category = row["category"]

    if category in ["structure", "combination"]:
        # structure is evaluated on its own
        detailed_scores[category].append(
            {
                "base": np.array(row.score).mean(),
                "derived": np.array(row.score).mean(),
            }
        )
    else:
        # semantic flips, paraphrase
        gen_correct = None
        for derivative in derives_from:
            base_correct = base[base["id"] == derivative]["score"]
            if not len(base_correct):
                if LOG:
                    warnings.warn(
                        "Warning: derivation not found in same set, skipping"
                    )
                continue
            gen_correct = row["score"]
            assert (
                type(gen_correct) == float and len(base_correct) == 1
            ), "had more than one response {}, {}".format(
                gen_correct, base_correct
            )
            base_correct = base_correct.iloc[0]

        if gen_correct is not None:
            detailed_scores[category].append(
                {"base": base_correct, "derived": gen_correct}
            )
        elif LOG:
            warnings.warn("ERROR could not recover task {}".format(row))


def gather_standard_results_thresholded(list_of_dicts: List[Dict],):
    """
    The typical analysis of results format and domain and generalization
    This does NOT include the evaluation of the results paired, thus,
    this is just for those that are curious

    Args:
    -----
        list_of_dicts: a list of dictionaries where each dictionary contains the results
            across some area we want to measure (format, domain, etc.)

    Returns:
    --------
        a dictionary containing each area of analysis and their results
    """
    final_results = {}
    overall_scores = []
    for name_of_dict, dict_values in zip(ANALYSIS_AREAS, list_of_dicts):
        if LOG:
            print("Evaluating", name_of_dict)

        # set up empty dicts
        if name_of_dict not in final_results:
            final_results[name_of_dict] = {}

        # go through each generalization type
        for generalization_key, results_list in dict_values.items():

            final_results[name_of_dict][generalization_key] = threshold_results(
                results_list
            )

    return final_results


def threshold_results(results_list):
    """
    Given a Numpy Array, thresholds the results

    Args:
    ----
        results_list: a numpy array of scores of floats

    Returns:
    --------
        A dictionary containing thresholds at various levels, as well as min/max/mean
    """
    return {
        "mean": np.mean(np.array(results_list)),
        "75": (np.array(results_list) >= 0.75).mean(),
        "90": (np.array(results_list) >= 0.90).mean(),
    }


def gather_generalization_metrics(gen_dict):
    """
    The main evaluation script

    Compares the results accross all the areas of generalization.

    Paraphrase and Semantic Flips are linked to the base questions
    Structure and Combinations are just thresholded by the competence metric

    Args:
    -----
        `gen_dict`: the generalization dictionary with corresponding results

    Returns:
    --------
        A dictionary containing the results for each area of generalization
    """
    gen_results = {}
    for name_of_dict, dict_values in gen_dict.items():

        if name_of_dict in ["paraphrase", "structure", "target_semantics"]:
            # score whether both the `base_score` and the `gen_score` are above a threshold
            # since we need the min score to above the threshold for it to count, just pass the min
            scores = [
                min(question_dict["base"], question_dict["derived"])
                for question_dict in dict_values
            ]
            metric_results = threshold_results(np.array(scores))

        elif name_of_dict in ["combination"]:
            scores = [question_dict["derived"] for question_dict in dict_values]
            metric_results = threshold_results(np.array(scores))

        else:
            raise NotImplementedError(
                "Cannot parse generalization type: {}".format(name_of_dict)
            )

        gen_results[name_of_dict] = metric_results

    return gen_results


def get_key_value_pairs_as_list(cur_dict: Dict[str, str]) -> List:
    """
    A helper function to convert (key: value) pairs into
    a list of answers ready to be evaluated for the structure
    """
    answers = []
    multiples = []
    # the units to be compared are `{key} {value}` so that we can't mixup keys and get 100%.
    for key, value in cur_dict.items():
        if type(value) == list:
            multiples.append((key, value))
            continue

        if "|" in str(value):
            # was multiple answers for key, give them both the answer
            # don't penalize on full key with underscores, give partial credit
            answers.extend(
                [
                    "{} {}".format(key.replace("_", " "), str(new_value))
                    for new_value in value.split("|")
                ]
            )
        else:
            # don't penalize on full key with underscores, give partial credit
            answers.append("{} {}".format(key.replace("_", " "), str(value)))

    if len(multiples):
        final_answers = get_recursive_lists(multiples, answers)
    else:
        final_answers = [answers]

    return final_answers


def get_recursive_lists(
    multiples: List[Tuple[str, List[str]]], answers: List[str],
):
    if len(multiples) == 0:
        return answers

    (key, values) = multiples.pop()
    new_base = copy.deepcopy(answers)
    all_answers = [copy.deepcopy(new_base) for _ in range(len(values))]
    for index, value in enumerate(values):
        if "|" in str(value):
            # was multiple answers for key, give them both the answer
            # don't penalize on full key with underscores, give partial credit
            all_answers[index].extend(
                [
                    "{} {}".format(key.replace("_", " "), new_value)
                    for new_value in value.split("|")
                ]
            )
        else:
            # don't penalize on full key with underscores, give partial credit
            all_answers[index].append(
                "{} {}".format(key.replace("_", " "), value)
            )

    final_list = []
    for answer_set in all_answers:
        output = get_recursive_lists(multiples, answer_set)
        if type(output[0]) == list:
            final_list.extend(output)
        else:
            final_list.append(output)
    return final_list


def evaluate_structure(gold: List[Dict], potential_preds: List[Dict]) -> float:
    """
    Given the gold and predicted structure, returns the score

    Args:
    -----
        gold: a list containing dictionaries of results
        potential_preds: a list containing predicted dictionaries of results

    Returns:
        the score of the predicted structure vs the gold structure
    """
    scores = []
    # get the scores for each answer in the list of dicts
    for gold_index, gold_answer in enumerate(gold):
        # add each key and value as a seperate answer to match
        cur_pred = potential_preds[gold_index]
        pred_answers = get_key_value_pairs_as_list(cur_pred)
        if type(pred_answers[0]) == list:
            pred_answers = pred_answers[0]

        gold_answer_list = get_key_value_pairs_as_list(gold_answer)
        best_score = -1
        for gold_answers in gold_answer_list:
            cur_score = get_metrics(pred_answers, gold_answers)[1]
            if cur_score > best_score:
                best_score = cur_score

        scores.append(best_score)

    # average score of the list of dicts
    return np.mean(scores)


def pad_smaller_answer(
    gold: List[Dict], predictions: List[Dict]
) -> Tuple[List[Dict]]:
    """
    Since we want to compare the dictionaries, we pad the smaller one to
    allow for a direct comparison (empty dicts = no prediction)

    Args:
    ------
        gold: a list of dictionaries of the true answers
        predictions: a list of dictionaries of the predicted answers

    Returns:
    --------
        gold and predictions in a tuple, where the smaller one is padded with empty dicts
    """
    # counting on pass by reference here
    smaller = gold if len(gold) < len(predictions) else predictions
    diff = abs(len(gold) - len(predictions))

    for _ in range(diff):
        smaller.append({})

    assert len(gold) == len(predictions)
    return gold, predictions


def get_greedy_matches(gold: List[Dict], predictions: List[Dict]) -> List[Dict]:
    """
    To evaluate structure, we first need to best-align the instances in the json list
    then align each key value in the json list's dicts.  This function
    does a greedy search over each instance in the JSON list to greedily match
    each dict instance to the best instance in the `gold` array

    Args:
    -----
        gold: the gold list of answers, a list of dicts
        predictions: the predicted answer list, also a list of dicts

    Returns:
    --------
        The predicted list, in best order
    """
    assert len(gold) == len(predictions)
    available_gold_indexes = list(range(len(gold)))
    greedy_matches = []
    for _, pred_answer in enumerate(predictions):
        scores_for_pred = []
        for gold_index, gold_answer_idx in enumerate(available_gold_indexes):
            gold_score = evaluate_structure(
                [gold[gold_answer_idx]], [pred_answer]
            )
            if gold_score == 1.0:  # if perfect, we'll use it
                greedy_matches.append(available_gold_indexes[gold_index])
                del available_gold_indexes[gold_index]
                break
            scores_for_pred.append(gold_score)
        else:  # take the best match if we never find the perfect one
            best_gold_idx = np.argmax(np.array(scores_for_pred))
            greedy_matches.append(available_gold_indexes[best_gold_idx])
            del available_gold_indexes[best_gold_idx]

    assert len(available_gold_indexes) == 0
    assert len(greedy_matches) == len(gold)
    assert set(greedy_matches) == set(
        list(range(len(gold)))
    ), f"Greedy: {set(greedy_matches)}, Gold: {set(list(range(len(gold))))}"

    final_preds = [
        predictions[index] for index in greedy_matches
    ]  # get the values from the ordering
    return final_preds


def score_prediction_and_answer(gold: List, predictions: str) -> float:
    """
    How to score the prediction based on the answer.  Borrows from Quoref/DROP.

    If it's a structure question, does additional alignments, then evaluates each key
    according to metrics defined in DROP.

    If classification or text, evaluates it using the metrics in DROP.

    Args:
    ------
        gold: a list of the true answers
        predictions: a string containing the predictions

    Returns:
    ---------
        The score of the predicted answer compared to the gold
    """
    if sum(
        [
            type(gold_answer) == str and "}" in gold_answer
            for gold_answer in gold
        ]
    ):
        # is structure question - should only have one true list of answers
        assert (
            len(gold) == 1 and type(gold) == list
        ), f"{gold} with type {type(gold)}"
        gold = gold[0]
        assert type(gold) == str, type(gold)  # list is in JSON form
        assert type(predictions) == str

        gold = json.loads(gold)
        try:
            predictions = json.loads(predictions)
            assert type(predictions) == list
        except Exception:
            # could not parse the prediction structure
            gold, pred = pad_smaller_answer(gold, [])
            assert len(gold) == len(pred)
            print("# ERROR: was a structure question but could not parse #")
            return 0.0, (gold, pred)

        # pad the smaller with empty dicts to be able to compare them
        gold, predictions = pad_smaller_answer(gold, predictions)
        if gold == predictions:
            return 1.0, (gold, predictions)

        predicted_answer_permutation = get_greedy_matches(gold, predictions)

        max_score = -1
        best_prediction = []
        for permutation in tqdm.tqdm(
            [predicted_answer_permutation], leave=False
        ):
            # this function handles the scorings
            score = evaluate_structure(gold, list(permutation))
            if score > max_score:
                max_score = score
                best_prediction = list(permutation)
                if max_score == 1.0:
                    break

        return max_score, (gold, best_prediction)

    else:
        best_score = 0.0
        all_scores = []

        if "[" in predictions:
            try:
                predictions = json.loads(predictions)
            except Exception:
                pass  # was not JSON

        if type(predictions) == list:
            warnings.warn(
                "Was given a prediction that contains a list, will randomly sample one of them..."
            )
            predictions = random.choice(predictions)

        if type(predictions) != str:
            # invalid type
            print(
                "# ERROR was given the wrong type for a non-structure answer #"
            )
            return 0.0, None

        predictions = predictions.split("|")  # can have multiple answers
        for gold_answer in gold:
            gold_check = gold_answer.split("|")
            # use the DROP/Quoref metric already defined
            score = get_metrics(predictions, gold_check)[1]  # f1
            all_scores.append(score)
            if score > best_score:
                best_score = score
                if best_score == 1.0:
                    break

        return best_score, None


def score_and_decompose_structure(
    structure_alignment: Tuple[List[Dict]], mapping: Dict[str, List],
):
    """
    Scores the best structure alignment by getting the answer for each item in each dictionary.
    Score is equal to the score of the values compared weighted by the score of the keys compared.
    These scores are added to the mapping in order to compute the task level F1

    Args:
    -----
        structure_alignment: the gold and predicted answers, aligned, if structure is being used, else None
        mapping: a dictionary containing the true positive, false positive, etc. scores for F1 calculation later
    """
    gold, prediction = structure_alignment
    assert len(gold) == len(prediction)
    for dictionary_index in range(len(gold)):
        gold_dict = gold[dictionary_index]
        prediction_dict = prediction[dictionary_index]

        gold_keys = list(gold_dict.keys())
        pred_keys = list(prediction_dict.keys())

        gold_values = list(gold_dict.values())
        pred_values = list(prediction_dict.values())

        for item_index in range(len(gold_keys)):
            gold_key = [gold_keys[item_index].replace("_", " ")]
            if len(pred_keys) > item_index:
                pred_key = pred_keys[item_index].replace("_", " ")
                key_score, _ = score_prediction_and_answer(gold_key, pred_key)

                gold_value = (
                    gold_values[item_index]
                    if not is_na(gold_values[item_index])
                    else "n/a"
                )
                pred_value = (
                    pred_values[item_index]
                    if not is_na(pred_values[item_index])
                    else "n/a"
                )
                if type(gold_value) == str:
                    gold_value = [gold_value]

                value_score, _ = score_prediction_and_answer(
                    gold_value, pred_value
                )
                score = (
                    value_score * key_score
                )  # weight by key mapping, will almost always be 1.0
            else:
                gold_value = (
                    gold_values[item_index]
                    if not is_na(gold_values[item_index])
                    else "n/a"
                )
                if type(gold_value) == str:
                    gold_value = [gold_value]
                score = 0

            add_to_mapping_precision_and_recall(
                mapping,
                gold_value,
                pred_value if len(pred_keys) > item_index else "",
                score,
            )


def add_to_mapping_precision_and_recall(
    mapping: dict, all_answers: List, prediction: str, score: float
):
    """
    Adds the predicted answer and score to the correct bucket, in order to
    calculate F1 later.  Uses `all_answers` to determine if the answer is NaN
    or not.

    Args:
    -----
        mapping: a dictionary containing the results for the task
        all_answers: a list of the correct answer, to show if it's NaN or not
        prediction: a string of the prediction
        score: the float score of the prediction on the gold answer
    """
    gold = all_answers[
        0
    ]  # if there are multiple, it means it's not nan, so it's okay to do it this way
    if is_na(prediction):
        if is_na(gold.lower()):
            mapping["tn"].append(score)
        else:
            mapping["fn"].append(score)
            mapping["all_gold_positive"].append(1)
    else:
        # not NAN prediction
        if is_na(gold.lower()):
            mapping["fp"].append(score)
        else:
            mapping["tp"].append(score)
            mapping["all_gold_positive"].append(1)


def calculate_f1(mapping: Dict[str, List]):
    """
    Computes the F1 score for the given task

    Args:
    -----
        mapping: a dictionary of string -> lists, each string being one of the four areas of true positive, false positive, etc.
            that map to a list of scores

    Returns:
    --------
        The f1 score for the task
    """

    if LOG:
        if len(mapping["all_gold_positive"]) == 0:
            warnings.warn("### No positives in instance ###")
        if (len(mapping["tp"]) + len(mapping["fp"])) == 0:
            warnings.warn("### No TP or FP in instance ###")

    precision = (
        sum(mapping["tp"]) / (len(mapping["tp"]) + len(mapping["fp"]))
        if (len(mapping["tp"]) + len(mapping["fp"])) != 0
        else 1
    )
    recall = (
        sum(mapping["tp"]) / len(mapping["all_gold_positive"])
        if len(mapping["all_gold_positive"]) != 0
        else 1
    )
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    if LOG:
        print("Precision {}, Recall {}, F1 {}".format(precision, recall, f1))
    return f1


def evaluate_predictions(dev: List[Dict], predictions: List, output_path: str):
    """
    The core function to evaluate the predictions on the dev set.  Will write a json file of results
    to `output_path`.

    Args
    ----
        dev: a list of dictionaries, where each dictionary contains an example instances from the dataset
        predictions: a list of strings, where each string is the answer to the corresponding dev instance
        output_path: a string indicating where to write the final results to
    """
    # for standard analysis
    type_scores = collections.defaultdict(list)
    domain_scores = collections.defaultdict(list)
    by_format = collections.defaultdict(list)
    # for generalization analysis
    detailed_scores = collections.defaultdict(list)

    incorrects = []
    normal_instances = []
    paraphrase_instances = []
    generalization_instances = []
    prediction_counter = 0

    # get the base question info
    for _, cur_dev in enumerate(tqdm.tqdm(dev)):
        category = cur_dev["type"]["generalization_type"]
        domain = cur_dev["type"]["domain"]
        type_format = get_type_hint([cur_dev["question"]])[0]

        if (
            cur_dev["question"]
            == "Are there any buildings named after this president?"
        ):
            prediction_counter += len(cur_dev["examples"])
            continue

        n_correct = 0
        predictions_for_q = []
        matches = []
        mapping_of_results = collections.defaultdict(list)
        for cur_example in cur_dev["examples"]:
            prediction = predictions[prediction_counter]
            predictions_for_q.append(prediction)
            prediction_counter += 1

            correct_answers = (
                [cur_example["answer"]]
                if type(cur_example["answer"]) != list
                else cur_example["answer"]
            )

            correct_score, structure_alignment = score_prediction_and_answer(
                correct_answers, prediction
            )
            if structure_alignment is not None:
                # we evaluate structure a little differently
                score_and_decompose_structure(
                    structure_alignment, mapping_of_results
                )
            else:
                add_to_mapping_precision_and_recall(
                    mapping_of_results,
                    correct_answers,
                    prediction,
                    correct_score,
                )
            matches.append(correct_score)

            if correct_score != 1.0:
                bad_answer = {
                    "question": cur_dev["question"],
                    "context": cur_example["context"],
                    "prediction": prediction,
                    "gold": correct_answers,
                }
                incorrects.append(bad_answer)
                if LOG:
                    print(
                        f"\nIncorrect! Predicted {prediction}\n, Gold: {correct_answers}\n"
                    )

        if category == "structure":
            # additional logging
            question_score = calculate_f1(
                mapping_of_results
            )  # f1 is normal score
        else:
            question_score = calculate_f1(
                mapping_of_results
            )  # f1 is normal score
        type_scores[category].append(question_score)
        domain_scores[domain].append(question_score)
        by_format[type_format].append(question_score)
        derivation_info = {
            "id": cur_dev["id"],
            "question": cur_dev["question"],
            "score": question_score,
            "category": category,
            "derives_from": cur_dev["type"]["derives_from"],
            "predictions": predictions_for_q,
            "dev_examples": cur_dev["examples"],
            "format": type_format,
            "matches": matches,  # each question score for correctness
        }

        if category == "normal":
            normal_instances.append(derivation_info)
        else:
            generalization_instances.append(derivation_info)
            if category == "paraphrase":
                paraphrase_instances.append(derivation_info)

    # each row is a question with 20 examples
    generalize = pd.DataFrame(generalization_instances)
    base = pd.DataFrame(normal_instances)

    # get the scores for each type of generalization
    generalize.apply(
        lambda x: get_generalization_detailed(x, base, detailed_scores), axis=1
    )

    final_results_standard = gather_standard_results_thresholded(
        [type_scores, domain_scores, by_format]
    )
    final_results_gen = gather_generalization_metrics(detailed_scores)

    # add normal, since `gather_generalization_metrics` won't do that
    final_results_gen["normal"] = threshold_results(base.score.to_numpy())

    print("Writing results to {}\n".format(output_path))
    print("Eval Standard: {}\n".format(final_results_standard))
    print("Eval Official: {}\n".format(final_results_gen))

    # save as a CSV too, for reporting ease
    all_scores = []
    row_names = []
    for area in final_results_gen.keys():
        names = final_results_gen[area].keys()
        row_names.append(area)
        all_scores.append(np.array(list(final_results_gen[area].values())))

    df = pd.DataFrame(all_scores, columns=names, index=row_names)
    df.loc["average"] = df.mean()
    df = (df.round(2) * 100).astype(int)  # make numbers pretty
    df.to_csv(output_path + ".csv")
    print(df)

    errors = pd.DataFrame(incorrects)
    errors.to_csv(output_path + "errors.csv")
    errors.sample(n=min(100, len(errors)), random_state=RANDOM_SEED).to_csv(
        output_path + "_sampled_errors.csv"
    )

    final_results_gen["overall"] = df.loc["average"]["90"]
    print(
        "\n#### The overall score is: {} ####".format(
            final_results_gen["overall"]
        )
    )

    def convert(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError

    with open(output_path, "w") as fout:
        json.dump(final_results_gen, fout, default=convert)


@click.command()
@click.option(
    "--predictions-path",
    "-p",
    type=str,
    help="The path to the predictions file (single column CSV file)",
    default="",
)
@click.option(
    "--dev-path",
    "-d",
    type=str,
    help="the path to the dev data (in jsonl form)",
    default="",
)
@click.option(
    "--output-path",
    "-o",
    type=str,
    help="The name of the file to write the results to",
    default="evaluation/results.json",
)
@click.option("--verbose", "-v", is_flag=True, help="Print more output.")
def main(predictions_path: str, dev_path: str, output_path: str, verbose: bool):
    global LOG
    """
    The wrapper file to get and evaluate the predictions on the dev set, writing to `output_path`

    Args
    ----
        predictions_path: a string to the predictions file. This file should be a one column CSV, with no header or index.
            The CSV is so that any commas in the answer are escaped.
        dev_path: a string of the path to the dev dataset
        output_path: a string of the location to write the results to.
    """
    assert (
        predictions_path.strip() and dev_path.strip()
    ), "have paths to dev and predictions non-empty"

    if verbose:
        LOG = True

    # predictions file should be a one column CSV with no header or index
    predictions = []
    with open(predictions_path, "r") as fin:
        for line in fin:
            line_value = line.strip()

            # check to see if it was encoded as json or not
            if line_value[0] == '"' and line_value[-1] == '"':
                line_value = json.loads(line_value)

            while "⁇" in line_value:
                # T5 can't do curly braces, outputs these
                line_value = line_value.replace("⁇", "{", 1)
                line_value = line_value.replace("⁇", "}", 1)
            predictions.append(line_value)
    assert len(predictions)

    dev = []
    with open(dev_path, "r") as fin:
        for line in fin:
            dev.append(json.loads(line))

    evaluate_predictions(dev, predictions, output_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
