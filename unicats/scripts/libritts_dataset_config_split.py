from glob import glob
from typing import List, Tuple

import argparse
import os

TRAIN_CLEAN_100 = "train-clean-100"
TRAIN_CLEAN_360 = "train-clean-360"
TRAIN_OTHER_500 = "train-other-500"


def _load_utterances(dataset_path: str, au_format: str) -> List[str]:
    utterances = []

    for train_folder in [TRAIN_CLEAN_100, TRAIN_CLEAN_360, TRAIN_OTHER_500]:
        train_path = os.path.join(dataset_path, train_folder)
        utterances.extend(glob(os.path.join(train_path, f"**/*.{au_format}"), recursive=True))

    return utterances

def _assign_utterance_context(utterances: List[str], two_context_config_percentage: float, one_context_config_percentage: float) -> Tuple[List[str], List[str], List[str]]:
    if two_context_config_percentage + one_context_config_percentage >= 1.0:
        raise ValueError("sum of two_context_config_percentage and one_context_config_percentage is greater than 1.0")

    two_context_count = round(len(utterances) * two_context_config_percentage)
    one_context_count = round(len(utterances) * one_context_config_percentage)

    two_context_utterances = utterances[:two_context_count]
    one_context_utterances = utterances[two_context_count:two_context_count + one_context_count]
    no_context_utterances = utterances[two_context_count + one_context_count:]

    return (no_context_utterances, one_context_utterances, two_context_utterances)

def _create_utterance_config_file(utterance_contexts: Tuple[List[str], List[str], List[str]], utterance_config_file_path: str):
    with open(utterance_config_file_path, "w") as utterance_config_fp:
        for index, utterance_context in enumerate(utterance_contexts):
            for utterance in utterance_context:
                utterance_config_fp.write(f"{utterance},{index}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="LibriTTS dataset config splitter",
        usage="This is used to associate each utterance with a configuration based on the percentage defined."
    )

    parser.add_argument(
        "dataset_path",
        help="the path to the dataset to split from"
    )
    parser.add_argument(
        "--two_context_config_percentage",
        help="the percentage of the dataset that should use the two context configuration",
        default=0.6,
        type=float
    )
    parser.add_argument(
        "--one_context_config_percentage",
        help="the percentage of the dataset that should use the one context configuration",
        default=0.3,
        type=float
    )
    parser.add_argument(
        "--au_format",
        help="the audio format of the utterances",
        default="wav",
        type=str
    )
    parser.add_argument(
        "--utterance_config_file_path",
        help="the file path for the utterance config",
        default="./utterance_config.csv",
        type=str
    )

    args = parser.parse_args()

    utterances = _load_utterances(args.dataset_path, args.au_format)

    utterance_contexts = _assign_utterance_context(
        utterances,
        args.two_context_config_percentage,
        args.one_context_config_percentage
    )

    _create_utterance_config_file(utterance_contexts, args.utterance_config_file_path)

if __name__ == "__main__":
    main()