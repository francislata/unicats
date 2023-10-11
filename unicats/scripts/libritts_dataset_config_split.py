from glob import glob
from typing import List

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

def _assign_utterance_configs(
    utterances: List[str],
    two_context_config_percentage: float,
    one_context_config_percentage: float,
    no_context_config_percentage: float
):
    two_context_count = round(len(utterances) * two_context_config_percentage)
    one_context_count = round(len(utterances) * one_context_config_percentage)
    no_context_count = round(len(utterances) * no_context_config_percentage)

    print(f"--total: {len(utterances)} ---two: {two_context_count} --one: {one_context_count} --zero: {no_context_count}")


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
        "--no_context_config_percentage",
        help="the percentage of the dataset that should use the no context configuration",
        default=0.1,
        type=float
    )
    parser.add_argument(
        "--au_format",
        help="the audio format of the utterances",
        default="wav",
        type=str
    )

    args = parser.parse_args()

    utterances = _load_utterances(args.dataset_path, args.au_format)

    _assign_utterance_configs(
        utterances,
        args.two_context_config_percentage,
        args.one_context_config_percentage,
        args.no_context_config_percentage
    )

if __name__ == "__main__":
    main()