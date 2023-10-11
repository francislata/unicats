from glob import glob
from typing import Set
from tqdm import tqdm

import argparse
import os

def _parse_test_utterances(file_path: str) -> Set[str]:
    utterances = []

    with open(file_path, "r") as split_file_fp:
        lines = split_file_fp.readlines()

        for line in lines:
            utterances.extend(line.strip().split(" "))

    return set(utterances)

def _move_test_utterances(
    test_utterances: Set[str],
    dataset_path: str,
    test_set_path: str,
    dry_run: bool = False
):
    for test_utterance in tqdm(test_utterances, desc="utterances"):
        results = glob(f'{dataset_path}/**/{test_utterance}.wav', recursive=True)

        if len(results) == 1:
            src_file_path = results[0]
            filename = os.path.basename(src_file_path)
            dest_file_path = os.path.join(test_set_path, filename)

            print(f"Moving {src_file_path} to {dest_file_path}")

            if not dry_run:
                if not os.path.exists(test_set_path):
                    print(f"Making test set path {test_set_path}")
                    os.makedirs(test_set_path)

                os.rename(src_file_path, dest_file_path)


def main():
    parser = argparse.ArgumentParser(
        prog="LibriTTS test set splitter",
        usage="This is used to generate the test sets specified in the UniCATS paper."
    )

    parser.add_argument(
        "dataset_path",
        help="the path to the dataset to split from",
        type=str
    )
    parser.add_argument(
        "--utterance_split_file",
        help="the file which contains the utterances to split from the dataset.",
        type=str
    )
    parser.add_argument(
        "--test_set_path",
        default="./test_set",
        help="the path of the generated test set",
        type=str
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="performs a dry run of the creation of LibriTTS test sets."
    )

    args = parser.parse_args()

    test_utterances = _parse_test_utterances(args.utterance_split_file)

    _move_test_utterances(test_utterances, args.dataset_path, args.test_set_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()