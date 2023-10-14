# UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding

## Overview
This project is an unofficial implementation of [UniCATS](https://arxiv.org/pdf/2306.07547v2.pdf) paper.

## Test sets
To create the test sets, the script `unicats/scripts/libritts_test_set_split.py` can be used. Ensure to download the list of utterances from the official GitHub page: [test set A](https://cpdu.github.io/unicats/resources/testsetA_utt2prompt) and [test set B](https://cpdu.github.io/unicats/resources/testsetB_utt2prompt).

## Assign context configs to training set
As specified in the paper, they have assigned certain percentages with different configurations:
- Two context: Context A, x<sub>0</sub>, Context B for 60% of dataset
- One context: Context A, x<sub>0</sub> for 30% of dataset
- No context: x<sub>0</sub> for 10% of dataset

The script `unicats/scripts/libritts_dataset_config_split.py` can be used to create a `csv` file that will contain the utterance file path and the corresponding configuration it should use during training.

## Credits
- [VQ Diffusion](https://github.com/microsoft/VQ-Diffusion)