# UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding

## Overview
This project is an unofficial implementation of [UniCATS](https://arxiv.org/pdf/2306.07547v2.pdf) paper.

Please note that as the official implementation has been [released](https://github.com/cantabile-kwok/UniCATS-CTX-vec2wav) for the CTX-vec2wav model, this repository will be using the same setup. This provides consistency and compatibility for future updates to the project.

**Note:** Please refer to the official implementations of [CTX-text2vec](https://github.com/cantabile-kwok/UniCATS-CTX-text2vec) and [CTX-vec2wav](https://github.com/cantabile-kwok/UniCATS-CTX-vec2wav).

## Setup
To get started, run the following after going inside the repository's root directory:
```shell
pip install -e .
```

## Dataset
This project is using the **LibriTTS dataset** in the 24 kHz sampling rate. To follow the same dataset splits as in the paper, please follow the steps [on this guide](https://github.com/cantabile-kwok/UniCATS-CTX-vec2wav/blob/main/data_prep.md).

## Credits
- [VQ Diffusion](https://github.com/microsoft/VQ-Diffusion) by Microsoft
- [UniCATS-CTX-vec2wav](https://github.com/cantabile-kwok/UniCATS-CTX-vec2wav) by [cantabile-kwok](https://github.com/cantabile-kwok)

## Citation
```
@article{du2023unicats,
  title={UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding},
  author={Du, Chenpeng and Guo, Yiwei and Shen, Feiyu and Liu, Zhijun and Liang, Zheng and Chen, Xie and Wang, Shuai and Zhang, Hui and Yu, Kai},
  journal={arXiv preprint arXiv:2306.07547},
  year={2023}
}
```
