# AdaMatch-pytorch

This is an unofficial implementation of [AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation](https://arxiv.org/abs/2106.04732). Official code is [here](https://github.com/google-research/adamatch) written by Google Research with Jax. [Paper summary](https://smkim7.notion.site/AdaMatch-A-Unified-Approach-to-Semi-Supervised-Learning-and-Domain-Adaptation-Korean-9e6e221cca5e46cb80b9d36e6153553c) and [video presentation](https://www.youtube.com/watch?v=VMZZNaHSTf4&t=162s) are done by myself (in Korean unfortunately).

Step-by-step explanations in Colab notebooks are [here](https://colab.research.google.com/drive/1FY67_4dzLxIcWVkJR6IzwhzKWBcfesO9#scrollTo=SRFCRNEoNerZ).

### Requirements

You can easily install all requirements by the command

```
pip install -r requirements.txt
```

### Datasets

The code supports source to target domain adaptation from SVHN to MNIST (part of DigitFive dataset presented in the paper) . 

### Training

```
python main.py --uratio 3 --tau 0.9
```

The code includes different hyperparameters for config including

* uratio (default=3): Ratio between source and target batch size (uratio * source = target)
* tau (default=0.9): Pseudolabel threshold for Relative confidence threshold

Default all follows from the paper.

### References

* [AdaMatch-pytorch](https://github.com/zysymu/AdaMatch-pytorch) by zysymu

```
@article{berthelot2021adamatch,
  title={AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation},
  author={Berthelot, David and Roelofs, Rebecca and Sohn, Kihyuk and Carlini, Nicholas and Kurakin, Alex},
  journal={arXiv preprint arXiv:2106.04732},
  year={2021}
}
```

