# Learning-Prototype-oriented-Set-Representations-for-Meta-Learning
Code for 《Learning Prototype-oriented Set Representations for Meta-Learning》, published in ICLR 2022

Dependencies:

This code is written in python. To use it you will need: python3.8.3， torchvision 0.8.1+cu110， torch 1.7.0+cu110 A recent version of NumPy and SciPy

You can download and pre-process datasets following our paper.

In addition to dealing with the set-structured data, our plug-and-play framework can also be applied to many meta-learning problems. Here, we instantiate the proposed POT to the cases of improving summary networks and few-shot classification.

If you find this repo useful to your project, please cite it with following bib:

@inproceedings{ guo2022learning, title={Learning Prototype-oriented Set Representations for Meta-Learning}, author={Dandan Guo and Long Tian and Minghe Zhang and Mingyuan Zhou and Hongyuan Zha}, booktitle={ICLR 2022: International Conference on Learning Representations}, year={2022}, url={https://openreview.net/forum?id=WH6u2SvlLp4}, pdf={https://openreview.net/pdf?id=WH6u2SvlLp4}, }