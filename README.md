# Memory Asymmetry Creates Heteroclinic Orbits to Nash Equilibrium in Learning in Zero-Sum Games
Code for reproducing results in the paper "[Memory Asymmetry Creates Heteroclinic Orbits to Nash Equilibrium in Learning in Zero-Sum Games](https://arxiv.org/abs/2305.13619)".

## About
Learning in games considers how multiple agents maximize their own rewards through repeated games. Memory, an ability that an agent changes his/her action depending on the history of actions in previous games, is often introduced into learning to explore more clever strategies and discuss the decision-making of real agents like humans. However, such games with memory are hard to analyze because they exhibit complex phenomena like chaotic dynamics or divergence from Nash equilibrium. In particular, how asymmetry in memory capacities between agents affects learning in games is still unclear. In response, this study formulates a gradient ascent algorithm in games with asymmetry memory capacities. To obtain theoretical insights into learning dynamics, we first consider a simple case of zero-sum games. We observe complex behavior, where learning dynamics draw a heteroclinic connection from unstable fixed points to stable ones. Despite this complexity, we analyze learning dynamics and prove local convergence to these stable fixed points, i.e., the Nash equilibria. We identify the mechanism driving this convergence: an agent with a longer memory learns to exploit the other, which in turn endows the other's utility function with strict concavity. We further numerically observe such convergence in various initial strategies, action numbers, and memory lengths. This study reveals a novel phenomenon due to memory asymmetry, providing fundamental strides in learning in games and new insights into computing equilibria.

## Installation
This code is written in Python 3. To install the required dependencies, execute the following command:
```
$ pip install numpy
```

## How to use

Please run the target script as follows:

```
$ pytyon3 cMMGA_210.py
```

- [cMMGA_210.py](cMMGA_210.py) outputs the data for Fig. 1 (right panel) and Fig. 3.
- [dMMGA_210.py](dMMGA_210.py), [dMMGA_220.py](dMMGA_220.py), and [dMMGA_221.py](dMMGA_221.py) output the data for Fig. 4-A, B, and C, respectively.
- [dMMGA_310.py](dMMGA_310.py) and [dMMGA_410.py](dMMGA_410.py) output the data for Fig. 5-A and B, respectively.
- Fig. A1 is generated by the statistical data output by all the codes of "dMMGA_[m][nx][ny].py".
- Fig. A2 (the left and center panels) is generated by the statistical data output by all the codes of [dMMGA_310.py](dMMGA_310.py).

## Size of text files
All codes output text files of time series.
The sizes of the files output by the preset parameters are estimated as 10MB in [cMMGA_210.py](cMMGA_210.py), 100MB in [dMMGA_210.py](dMMGA_210.py), 400MB in [dMMGA_220.py](dMMGA_220.py) [dMMGA_221.py](dMMGA_221.py) [dMMGA_310.py](dMMGA_310.py), and 800MB in [dMMGA_410.py](dMMGA_410.py), respectively.

## Citation
If you use our code in your work, please cite our paper:
```
@article{fujimoto2023memory,
  title={Memory Asymmetry Creates Heteroclinic Orbits to Nash Equilibrium in Learning in Zero-Sum Games},
  author={Fujimoto, Yuma and Ariu, Kaito and Abe, Kenshi},
  journal={arXiv preprint arXiv:2305.13619},
  year={2023}
}
```
