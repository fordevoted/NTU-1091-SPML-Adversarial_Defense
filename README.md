# Security and Privacy of Machine Learning HW2: Black Box Defense

## Overview
In this homework, you need to train a robust model for CIFAR-10 that can defend the adversarial examples.
That the attacker will use adversarial examples (up to epsilon=8 in the L_infinity norm) to attack your model.
## Model
### Defense Model
* resnet50

## Attack
* PGD
* FGSM

## Defense technique
* PGD adversarial training
* data pre-processing
* SLQ
* proposed method

## Requirement
* pytorch 1.5.0
* python 3
* built-in modules and numpy
* opencv
**hw2.py has been tested under this env**
## Usage
### For TA Evaluating
run `python hw2.py [image folder]`
##  Testing
`cd src/example`<br>
`python adversarial_testing.py`
## Ablation Experiment
`cd src/example`<br>
`python ablation_study.py`
## Generate PGD examples
`cd src/example`<br>
`python test_PGD.py`

##  Result 
67.2% accuracy for PGD attack(max epsilon=0.3)

# Contact 
If there are any problem that make you cannot execute the program, please contact<br>
**R09942066@ntu.edu.tw** before penalty, please QQ(I have already test under the specify env).