# NTU-1091-Security and Privacy of Machine Learning: Adversarial Defense

## Overview
the attacker will use adversarial examples (up to epsilon=8 in the L_infinity norm) to attack your model, which is under white-box setting.
In this project, I train a robust model for CIFAR-10 that can defend the adversarial examples.


## Model
My defense pipeline include image preprocessing, and adversarial training, etc. The detail can be found in the report *hw2_r09942066.pdf*, please let me know if you need the model weight(email me), then I will send you the download link. 

### Defense Model
* resnet50

## Attack for evaluation 
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
### For Evaluating
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
67.2% accuracy for PGD attack(max epsilon=0.3) under white-box setting

# Contact 
If there are any problem that make you cannot execute the program, please contact<br>
**R09942066@ntu.edu.tw** (I have already test under the specify env).
