# E-SAGE: Explainability-based Defense Against Backdoor Attacks on Graph Neural Networks
An official implementation of "E-SAGE: Explainability-based Defense Against Backdoor Attacks on Graph Neural Networks"[[paper]]()

Our method is implemented based on the [[UGBA]](https://github.com/ventr1c/UGBA) framework. To reproduce our work, you also need to install Pytorch from [[source]](https://github.com/pytorch/pytorch) to get the  explainability component.
You can also install only relevant packages, which will not conflict with existing environments.

To perform the relevant experiments, you need to first reproduce the relevant environment of UGBA (see the previous link, this is very simple), then install the relevant explainability packages, and finally copy our files to the main folder of UGBA.
* `./run_adaptive_exp_new.py`: The program to defense UGBA attack.Use " --defense_mode='exp' " to select our defense method for experimentation.
* `./run_distribute.py`:The program used to draw the distribution map in the paper can be used to study the impact of backdoor attacks.
