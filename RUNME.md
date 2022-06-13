# Inventory Control Installation and Run Instructions

Installation instructions and code are based on this [OpenAI Gym environment](https://github.com/paulhendricks/gym-inventory) source.

## Installation

1) Install [PyTorch](https://pytorch.org/) and [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/).

2) Install gym-inventory by cloning this repository into your working directory.

3) Run the following code:\
`cd gym-inventory`\
`pip install -e .`

## Run
`inventory_control_random.py`, `inventory_control_A2Ctrain.py`, and `inventory_control_PPOtrain.py` can be opened and run in any Python IDE. We recommend VSCode.\
Note: To run Tensorboard, [Tensorflow](https://www.tensorflow.org/install/) must be installed.
