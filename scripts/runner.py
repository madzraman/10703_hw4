'''
	Model-Based Actor-Critic Script: MBPO
	Do not modify.
'''
# pylint: disable=E0401
import sys
from ruamel.yaml import YAML
from src.mbpo import MBPO

if __name__ == "__main__":
	# load the yaml config file
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    # initialize the main class
    agent = MBPO(train_kwargs=v["train_kwargs"],
    	         model_kwargs=v["model_kwargs"],
    	         TD3_kwargs=v["TD3_kwargs"])
    # run the training routine
    agent.train()
