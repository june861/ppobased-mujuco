import argparse
import sys
import os
import torch
import time
import yaml
from base import getLogger
from prettytable import PrettyTable

def mujoco_conf():
    parser = argparse.ArgumentParser(description="Anchor PPO Exeperiment")
    # toml config
    parser.add_argument('--yaml', type=str, default=None, help="configuration file to launch exp through toml file!")
    parser.add_argument('--env_id', type=str, default="BreakoutNoFrameskip-v4", help="The id of the environment")
    parser.add_argument('--seed', type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument('--algo', type=str, default="ppo-clip", help="Which algorithm to test", choices=["appo-all", "appo-two","ppo-clip"])
    args, remaining_argv = parser.parse_known_args()

    if not os.path.exists(args.yaml):
        raise FileNotFoundError(f"{args.yaml}")
    with open(args.yaml, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Re-create argument parser to handle config values
    parser = argparse.ArgumentParser(description="Anchor PPO Experiment with Config Overwrite")
    parser.add_argument('--env_id', type=str, default=args.env_id, help="The id of the environment")
    parser.add_argument('--yaml', type=str, default=args.yaml, help="configuration file to launch exp through toml file!")
    parser.add_argument('--seed', type=str, default="BreakoutNoFrameskip-v4", help="The id of the environment")
    parser.add_argument('--algo', type=str, default=args.algo, help="Which algorithm to test", choices=["appo-all", "appo-two","ppo-clip"])
    for BigClass, Dict_ in yaml_config.items():
        for param, value in Dict_.items():
            parser.add_argument(f"--{param}", type=type(value), default=value)

    # Parse remaining arguments, allowing command-line overrides
    args = parser.parse_args(remaining_argv)
    args.exp_name = f'{args.env_id}_{args.algo}_seed{args.seed}_update{args.update_epochs}_clipcoef{args.clip_coef}'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.logger = getLogger(f"{args.env_id}_{int(time.time())}", "colored")
    # Compute runtime parameters dynamically
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Print configuration in a table
    config_table = PrettyTable()
    config_table.field_names = ["Parameter", "Value"]
    for arg in vars(args):
        config_table.add_row([arg, getattr(args, arg)])
    args.logger.info(f"\n🔍 Configuration Table\n{config_table}")

    return args
