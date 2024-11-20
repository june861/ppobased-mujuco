# No Representation, No Trust: Connecting Representation, Collapse, and Trust Issues in PPO

**Skander Moalla (EPFL), Andrea Miele (EPFL), Daniil Pyatko (EPFL), Razvan Pascanu (Google DeepMind), Caglar Gulcehre (EPFL)**

**Paper**: [https://openreview.net/forum?id=Wy9UgrMwD0](https://openreview.net/forum?id=Wy9UgrMwD0)

## Abstract

Reinforcement learning (RL)
is inherently rife with non-stationarity since the states and rewards the agent observes
during training depend on its changing policy.
Therefore, networks in deep RL must be capable of adapting to new observations and fitting new targets.
However,
previous works have observed that networks trained under non-stationarity exhibit an inability to continue learning,
termed loss of plasticity, and eventually a collapse in performance.
For off-policy deep value-based RL methods,
this phenomenon has been correlated with a decrease in representation rank and the ability to fit random targets,
termed capacity loss.
Although this correlation has generally been attributed to neural network learning under non-stationarity,
the connection to representation dynamics has not been carefully studied in on-policy policy optimization methods.
In this work,
we empirically study representation dynamics in Proximal Policy Optimization (PPO) on the Atari and MuJoCo environments,
revealing that PPO agents are also affected by feature rank deterioration and capacity loss.
We show that this is aggravated by stronger non-stationarity,
ultimately driving the actor's performance to collapse, regardless of the performance of the critic.
We ask why the trust region, specific to methods like PPO,
cannot alleviate or prevent the collapse
and find a connection between representation collapse and the degradation of the trust region,
one exacerbating the other.
Finally, we present Proximal Feature Optimization (PFO), a novel auxiliary loss that, along with other interventions,
shows that regularizing the representation dynamics mitigates the performance collapse of PPO agents.

![](outputs/github-figures/figure-1.png "Figure 1")
![](outputs/github-figures/figure-3.png "Figure 3")
![](outputs/github-figures/figure-6.png "Figure 6")

## Getting started

### Code and development environment

We support the following methods and platforms for installing the project dependencies and running the code.

- **Docker/OCI-container for AMD64 machines (+ NVIDIA GPUs)**:
  This option works for machines with AMD64 CPUs and NVIDIA GPUs.
  E.g. Linux machines (EPFL HaaS servers, VMs on cloud providers),
  Windows machines with WSL, and clusters running OCI-compliant containers,
  like the EPFL Run:ai (Kubernetes) clusters.

  Follow the instructions in `installation/docker-amd64-cuda/README.md` to install the environment
  then get back here for the rest of the instructions to run the experiments.

  We ran our experiments on NVIDIA 80GB A100 GPUs and NVIDIA 32GB V100 GPUs.
  The Atari experiments require around 3GB of GPU memory with only the training device set to CUDA
  and around 10GB of GPU memory with all devices set to CUDA.
  The plasticity experiments require 80GB of memory to run on GPU.

- **Conda for osx-arm64**
  This option works for macOS machines with Apple Silicon and can leverage MPS acceleration.

  Follow the instructions in `installation/conda-osx-arm64-mps/README.md` to install the environment
  then get back here for the rest of the instructions to run the experiments.

  We ran some toy experiments with the CarPole environment on an M2 MacBook Air.
  You can run the Atari experiments with MPS acceleration setting `device.training=mps`.


### Logging and tracking experiments

We use [Weights & Biases](https://wandb.ai/site) to log and track our experiments.
If you're logged in, your default entity will be used (a fixed entity is not set in the config),
and you can set another entity with the `WANDB_ENTITY` environment variable.
Otherwise, the runs will be anonymous (you don't need to be logged in).

## Reproduction and experimentation

### Reproducing our results

We provide scripts to fully reproduce our work in the `reproducibility-scripts/` directory.
It has a README at its root describing which scripts reproduce which experiments.
This also includes the plotting notebook.

We provide the raw logs and model checkpoints of all our runs
([download link](https://datasets.epfl.ch/claire/no-representation-no-trust-release/release-compressed.tar.gz)) and `.csv` files
([download link](https://datasets.epfl.ch/claire/no-representation-no-trust-release/combined-raw-logs-compressed.tar.gz))
combining all the raw logs to reproduce our plots.
These logs can be used to perform further analysis without re-running the experiments.
Refer to `outputs/README.md` for more information.

Furthermore, all of our runs can be found in one of these three W&B project projects:
1. [Release](https://wandb.ai/claire-labo/no-representation-no-trust-release?nw=wt64llanggc) (all runs with our implementation except for figures 32-35)
2. [Release CleanRL](https://wandb.ai/claire-labo/no-representation-no-trust-release-cleanrl?nw=fk063z9y4v9) (CleanRL runs for replication)
3. [Release Atari Extra](https://wandb.ai/claire-labo/no-representation-no-trust-release-extra?nw=fpi50tbxbc) (runs for figures 32-35)

We provide short W&B reports demonstrating the replicability of runs with CleanRL on
[Atari](https://wandb.ai/claire-labo/no-representation-no-trust-release-cleanrl/reports/Replication-of-Atari-with-CleanRL--Vmlldzo5OTQwMzMy) and [MuJoCo](https://wandb.ai/claire-labo/no-representation-no-trust-release-cleanrl/reports/Replication-of-MuJoCo-with-CleanRL--Vmlldzo5OTQwMDEx).
We also provide a short W&B report to demonstrate the collapse phenomenon on MuJoCo with minimal modifications to the CleanRL code [here](https://wandb.ai/claire-labo/no-representation-no-trust-release-cleanrl/reports/CleanRL-Original-Setting-MuJoCo-Collapse--Vmlldzo5OTQwMTM0).

![](outputs/github-figures/replication.png "Replication with CleanRL")

### Experiment with different configurations

The default configuration for each script is stored in the `configs/` directory.
They are managed by [Hydra](https://hydra.cc/docs/intro/).
You can experiment with different configurations by passing the relevant arguments.
You can get examples of how to do so in the `reproducibility-scripts/` directory.

## Repository structure

Below, we give a description of the main files and directories in this repository.

```
└── src/                                        # Source code.
    └── po_dynamics                             # Our package.
        ├── configs/                            # configuration files for environments, models, and algorithms.
        ├── modules/                            # Environment builders, models, losses, and logged metrics.
        ├── solve.py                            # Main script to train models.
        ├── capacity.py                         # Script to compute capacity loss.
        └── toy_problem.py                      # Script to run the toy setting of Figure 5.

    └─── cleanrl/                               # Scripts to mimic our implementation in CleanRL (with very limited features).
                                                # Used to reproduce and verify that our implementation doesn't have a random bug.
        ├── ppo_atari_original.py               # CleanRL's PPO implementation on Atari.
        ├── ppo_mujoco_original.py              # CleanRL's PPO implementation on MuJoCo.
        ├── ppo_atari_1model.py                 # Modified CleanRL's PPO to have the same setting as out codebase.
        ├── ppo_atari_2models.py                # Same but with separate actor and critic.
        ├── ppo_mujoco_torch.py                 # Modified CleanRL's PPO to have the same setting as out codebase.
        └── ppo_mujoco_original_collapse.py     # Minimal changes to CleanRL's PPO implementation on MuJoCo with collapse.

```

## Contributing

This repository is a frozen copy of the codebase to reproduce the results of the paper.
You can fork the repository and use the following code-quality tools to contribute to the codebase.

We use [`pre-commit`](https://pre-commit.com) hooks to ensure high-quality code.
Make sure it's installed on the system where you're developing
(it is in the dependencies of the project, but you may be editing the code from outside the development environment.
If you have conda you can install it in your base environment, otherwise, you can install it with `brew`).
Install the pre-commit hooks with

```bash
# When in the PROJECT_ROOT.
pre-commit install --install-hooks
```

Then every time you commit, the pre-commit hooks will be triggered.
You can also trigger them manually with:

```bash
pre-commit run --all-files
```
