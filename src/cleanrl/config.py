import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Anchor PPO Exeperiment")
    
    # Experiment arguments
    parser.add_argument('--exp_name', type=str, default=None, required=True,
                        help="The name of this experiment")
    parser.add_argument('--seed', type=int, default=1, help="Seed of the experiment")
    parser.add_argument('--torch_deterministic', type=bool, default=True, 
                        help="If toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument('--cuda', type=bool, default=True, help="If toggled, cuda will be enabled by default")
    parser.add_argument('--track', action="store_false", default=True, help="If toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument('--wandb_project_name', type=str, default="cleanRL-mujuco-v2", help="The wandb's project name")
    parser.add_argument('--wandb_entity', type=str, default=None, help="The entity (team) of wandb's project")
    parser.add_argument('--capture_video', type=bool, default=False, 
                        help="Whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument('--save_model', type=bool, default=False, help="Whether to save model into the `runs/{run_name}` folder")
    parser.add_argument('--upload_model', type=bool, default=False, help="Whether to upload the saved model to huggingface")
    parser.add_argument('--hf_entity', type=str, default="", help="The user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument('--env_id', type=str, default="HalfCheetah-v4", help="The id of the environment")
    parser.add_argument('--total_timesteps', type=int, default=1000000, help="Total timesteps of the experiments")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help="The learning rate of the optimizer")
    parser.add_argument('--num_envs', type=int, default=8, help="The number of parallel game environments")
    parser.add_argument('--num_steps', type=int, default=2048, help="The number of steps to run in each environment per policy rollout")
    parser.add_argument('--anneal_lr', type=bool, default=True, help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gamma', type=float, default=0.99, help="The discount factor gamma")
    parser.add_argument('--gae_lambda', type=float, default=0.95, help="The lambda for the general advantage estimation")
    parser.add_argument('--num_minibatches', type=int, default=32, help="The number of mini-batches")
    parser.add_argument('--update_epochs', type=int, default=10, help="The K epochs to update the policy")
    parser.add_argument('--norm_adv', type=bool, default=True, help="Toggles advantages normalization")
    parser.add_argument('--clip_coef', type=float, default=0.2, help="The surrogate clipping coefficient")
    parser.add_argument('--clip_vloss', type=bool, default=True, help="Toggles whether or not to use a clipped loss for the value function")
    parser.add_argument('--ent_coef', type=float, default=0.0, help="Coefficient of the entropy")
    parser.add_argument('--vf_coef', type=float, default=0.5, help="Coefficient of the value function")
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help="The maximum norm for the gradient clipping")
    parser.add_argument('--target_kl', type=float, default=None, help="The target KL divergence threshold")

    # to be filled in runtime
    parser.add_argument('--batch_size', type=int, default=0, help="The batch size (computed in runtime)")
    parser.add_argument('--minibatch_size', type=int, default=0, help="The mini-batch size (computed in runtime)")
    parser.add_argument('--num_iterations', type=int, default=0, help="The number of iterations (computed in runtime)")

    # Action sample parameters
    parser.add_argument('--sample_action_num', type=int, default=None, help="Number of actions to sample")
    parser.add_argument('--wandb_group', type=str, default=None, help="the wandb group name")

    
    return parser