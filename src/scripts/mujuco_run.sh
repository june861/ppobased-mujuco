echo "Mujuco Run Scripts"
# seeds=(1 2 3)
seeds=(1)
envs=(
    "HumanoidStandup-v4"
    # "Hopper-v4"
    # "Reacher-v4"
    # "Walker2d-v4"
    # "InvertedDoublePendulum-v4"
    # "InvertedPendulum-v4"
    # "Humanoid-v4"
    # "HalfCheetah-v4"
    # "Ant-v4"
)
env_exp="cleanrl-mujuco"
sample_action_num=(1 2)

# 循环遍历每个 seed 值，启动 main.py
for seed in "${seeds[@]}"
do
    for env_id in "${envs[@]}"
    do
        for action_num in "${sample_action_num[@]}"
        do
            two_act="logs/${env_id}_${action_num}_act_${seed}.log"
            exp_name="sample${action_num}"
            /usr/bin/python ./src/cleanrl/ppo_mujoco_original.py --seed $seed --exp_name $exp_name \
                --env_id $env_id --sample_action_num $action_num --wandb_project_name $env_exp --total_timesteps 1000000 

        done
        echo "Experiment with seed=$seed finished."
    done
done

