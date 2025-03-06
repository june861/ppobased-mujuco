echo "Mujuco Run Scripts"
# seeds=(1 2 3)
seeds=(1)
envs=(
    "HumanoidStandup-v4"
    "Hopper-v4"
    # "Reacher-v4"
    # "Walker2d"
    # "InvertedDoublePendulum-v4"
    # "InvertedPendulum-v4"
    # "Humanoid-v4"
    # "HalfCheetah-v4"
    # "Ant-v4"
)
env_exp="AnchorPPO-Mujuco-supply"
sample_action_num=(1 2)


for env_id in "${envs[@]}"
do
    # 循环遍历每个 seed 值，启动 main.py
    for seed in "${seeds[@]}"
    do
        for action_num in "${sample_action_num[@]}"
        do
            two_act="logs/${env_id}_${action_num}_act_${seed}.log"
            exp_name="sample${action_num}_act_clipcoef01"
            /usr/bin/python ./src/cleanrl/ppo_mujoco_original.py --seed $seed --exp_name $exp_name \
                --env_id $env_id --sample_action_num $action_num --wandb_project_name $env_exp --total_timesteps 1000000 --clip_coef 0.1 --update_epochs 5

            two_act="logs/${env_id}_${action_num}_act_${seed}.log"
            exp_name="sample${action_num}_act_clipcoef02"
            /usr/bin/python ./src/cleanrl/ppo_mujoco_original.py --seed $seed --exp_name $exp_name \
                --env_id $env_id --sample_action_num $action_num --wandb_project_name $env_exp --total_timesteps 1000000  --update_epochs 5 

            echo "Experiment with seed=$seed sample_act_num=$action_num exp_name=$exp_name"
            echo "Log File is ${two_act}"
        done
        echo "Experiment with seed=$seed finished."
    done
done

