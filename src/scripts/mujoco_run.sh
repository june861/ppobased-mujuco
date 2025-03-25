echo "Mujoco Run Scripts"
# seeds=(1 2 3)
seeds=(1 2 3)
update_epochs=(10)
envs=(
    # "HumanoidStandup-v4"
    # "Humanoid-v4"
    # "HalfCheetah-v4"
    # "Ant-v4"

    "Hopper-v4"
    "Reacher-v4"
    "Walker2d-v4"
    "InvertedDoublePendulum-v4"
    "InvertedPendulum-v4"
)
env_exp="cleanrl-mujuco-v2"
sample_action_num=(2)

# 循环遍历每个 seed 值，启动 main.py
for seed in "${seeds[@]}"
do
    for e in "${update_epochs[@]}"
    do
        for env_id in "${envs[@]}"
        do
            for action_num in "${sample_action_num[@]}"
            do

                two_act="logs/${env_id}_${action_num}_act_${seed}.log"
                exp_name="sample${action_num}_${e}"
                /home/weijun.luo/.conda/envs/mujoco_py311/bin/python ./src/cleanrl/ppo_mujoco_original.py --seed $seed --exp_name $exp_name \
                    --env_id $env_id --sample_action_num $action_num --wandb_project_name $env_exp --total_timesteps 10000000 --update_epoch $e --track
            done            
        done
        echo "Experiment with seed=$seed finished."
    done
done

