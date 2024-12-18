echo "Mujuco Run Scripts"
seeds=(1 2 3 4 5)
envs=(
    # "HalfCheetah-v4"
    # "HumanoidStandup-v4"
    # "Humanoid-v4"
    # "Walker2d"
    # "Swimmer"
    # "Ant-v4"
    # "InvertedDoublePendulum-v4"
    # "Hopper-v4"
    "Reacher-v4"
)
sample_action_num=(1 2 4 8 16 32)

for env_id in "${envs[@]}"
do
    # 循环遍历每个 seed 值，启动 main.py
    for seed in "${seeds[@]}"
    do
        # echo "Starting experiment with env_id=$env_id seed=$seed..."
        # # 拉起训练
        # baseline="logs/${env_id}_baseline_${seed}.log"
        # nohup python ./src/cleanrl/ppo_mujoco_original.py --seed $seed --exp_name "baseline" \
        #     --env_id $env_id --sample_action_num 1 >> $baseline 2>&1 &
        
        # all_act="logs/${env_id}_all_act_${seed}.log"
        # nohup python ./src/cleanrl/ppo_mujoco_original.py --seed $seed --exp_name "all_act" \
        #     --env_id $env_id >> $all_act 2>&1 &

        for action_num in "${sample_action_num[@]}"
        do
            two_act="logs/${env_id}_${action_num}_act_${seed}.log"
            exp_name="sample${action_num}_act"
            nohup /usr/bin/python ./src/cleanrl/ppo_mujoco_original.py --seed $seed --exp_name $exp_name \
                --env_id $env_id --sample_action_num $action_num >> $two_act 2>&1 &
            
            echo "Experiment with seed=$seed sample_act_num=$action_num exp_name=$exp_name"
            echo
        done
        echo "Experiment with seed=$seed finished."
    done
done

