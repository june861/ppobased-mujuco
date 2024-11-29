echo "Mujuco Run Scripts"
seeds=(10 21 47)

# 循环遍历每个 seed 值，启动 main.py
for seed in "${seeds[@]}"
do
    echo "Starting experiment with seed=$seed..."
    # 拉起训练
    baseline="logs/baseline_${seed}.log"
    nohup python ./src/cleanrl/ppo_mujoco_original.py --seed $seed --exp_name "baseline" --sample_action_num 1 >> $baseline 2>&1 &

    two_act="logs/two_act_${seed}.log"
    nohup python ./src/cleanrl/ppo_mujoco_original.py --seed $seed --exp_name "two_act" --sample_action_num 2 >> $two_act 2>&1 &


    all_act="logs/all_act_${seed}.log"
    nohup python ./src/cleanrl/ppo_mujoco_original.py --seed $seed --exp_name "all_act"  >> $all_act 2>&1 &
    
    echo "Experiment with seed=$seed finished."
    echo
done

