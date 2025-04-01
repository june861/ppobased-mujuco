#!/bin/bash

exp="atari"
seeds=(1 2 3)
update_epochs=(4)
algo="appo"
envs=(
    "AssaultNoFrameskip-v4"
    "AlienNoFrameskip-v4"
    "AmidarNoFrameskip-v4"
    # "AsterixNoFrameskip-v4"
    # "AsteroidsNoFrameskip-v4"
    # "AtlantisNoFrameskip-v4"
    # "BankHeistNoFrameskip-v4"
    # "BattleZoneNoFrameskip-v4"
    # "BeamRiderNoFrameskip-v4"
    # "BerzerkNoFrameskip-v4"
    # "BowlingNoFrameskip-v4"
    # "BoxingNoFrameskip-v4"
    # "BreakoutNoFrameskip-v4"
    # "CentipedeNoFrameskip-v4"
    # "ChopperCommandNoFrameskip-v4"
    # "CrazyClimberNoFrameskip-v4"
    # "DemonAttackNoFrameskip-v4"
    # "DoubleDunkNoFrameskip-v4"
    # "EnduroNoFrameskip-v4"
    # "FishingDerbyNoFrameskip-v4"
    # "FreewayNoFrameskip-v4"
    # "FrostbiteNoFrameskip-v4"
    # "GopherNoFrameskip-v4"
    # "GravitarNoFrameskip-v4"
    # "HeroNoFrameskip-v4"
    # "IceHockeyNoFrameskip-v4"
    # "JamesbondNoFrameskip-v4"
    # "KangarooNoFrameskip-v4"
    # "KrullNoFrameskip-v4"
    # "KungFuMasterNoFrameskip-v4"
    # "MontezumaRevengeNoFrameskip-v4"
    # "MsPacmanNoFrameskip-v4"
    # "NameThisGameNoFrameskip-v4"
    # "PhoenixNoFrameskip-v4"
    # "PitfallNoFrameskip-v4"
    # "PongNoFrameskip-v4"
    # "PrivateEyeNoFrameskip-v4"
    # "QbertNoFrameskip-v4"
    # "RiverraidNoFrameskip-v4"
    # "RoadRunnerNoFrameskip-v4"
    # "RobotankNoFrameskip-v4"
    # "SeaquestNoFrameskip-v4"
    # "SpaceInvadersNoFrameskip-v4"
    # "StarGunnerNoFrameskip-v4"
    # "TennisNoFrameskip-v4"
    # "TimePilotNoFrameskip-v4"
    # "TutankhamNoFrameskip-v4"
    # "UpNDownNoFrameskip-v4"
    # "VentureNoFrameskip-v4"
    # "VideoPinballNoFrameskip-v4"
    # "WizardOfWorNoFrameskip-v4"
    # "YarsRevengeNoFrameskip-v4"
    # "ZaxxonNoFrameskip-v4"
)

# set root dir, it will return "src/" abs path
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
CONFIG_PATH="$PROJECT_ROOT/conf/dis_exp_run.yaml"

# check config file
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# 循环遍历每个 seed 值，启动 main.py
for seed in "${seeds[@]}"
do
    for e in "${update_epochs[@]}"
    do
        for env_id in "${envs[@]}"
        do

            python -m src.main --seed $seed  --yaml $CONFIG_PATH --env_id $env_id --env_type $exp --algo $algo

            # echo "ppo-clip continous action space"
            # ppoclip_yaml="/home/wangchenxu/ppobased-mujuco/src/conf/con_ppoclip_run.yaml"
            # /home/wangchenxu/anaconda3/envs/mujoco_v4/bin/python ./src/algos/mujoco/run.py --seed $seed  --yaml $appo_yaml --env_id $env_id --env_type $exp         
        done
        echo "Experiment with env_id=$env_id seed=$seed finished."
    done
done