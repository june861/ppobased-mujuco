#!/bin/bash

# Round 1
# nohup bash src/scripts/appo_atari.sh    > src/scripts/appo_atari_1.log    2>&1 &
# nohup bash src/scripts/appo_atari_1.sh  > src/scripts/appo_atari_1_1.log  2>&1 &
# nohup bash src/scripts/ppo2_atari.sh    > src/scripts/ppo2_atari_1.log    2>&1 &
# nohup bash src/scripts/ppo1_atari.sh    > src/scripts/ppo1_atari_1.log    2>&1 &

# Round 2
# nohup bash src/scripts/appo_atari.sh    > src/scripts/appo_atari_2.log    2>&1 &
# nohup bash src/scripts/appo_atari_1.sh  > src/scripts/appo_atari_1_2.log  2>&1 &
# nohup bash src/scripts/ppo2_atari.sh    > src/scripts/ppo2_atari_2.log    2>&1 &
# nohup bash src/scripts/ppo1_atari.sh    > src/scripts/ppo1_atari_2.log    2>&1 &

# # Round 3
# nohup bash src/scripts/appo_atari.sh    > src/scripts/appo_atari_3.log    2>&1 &
# nohup bash src/scripts/appo_atari_1.sh  > src/scripts/appo_atari_1_3.log  2>&1 &
# nohup bash src/scripts/ppo2_atari.sh    > src/scripts/ppo2_atari_3.log    2>&1 &
# nohup bash src/scripts/ppo1_atari.sh    > src/scripts/ppo1_atari_3.log    2>&1 &

# # Round 4
nohup bash src/scripts/appo_atari.sh    > src/scripts/appo_atari_4.log    2>&1 &
nohup bash src/scripts/appo_atari_1.sh  > src/scripts/appo_atari_1_4.log  2>&1 &
nohup bash src/scripts/ppo2_atari.sh    > src/scripts/ppo2_atari_4.log    2>&1 &
nohup bash src/scripts/ppo1_atari.sh    > src/scripts/ppo1_atari_4.log    2>&1 &
