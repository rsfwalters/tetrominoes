#!/bin/sh

N_train=60000
N_test=5000
K=6
pi="3.141592653589793"
max_rotation=$(echo "scale=4; $pi*2/4" | bc) # (360 / (K - 1 - 1)) # -1 so that we have more chance of covering all the space
max_translation=0
max_color_rotation=0
max_acceleration=$(echo "scale=4; $pi*2/60" | bc)
init_x=0
init_y=0
init_color=$(echo "scale=4; -1*$pi" | bc)
shape='[1]'
group='so2'

mode='constant_velocity'
python3 tetrominoes/generate_tetromino.py --num-timesteps=$N_train --K=$K --fname="../group-vae/data/tetrominoes_K=${K}_${N_train}_${group}_1c_shape=1_cv_maxTheta=2pi-4_maxX=0_maxC=0_maxAcc=2pi-60_train.h5" --seed=0 --max-rotation=$max_rotation --max-translation=$max_translation --max-color-rotation=$max_color_rotation --init-x=$init_x --init-y=$init_y --init-color=$init_color --one-channel --shape=$shape --mode=$mode --max-acceleration=$max_acceleration
python3 tetrominoes/generate_tetromino.py --num-timesteps=$N_test --K=$K --fname="../group-vae/data/tetrominoes_K=${K}_${N_test}_${group}_1c_shape=1_cv_maxTheta=2pi-4_maxX=0_maxC=0_maxAcc=2pi-60_test.h5" --seed=1 --max-rotation=$max_rotation --max-translation=$max_translation --max-color-rotation=$max_color_rotation --init-x=$init_x --init-y=$init_y --init-color=$init_color --one-channel --shape=$shape --mode=$mode --max-acceleration=$max_acceleration


mode='small_acceleration'
python3 tetrominoes/generate_tetromino.py --num-timesteps=$N_train --K=$K --fname="../group-vae/data/tetrominoes_K=${K}_${N_train}_${group}_1c_shape=1_sa_maxTheta=2pi-4_maxX=0_maxC=0_maxAcc=2pi-60_train.h5" --seed=0 --max-rotation=$max_rotation --max-translation=$max_translation --max-color-rotation=$max_color_rotation --init-x=$init_x --init-y=$init_y --init-color=$init_color --one-channel --shape=$shape --mode=$mode --max-acceleration=$max_acceleration
python3 tetrominoes/generate_tetromino.py --num-timesteps=$N_test --K=$K --fname="../group-vae/data/tetrominoes_K=${K}_${N_test}_${group}_1c_shape=1_sa_maxTheta=2pi-4_maxX=0_maxC=0_maxAcc=2pi-60_test.h5" --seed=1 --max-rotation=$max_rotation --max-translation=$max_translation --max-color-rotation=$max_color_rotation --init-x=$init_x --init-y=$init_y --init-color=$init_color --one-channel --shape=$shape --mode=$mode --max-acceleration=$max_acceleration
