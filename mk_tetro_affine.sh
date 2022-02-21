N_data=100
K=10
min_rotation=0
max_rotation=36
min_translation=0
max_translation=2
init_x=0
init_y=0

python3 generate_tetromino_h5_affine.py --fname data/tetrominoes_1c_K=${K}_${N_data}_train.h5 --one-channel --K=$K --num_timesteps $N_data --seed 1 --min-rotation=$min_rotation --max-rotation=$max_rotation --min-translation=$min_translation --max-translation=$max_translation #--init-rotation=0 #--init-x=$init_x --init-y=$init_y
python3 generate_tetromino_h5_affine.py --fname data/tetrominoes_1c_K=${K}_${N_data}_test.h5 --one-channel --K=$K --num_timesteps $N_data --seed 2 --min-rotation=$min_rotation --max-rotation=$max_rotation --min-translation=$min_translation --max-translation=$max_translation #--init-rotation=0 #--init-x=$init_x --init-y=$init_y

