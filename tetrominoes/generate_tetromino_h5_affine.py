import argparse
import numpy as np
from multiprocessing import Queue, JoinableQueue, Process
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tetrominoes import *
import os as os
import h5py as h5py
from tqdm import tqdm

def init_episode_dict(K):

    return {
        **{f'obs_{k}': [] for k in range(1, K+1)},
        **{f'state_{k}': [] for k in range(1, K+1)},
        **{f'state_matrix_{k}': [] for k in range(1, K+1)},
        'action_matrix': [],
    }


def save_single_ep_h5py(array_dict, fname):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        grp = hf.create_group("data")
        for key in array_dict.keys():
            grp.create_dataset(key, data=array_dict[key])

# def get_state_action_matrices(state, next_state): 
#     state_matrix = get_state_matrix(*state)
#     next_state_matrix= get_state_matrix(*next_state)
#     action_matrix = next_state_matrix @ np.linalg.inv(state_matrix)
#     return state_matrix, next_state_matrix, action_matrix

def check_out_of_bounds(x, y):
    x_out = (x > 32 - MARGIN or x < MARGIN - 32)
    y_out = (y > 32 - MARGIN or y < MARGIN - 32)
    return x_out, y_out


def new_uniform_state_matrix(init_rotation, init_x, init_y):
    r = np.random.rand() * np.pi * 2 if init_rotation is None else init_rotation
    x = (64-2*MARGIN)*np.random.rand() + MARGIN - 32 if init_x is None else init_x
    y = (64-2*MARGIN)*np.random.rand() + MARGIN - 32 if init_y is None else init_y
    x_out, y_out = check_out_of_bounds(x, y)
    if x_out or y_out:
        raise ValueError("initial position is out of bounds")
    return np.array([[np.cos(r), -np.sin(r), x],
                     [np.sin(r), np.cos(r), y],
                     [0, 0, 1]])

def get_state(state_matrix):
    r = np.arctan2(state_matrix[1, 0], state_matrix[0, 0])
    # r = np.min(np.stack((abs(theta), abs(2*np.pi + theta)), axis=-1), axis=-1)
    d = r * 180 / np.pi
    x = state_matrix[0, 2]
    y = state_matrix[1, 2]
    return np.array([d, 0, SCALE, x + 32, y + 32, 0])

def get_state_matrix(d, c, s, x, y, shape):
    r = d / 180 * np.pi
    R = np.array([[np.cos(r), -np.sin(r), 0.],
                  [np.sin(r), np.cos(r), 0.],
                  [0., 0., 1.]])
    T = np.array([[1., 0., x],
                  [0., 1., y],
                  [0., 0., 1.]])
    return T @ R

def new_state_action_matrix(K, min_rotation, max_rotation, min_translation, max_translation, init_rotation=None, init_x=None, init_y=None, object_centric=False):
    success = False 
    while not success:
        state_matrix = new_uniform_state_matrix(init_rotation, init_x, init_y)

        g_d = np.random.rand() * 2 - 1
        g_d = g_d * (max_rotation - min_rotation) + min_rotation
    
        g_v = np.random.rand(2) - 0.5
        g_v = g_v / np.linalg.norm(g_v) * (max_translation - min_translation) + min_translation
        g = get_state_matrix(g_d, 0, 1, g_v[0], g_v[1], 0)

        s = state_matrix.copy()
        for k in range(K-1):
            if object_centric:
                s = s @ g
            else:
                s = g @ s
            x_out, y_out = check_out_of_bounds(s[0, 2], s[1, 2])
            if x_out or y_out:
                break
            if k == K-2:
                success = True
    return state_matrix, g


def main(args):
    replay_buffer = init_episode_dict(args.K)

    i = 0
    limit = args.num_timesteps
    for i in tqdm(range(limit-1)):
        state_matrix, g = new_state_action_matrix(args.K, args.min_rotation, args.max_rotation, args.min_translation, args.max_translation, args.init_rotation, args.init_x, args.init_y, object_centric=args.object_centric)
        state = get_state(state_matrix)
        data = Tetrominoes.get_data_by_state_matrix(state_matrix, scale=SCALE)
        if args.one_channel:
            data = data[...,0:1]

        replay_buffer['state_1'].append(state)
        replay_buffer['state_matrix_1'].append(state_matrix)
        replay_buffer['action_matrix'].append(g)
        replay_buffer['obs_1'].append(data.astype(np.float64))

        s = state_matrix 
        for k in range(2, args.K+1):
            if args.object_centric:
                s = s @ g
            else:
                s = g @ s
            replay_buffer[f'state_{k}'].append(get_state(s))
            replay_buffer[f'state_matrix_{k}'].append(s)
            data = Tetrominoes.get_data_by_state_matrix(s, scale=SCALE)
            if args.one_channel:
                data = data[...,0:1]
            replay_buffer[f'obs_{k}'].append(data.astype(np.float64))

    save_single_ep_h5py(replay_buffer, args.fname)

MARGIN = 12
SCALE = 6
# World Frame -> at zero, (0, 1), (1, 0)
# Simulator Frame -> at (-32, -32), (0, 1), (1, 0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='Total number of episodes to simulate.')
    parser.add_argument('--K', type=int, default=2,
                        help='Sequence length.')
    parser.add_argument('--fname', type=str, default='test.h5',
                        help='Save path for replay buffer.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--num_jobs', type=int, default=2)
    parser.add_argument('--all_actions', default=False, action='store_true')

    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--one-channel', default=False, action='store_true')
    parser.add_argument('--object_centric', action="store_true")
    parser.add_argument('--min-rotation', type=float, default=0.)
    parser.add_argument('--max-rotation', type=float, default=5.)
    parser.add_argument('--min-translation', type=float, default=0.)
    parser.add_argument('--max-translation', type=float, default=2.)
    parser.add_argument('--init-rotation', type=float, default=None)
    parser.add_argument('--init-x', type=float, default=None)
    parser.add_argument('--init-y', type=float, default=None)

    parsed = parser.parse_args()
    main(parsed)
