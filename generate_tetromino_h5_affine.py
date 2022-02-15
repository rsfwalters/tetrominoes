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

def get_state_action_matrices(state, next_state): 
    state_matrix = get_state_matrix(*state)
    next_state_matrix= get_state_matrix(*next_state)
    action_matrix = next_state_matrix @ np.linalg.inv(state_matrix)
    return state_matrix, next_state_matrix, action_matrix

def new_uniform_state():
    d = np.random.rand() * 360
    x = (64-2*MARGIN)*np.random.rand() + MARGIN
    y = (64-2*MARGIN)*np.random.rand() + MARGIN
    return np.array([d, 0, SCALE, x, y, 0])

def mk_new_g(K, state_matrix):
    success = False 
    while not success:
        g_d = 5
        g_v = np.array([np.random.rand() - 0.5, np.random.rand() - 0.5])
        g_v = g_v / np.linalg.norm(g_v) * 1
        g = get_state_matrix(g_d, 0, 1, g_v[0], g_v[1], 0)

        s = state_matrix.copy()
        for k in range(K-1):
            s = s @ g
            x_out = (s[0, 2] > 64 - MARGIN or s[0, 2] < MARGIN)
            y_out = (s[1, 2] > 64 - MARGIN or s[1, 2] < MARGIN)
            if x_out or y_out:
                break
            if k == K-2:
                success = True
    return g

def get_state_matrix(d, c, s, x, y, shape):
    r = d / 180 * np.pi
    R = np.array([[np.cos(r), -np.sin(r), 0.],
                  [np.sin(r), np.cos(r), 0.],
                  [0., 0., 1.]])
    T = np.array([[1., 0., x],
                  [0., 1., y],
                  [0., 0., 1.]])
    return T @ R

def get_state(state_matrix):
    d = np.arctan2(state_matrix[1, 0], state_matrix[0, 0])
    d = (2 * np.pi + d) if d < 0 else d
    d = d / (2*np.pi) * 360 
    x = state_matrix[0, 2]
    y = state_matrix[1, 2]
    return np.array([d, 0, SCALE, x, y, 0])

def main(args):
    replay_buffer = init_episode_dict(args.K)

    i = 0
    limit = args.num_timesteps
    for i in tqdm(range(limit-1)):
        state = new_uniform_state()
        state_matrix = get_state_matrix(*state)

        g = mk_new_g(args.K, state_matrix)
        # data = Tetrominoes.get_data_by_label(*state)
        data = Tetrominoes.get_data_by_label(*get_state(state_matrix))
        if args.one_channel:
            data = data[...,0:1]

        replay_buffer['state_1'].append(state)
        replay_buffer['state_matrix_1'].append(state_matrix)
        replay_buffer['action_matrix'].append(g)
        replay_buffer['obs_1'].append(data.astype(np.float64))

        s = state_matrix 
        for k in range(2, args.K+1):
            s = s @ g
            replay_buffer[f'state_{k}'].append(get_state(s))
            replay_buffer[f'state_matrix_{k}'].append(s)
            data = Tetrominoes.get_data_by_label(*get_state(s))
            if args.one_channel:
                data = data[...,0:1]
            replay_buffer[f'obs_{k}'].append(data.astype(np.float64))

    save_single_ep_h5py(replay_buffer, args.fname)

MARGIN = 12
SCALE = 6

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

    parsed = parser.parse_args()
    main(parsed)
