import argparse
import numpy as np
from multiprocessing import Queue, JoinableQueue, Process
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tetrominoes import *
import os as os
import h5py as h5py

def init_episode_dict():

    return {
        'obs': [],
        'action_matrix': [],
        'next_obs': [],
        'state': [],
        'next_state': [],
        'state_matrix': [],
        'next_state_matrix': []
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
    s = 5 #np.random.rand() * 7 + 3
    # better folmula
    x, y = 40.*np.random.rand() + 5, 40.*np.random.rand() + 5.
    return np.array([d, 0, s, x, y, 0])

def get_state_matrix(d, c, s, x, y, shape):
    r = d / 180 * np.pi
    S = np.array([[s, 0., 0.],
                  [0., s, 0.],
                  [0., 0., 1.]])
    R = np.array([[np.cos(r), -np.sin(r), 0.],
                  [np.sin(r), np.cos(r), 0.],
                  [0., 0., 1.]])
    T = np.array([[1., 0., x],
                  [0., 1., y],
                  [0., 0., 1.]])
    return T @ R @ S

def main(args):
    replay_buffer = init_episode_dict()
    state = np.array([360, 0, 7, 32, 32, 0])
    data = Tetrominoes.get_data_by_label(*state)
    if args.one_channel:
        data = data[...,0:1]

    i = 0
    limit = args.num_timesteps
    while i < limit-1:
        next_state = new_uniform_state()
        replay_buffer['state'].append(state)
        replay_buffer['next_state'].append(next_state)
        replay_buffer['obs'].append(data.astype(np.float64))
        data = Tetrominoes.get_data_by_label(*next_state)
        if args.one_channel:
            data = data[...,0:1]
        replay_buffer['next_obs'].append(data.astype(np.float64))

        state_matrix, next_state_matrix, action_matrix = get_state_action_matrices(state, next_state)
        replay_buffer['state_matrix'].append(state_matrix)
        replay_buffer['next_state_matrix'].append(next_state_matrix)
        replay_buffer['action_matrix'].append(action_matrix)
        state = next_state
        i += 1

    # Save replay buffer to disk.
    assert len(replay_buffer['obs']) == len(replay_buffer['action_matrix']) == \
        len(replay_buffer['next_obs']) == len(replay_buffer['state_matrix']) == \
        len(replay_buffer['next_state_matrix'])
    save_single_ep_h5py(replay_buffer, args.fname)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='Total number of episodes to simulate.')
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
