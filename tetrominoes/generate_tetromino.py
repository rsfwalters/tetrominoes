import argparse
import numpy as np
from multiprocessing import Queue, JoinableQueue, Process
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tetrominoes import *
import os as os
import h5py as h5py
from tqdm import tqdm

MARGIN = 12
SCALE = 8
# World Frame -> at zero, (0, 1), (1, 0)
# Simulator Frame -> at (-32, -32), (0, 1), (1, 0)

def init_episode_dict(K):
    replay_buffer = {}
    for k in range(1, K+1):
        replay_buffer['obs_%d' % k] = []
        replay_buffer['action_matrix_%d' % k] = []
        replay_buffer['state_matrix_%d' % k] = []
        replay_buffer['label_%d' % k] = [] # This is basically same as state_matrix but in vector format
    del replay_buffer['action_matrix_%d' % K] # there are only K-1 steps
    return replay_buffer

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

def check_out_of_bounds(x, y):
    x_out = (x > 32 - MARGIN or x < MARGIN - 32)
    y_out = (y > 32 - MARGIN or y < MARGIN - 32)
    return x_out, y_out

def new_uniform_state_matrix(init_rotation, init_x, init_y, init_color_rotation):
    """
    Matrix has the form:
    [affine, 0
     0     , so(2)]
     5 x 5
    """
    r = np.random.uniform(-np.pi, np.pi) if init_rotation is None else init_rotation
    x = (64-2*MARGIN)*np.random.rand() + MARGIN - 32 if init_x is None else init_x
    y = (64-2*MARGIN)*np.random.rand() + MARGIN - 32 if init_y is None else init_y
    c = np.random.uniform(-np.pi, np.pi) if init_color_rotation is None else init_color_rotation
    x_out, y_out = check_out_of_bounds(x, y)
    if x_out or y_out:
        raise ValueError("initial position is out of bounds")
    return np.array([[np.cos(r), -np.sin(r), x, 0,          0        ],
                     [np.sin(r),  np.cos(r), y, 0,          0        ],
                     [0,          0,         1, 0,          0        ],
                     [0,          0,         0, np.cos(c), -np.sin(c)],
                     [0,          0,         0, np.sin(c),  np.cos(c)]])

def get_label_vec(state_matrix, shape):
    r = np.arctan2(state_matrix[1, 0], state_matrix[0, 0])
    # r = np.min(np.stack((abs(theta), abs(2*np.pi + theta)), axis=-1), axis=-1)
    x = state_matrix[0, 2]
    y = state_matrix[1, 2]
    c = np.arctan2(state_matrix[4, 3], state_matrix[3, 3])
    return np.array([r, c, SCALE, x + 32, y + 32, shape])

def get_state_matrix(r, c, s, x, y, shape):
    R = np.array([[np.cos(r), -np.sin(r), 0., 0.,         0.],
                  [np.sin(r),  np.cos(r), 0., 0.,         0.],
                  [0.,         0.,        1., 0.,         0.],
                  [0.,         0.,        0., 1.,         0.],
                  [0.,         0.,        0., 0.,         1.]])
    T = np.array([[1.,         0.,        x,  0.,         0.],
                  [0.,         1.,        y,  0.,         0.],
                  [0.,         0.,        1., 0.,         0.],
                  [0.,         0.,        0., 1.,         0.],
                  [0.,         0.,        0., 0.,         1.]])
    C = np.array([[1.,         0.,        0., 0.,         0.],
                  [0.,         1.,        0., 0.,         0.],
                  [0.,         0.,        1., 0.,         0.],
                  [0.,         0.,        0., np.cos(c), -np.sin(c)],
                  [0.,         0.,        0., np.sin(c),  np.cos(c)]])
    return T @ R @ C

def new_state_action_matrix(K, min_rotation, max_rotation, min_translation,
                            max_translation, min_color_rotation, max_color_rotation,
                            init_rotation=None, init_x=None, init_y=None,
                            init_color_rotation=None, object_centric=False,
                            mode='constant_velocity', min_acceleration=0,
                            max_acceleration=0):
    """
    Sample a valid action matrix:
    Reject action matrices that lead to out of bound examples
    """
    success = False
    while not success:
        state_matrix = new_uniform_state_matrix(init_rotation, init_x, init_y, init_color_rotation)

        g_r = np.random.uniform(min_rotation, max_rotation)
        g_r = g_r * np.random.choice([-1,1])

        g_c = np.random.uniform(min_color_rotation, max_color_rotation)
        g_c = g_c * np.random.choice([-1,1])

        g_t = np.random.rand(2) - 0.5
        g_t = g_t / np.linalg.norm(g_t)
        g_t *= np.random.uniform(min_translation, max_translation)
        g = get_state_matrix(g_r, g_c, SCALE, g_t[0], g_t[1], None)

        if max_rotation == 0.:
            a_r = 0
        else:
            a_r = np.random.uniform(min_acceleration, max_acceleration)
            a_r = a_r * np.random.choice([-1,1])

        if max_color_rotation == 0.:
            a_c = 0.
        else:
            a_c = np.random.uniform(min_acceleration, max_acceleration)
            a_c = a_c * np.random.choice([-1,1])

        if max_translation == 0.:
            a_t = np.array([0., 0.])
        else:
            a_t = np.random.rand(2) - 0.5
            a_t = a_t / np.linalg.norm(a_t)
            a_t *= np.random.uniform(0., 1.)
        a = get_state_matrix(a_r, a_c, SCALE, a_t[0], a_t[1], None)
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
             # update velocity if small_acceleration
            if k != K:
                if mode == 'small_acceleration':
                    g = a @ g
    return state_matrix, g, a


def main(args):

    replay_buffer = init_episode_dict(args.K)

    N = args.num_timesteps
    for i in tqdm(range(N)):
        state_matrix, g, a = new_state_action_matrix(args.K, args.min_rotation,
        args.max_rotation, args.min_translation, args.max_translation,
        args.min_color_rotation, args.min_color_rotation, args.init_rotation,
        args.init_x, args.init_y, args.init_color, args.object_centric,
        args.mode, args.min_acceleration, args.max_acceleration)
        shape = np.random.choice(eval(args.shape))
        label = get_label_vec(state_matrix, shape)
        data = Tetrominoes.get_data_by_state_matrix(state_matrix, scale=SCALE, shape=shape)
        if args.one_channel:
            assert label[1] == -np.pi
            data = data[...,0:1]

        replay_buffer['label_1'].append(label)
        replay_buffer['state_matrix_1'].append(state_matrix)
        replay_buffer['action_matrix_1'].append(g)
        replay_buffer['obs_1'].append(data.astype(np.float64))

        s = state_matrix
        for k in range(2, args.K+1):
            if args.object_centric:
                s = s @ g
            else:
                s = g @ s
            label = get_label_vec(s, shape)
            replay_buffer[f'label_{k}'].append(label)
            replay_buffer[f'state_matrix_{k}'].append(s)
            data = Tetrominoes.get_data_by_state_matrix(s, scale=SCALE, shape=shape)
            if args.one_channel:
                assert label[1] == -np.pi
                data = data[...,0:1]
            replay_buffer[f'obs_{k}'].append(data.astype(np.float64))
            if k != args.K:
                if args.mode == 'small_acceleration':
                    g = a @ g
                replay_buffer[f'action_matrix_{k}'].append(g)

    save_single_ep_h5py(replay_buffer, args.fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-timesteps', type=int, default=1000,
                        help='Total number of episodes to simulate.')
    parser.add_argument('--K', type=int, default=2,
                        help='Sequence length.')
    parser.add_argument('--fname', type=str, default='test.h5',
                        help='Save path for replay buffer.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--object-centric', action="store_true", help='whether use gx or xg to transform.')
    parser.add_argument('--min-rotation', type=float, default=0.)
    parser.add_argument('--max-rotation', type=float, default=2 * np.pi / 4)
    parser.add_argument('--min-translation', type=float, default=0.)
    parser.add_argument('--max-translation', type=float, default=2.)
    parser.add_argument('--min-color-rotation', type=float, default=0.)
    parser.add_argument('--max-color-rotation', type=float, default=2 * np.pi / 4)
    parser.add_argument('--init-rotation', type=float, default=None, help='if specfied, fix the inital angle of the sequence to this.')
    parser.add_argument('--init-x', type=float, default=None, help='if specfied, fix the inital x-position of the sequence to this.')
    parser.add_argument('--init-y', type=float, default=None, help='if specfied, fix the inital y-position of the sequence to this.')
    parser.add_argument('--init-color', type=float, default=None, help='if specfied, fix the inital color of the sequence to this.')
    parser.add_argument('--one-channel', default=False, action='store_true')
    parser.add_argument('--shape', default='[1]', help='List of shapes to sample per sequence.')
    parser.add_argument('--mode', default='constant_velocity', choices=['constant_velocity', 'small_acceleration'])
    parser.add_argument('--min-acceleration', default=0, type=float)
    parser.add_argument('--max-acceleration', default=2 * np.pi / 60, type=float)
    parsed = parser.parse_args()
    main(parsed)
