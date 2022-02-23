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


def main(args):

    replay_buffer = init_episode_dict()

    # id_tetrominoes = Tetrominoes(num_train_per_shape = args.num_timesteps, num_test_per_shape = args.num_timesteps, lim_scales = [5,5],
    #     lim_xs = [7,64-7], lim_ys = [7,64-7] ,num_val_per_shape= args.num_timesteps, shapes=[0], num_angles=2,
    #     num_scales=1, num_colors=1, num_xs=1, num_ys=1)
    #
    # if args.visualize:
    #     id_tetrominoes.visualize(num_points=10)
    # get_data_by_label(angle=0, color=0, scale=1, x=16, y=16, shape=0, height=64, width=64)
    labels = np.array([360, 0, 7, 32, 32, 0])
    data = Tetrominoes.get_data_by_label(*labels)

    i = 0
    limit = args.num_timesteps
    while i < limit-1:

        # save state matrix
        replay_buffer['state_matrix'].append(labels.astype(np.float64))

        # save action
        replay_buffer['action_matrix'].append(np.array([0.0,0.0,0.0,0.0,0.0,0.0]))  #TODO

        # save obs
        replay_buffer['obs'].append(data.astype(np.float64))# * 255.)

        angle = np.random.rand() * 360
        labels = np.copy(labels)
        labels[0] = angle
        data = Tetrominoes.get_data_by_label(*labels)

        # update state
        #state = np.matmul(action_matrix, state)
        replay_buffer['next_state_matrix'].append(labels.astype(np.float64))
        replay_buffer['next_obs'].append(data.astype(np.float64))# * 255.)
        # import matplotlib.pyplot as plt
        # plt.imshow(replay_buffer['obs'][0])
        # plt.show()
        # plt.imshow(replay_buffer['next_obs'][0])
        # plt.show()

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

    parsed = parser.parse_args()
    main(parsed)
