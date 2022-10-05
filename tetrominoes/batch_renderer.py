from tetrominoes.generate_tetromino import SCALE
from tetrominoes.tetrominoes import Tetrominoes
import numpy as np


def batch_renderer(state_matrix, one_channel, shape=0):
    N = state_matrix.shape[0]
    images = []
    for n in range(N):
        image = Tetrominoes.get_data_by_state_matrix(state_matrix[n], scale=SCALE, shape=shape)
        color = np.arctan2(state_matrix[n][4, 3], state_matrix[n][3, 3])
        if one_channel:
            assert color == -np.pi
            image = image[...,0:1]
        images.append(image)
    return images
