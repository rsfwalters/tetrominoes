from tetrominoes.generate_tetromino_h5_affine import get_state
from tetrominoes.tetrominoes import Tetrominoes
import numpy as np


def batch_renderer(seed, states, one_channel):
    np.random.seed(seed)
    N = states.shape[0]
    images = []
    for n in range(N):
        image = Tetrominoes.get_data_by_label(*get_state(states[n]))
        if one_channel:
            image = image[...,0:1]
        images.append(image)
    return images
