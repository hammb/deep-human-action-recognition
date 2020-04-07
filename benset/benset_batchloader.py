import numpy as np
import math
from tensorflow.python.keras.utils import Sequence
# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class BatchLoader(Sequence):

    def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size

    def __len__(self):
            return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) *
            self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) *
            self.batch_size]

            return np.array([]), np.array([])

    