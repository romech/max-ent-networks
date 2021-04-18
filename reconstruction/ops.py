import numpy as np


def repeat_col(array, n):
    return array.reshape(-1, 1).repeat(n, axis=1)


def repeat_row(array, n):
    return array.reshape(1, -1).repeat(n, axis=0)
