import numpy as np

def normalize_angle(theta):
    """normalize theta to be in range (-pi, pi]"""
    return ((-theta + np.pi) % (2.0 * np.pi) - np.pi) * -1.0