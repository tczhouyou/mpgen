import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt


def get_colors():
    color_dist = {'black':[0.1, 0.1, 0.1],
                  'heavy gray': [0.4, 0.4, 0.4],
                  'dark gray': [0.7, 0.7, 0.7],
                  'light gray': [0.9, 0.9, 0.9],
                  'dark yellow': [0.984375, 0.7265625, 0],
                  'light yellow': [1, 1, 0.9]}

    return color_dist

def draw_cost(ax, costs):
    ax.plot()
    return

def draw_traj_dist(ax, traj_mu, traj_sig):
    return

def draw_2d_traj_dist_with_vmp(ax, vmp_mu, vmp_sig):
    return
