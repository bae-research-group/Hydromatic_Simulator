import numpy as np
import os

import matplotlib.ticker
import io

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import tensorflow as tf

from Hydromatic_Simulator.utils.config import config


def save_sequence_plot(pred_sequence, output_dir="./Hydromatic_Simulator/outputs"):
    os.makedirs(output_dir, exist_ok=True)
    pred_sequence = np.squeeze(pred_sequence)
    
    return plot_result(pred_sequence, output_dir)

def save_helper(plt, save_name):
    plt.savefig(save_name, format='png', dpi=500, transparent=False)


def draw_rect_contour(centerline, width=2):
    centerline = np.array(centerline)
    tangents = np.diff(centerline, axis=0)
    normals = np.zeros_like(centerline)

    for i in range(1, len(centerline) - 1):
        v1 = centerline[i] - centerline[i - 1]
        v2 = centerline[i + 1] - centerline[i]
        tangent = (v1 + v2) / 2
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        normals[i] = normal

    normals[0] = np.array([-tangents[0][1], tangents[0][0]]) / np.linalg.norm(tangents[0])
    normals[-1] = np.array([-tangents[-1][1], tangents[-1][0]]) / np.linalg.norm(tangents[-1])

    upper = centerline + width / 2 * normals
    lower = centerline - width / 2 * normals
    contour = np.vstack([upper, lower[::-1], upper[0:1]])
    
    return contour

def plot_result(pred, output_dir):

    image_paths = []

    title_lst = ['0 min'] + config['timesteps']
    for t in range(config['num-timesteps']+1):
        plt.figure(edgecolor='k', layout='constrained',
                   frameon=False, figsize=(5,5), facecolor='lightskyblue')
        plt.rcParams['axes.linewidth'] = 1.5
        if t == 0:
            x_pred = [0.0,] + [4.0625, 8.1250, 12.1875, 16.2500, 20.3125, 24.3750,
                               28.4375, 32.5000, 36.5625, 40.6250, 44.6875, 48.7500,
                               52.8125, 56.8750, 60.9375, 65.0000]
            y_pred = [0.0 for _ in range(config['num-coord'] +1)]
        else:
            x_pred = [0.0] + pred[:,t-1,0].tolist()
            y_pred = [0.0] + pred[:,t-1,1].tolist()

        plt.plot(x_pred, y_pred, c='r', marker='o', linewidth=2, markersize=8, label='Pred')

        contour = draw_rect_contour(list(zip(x_pred, y_pred)), width=3)
        plt.plot(contour[:, 0], contour[:, 1], 'b-', label='Contour', linewidth=2)
        plt.fill(contour[:, 0], contour[:, 1], alpha=0.4, label='Contour Area')

        plt.axhline(0, color='black', linestyle='--', linewidth=2, zorder=0, alpha=0.7)
        plt.axvline(0, color='black', linestyle='--', linewidth=2, zorder=0, alpha=0.7)

        ax = plt.gca()
        ax.set_facecolor('lightskyblue')
        plt.title(title_lst[t], fontsize=23, pad=10, weight='bold')

        
        ax.set_aspect('equal')
        ax.set_xlim(-70, 80)
        ax.set_ylim(-90, 60)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(25))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(25))

        plt.tick_params(which='major', labelsize=15, length=10, axis='both',
                        direction='out', width=1, color='k', pad=3)
        plt.tick_params(which='minor', labelsize=15, length=7, axis='both',
                        direction='out', width=1, color='k', pad=3)

        if t == 0:
            condition = '50°C (as-prepared)'
        elif t == config['num-timesteps']:
            condition = '20°C (equilibrium)'
        else:
            condition = '20°C'
        plt.text(0.03, 0.97, condition,
         transform=plt.gca().transAxes,
         fontsize=17, weight='bold', c='r',
         verticalalignment='top',
         horizontalalignment='left')
        

        filename = os.path.join(output_dir, f"frame_{t}.png")
        save_helper(plt, filename)

        image_paths.append(filename)
        
        plt.close()

    return image_paths
