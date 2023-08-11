import math
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def jackknife_est_error(model_list, samples):
        '''Returns plotted data from mean standard error of coefficients for each model with error bars'''
        c = np.concatenate([model.layers[0].coefficients for model in model_list], axis=1)

        m = np.mean(c, axis=1)
        se = np.std(c, axis=1) * np.sqrt(samples-1)
        figure, ax = plt.subplots()
        ax.plot(c, color='lightgray', lw=0.5)
        ax.errorbar(np.arange(len(m)), m, se*2)
        ax.axhline(0, ls='--', color='black', lw=0.5)
        return figure