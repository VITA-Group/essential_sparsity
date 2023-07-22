import random
import numpy as np
import torch
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def print_info(content):
    print("Time: {}\t{}\n".format(datetime.now().strftime("%H:%M:%S"), content)) 

def fprint_info(content, fopen):
    fopen.write("Time: {}\t{}\n".format(datetime.now().strftime("%H:%M:%S"), str(content))) 
    fopen.flush()

def save_dict(file, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)

def load_dict(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def plot_layer_distribution(parameters, save_file_name):
    fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize = (16, 8))
    layer = 0
    for i in range(0, 3):
        for j in range(0, 4):
            layer_params = parameters[layer]
            plot_layer_params = []
            for key in layer_params:
                plot_layer_params += list(layer_params[key])
            axs[i, j].hist(plot_layer_params, bins = 5000)
            axs[i, j].set_title('Layer : {} || Param Count : {} '.format(layer + 1, len(plot_layer_params)))
            axs[i, j].set_ylim([0, 160000])
            axs[i, j].set_xlim([-1, 1])
            layer += 1
    plt.tight_layout()
    plt.savefig(f"./plots/{save_file_name}")