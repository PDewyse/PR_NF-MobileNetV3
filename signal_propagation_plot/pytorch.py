"""
Signal propagation plots (SPP) for PyTorch models.
based on https://github.com/mehdidc/signal_propagation_plot
"""
from functools import partial
import torch
import torch.nn as nn
import numpy as np

def _plot_spp(model, input, metrics, save_dir="plots", plot_name="spp", *args, **kwargs):
    import os
    import matplotlib.pyplot as plt

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, axs = plt.subplots(len(metrics), 1, figsize=(38.4, 21.6))  # 8K resolution in 16:9 aspect ratio
    # go over all the metrics such as squared mean and variance
    for i, m in enumerate(metrics):
        name_values = m(model, input, *args, **kwargs)
        labels, values = zip(*[(name, value) for name, value in name_values])
        depth = np.arange(len(labels))

        axs[i].plot(depth, values, *args, **kwargs)
        axs[i].set_ylabel(m.__name__.replace("_", " "))
        # draw a vertical line for the max value in red and write the value
        max_value = max(values)
        max_index = values.index(max_value)
        axs[i].axvline(x=max_index, color='r', linestyle='--')
        axs[i].text(max_index+0.05, max_value+0.05, f"{max_value:.2f}", color='r')

    plt.xticks(depth, labels, rotation="vertical")
    plt.savefig(save_dir+f"/{plot_name}.png")
    # print(f"--> Saved plot to {save_dir}/{plot_name}.png")
    plt.close()

def plot_spp(model, input, metrics, save_dir="plots", plot_name="spp", *args, **kwargs):
    import os
    import matplotlib.pyplot as plt

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, axs = plt.subplots(len(metrics), 1, figsize=(6, 5))  # 8K resolution in 16:9 aspect ratio
    # go over all the metrics such as squared mean and variance
    colours = ["b", "r"]
    ys = ["Average Channel Squared Mean", "Average Channel Variance"]
    for i, m in enumerate(metrics):
        name_values = m(model, input, *args, **kwargs)
        labels, values = zip(*[(name, value) for name, value in name_values])
        depth = np.arange(len(labels))

        axs[i].plot(depth, values, *args, **kwargs, color=colours[i])
        axs[i].set_ylabel(ys[i])
    # share the labels on the horizontal axis
    
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, tick1On=True)
    plt.tight_layout()
    plt.savefig(save_dir+f"/{plot_name}.png")
    # print(f"--> Saved plot to {save_dir}/{plot_name}.png")
    plt.close()

def get_average_channel_squared_mean_by_depth(model,  *args, **kwargs):
    acts = extract_activations(model, *args, **kwargs)
    values = []
    for name, tensor in acts:
        values.append((name, average_channel_squared_mean(tensor)))
    return values

def get_average_channel_variance_by_depth(model,  *args, **kwargs):
    acts = extract_activations(model, *args, **kwargs)
    values = []
    for name, tensor in acts:
        values.append((name, average_channel_variance(tensor)))
    return values


def average_channel_squared_mean(x):
    if x.ndim == 4:
        return (x.mean(dim=(0,2,3))**2).mean().item()
    elif x.ndim == 2:
        return (x**2).mean().item()
    else:
        raise ValueError(f"not supported shape: {x.shape}")

def average_channel_variance(x):
    if x.ndim == 4:
        return x.var(dim=(0,2,3)).mean().item()
    elif x.ndim == 2:
        return x.var(dim=0).mean().item()
    else:
        raise ValueError(f"not supported shape: {x.shape}")

def extract_activations(model, *args, **kwargs):
    acts = []
    handles = []
    for name, module in model.named_modules():
        handle = module.register_forward_hook(partial(hook, name=name, store=acts))
        handles.append(handle)
    model(*args, **kwargs)
    for handle in handles:
        handle.remove()
    return acts

def hook(self, input, output, store=None, name=None):
    if store is None:
        store = []
    store.append((name, output))

if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt
    model = torchvision.models.resnet101()
    x = torch.randn(64,3,224,224)
    name_values = get_average_channel_squared_mean_by_depth(model, x)
    fig = plt.figure(figsize=(15, 10))
    plot(name_values)
    plt.show()
