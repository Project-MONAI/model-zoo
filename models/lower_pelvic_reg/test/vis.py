import numpy as np
import matplotlib.pyplot as plt


def vis(moving, fixed):
    axs = plt.figure(constrained_layout=True).subplots(2, 2, sharex=True, sharey=True)
    middle_index = moving["t2w"].shape[-1] // 2
    vis_dict = {
        "moving_t2w": (axs[0, 0], moving["t2w"][0, ..., middle_index]),
        "fixed_t2w": (axs[0, 1], fixed["t2w"][0, ..., middle_index]),
        "moving_seg": (axs[1, 0], moving["seg"][0, ..., middle_index]),
        "fixed_seg": (axs[1, 1], fixed["seg"][0, ..., middle_index]),
    }
    for title, (ax, img) in vis_dict.items():
        ax.set(title=title, aspect=1, xticks=[], yticks=[])
        ax.matshow(np.array(img))
    plt.show()
