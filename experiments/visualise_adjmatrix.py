from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from utils import highlight_first_line


def adjmatrix_figure(labelled_matrices: List[Tuple[str, np.ndarray]],
                     title: str,
                     figsize=(5.5, 4),
                     ncols=3):
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    plt.axis('off')
    ncols = min(ncols, len(labelled_matrices))
    nrows = int(np.ceil(len(labelled_matrices) / ncols))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(nrows, ncols),
                     cbar_mode='single',
                     axes_pad=(0.02, 0.5),
                     label_mode=1)
    for i, ((name, mat), ax) in enumerate(zip(labelled_matrices, grid)):
        im = ax.imshow(mat,
                       cmap='inferno',
                       norm=mpl.colors.Normalize(0, 1, clip=True))
        ax.set_title(highlight_first_line(name),
                     fontsize=8,
                     verticalalignment='top',
                     y=-0.15)
        ax.set_axis_off()
        if i == 0:
            grid.cbar_axes[0].colorbar(im)
