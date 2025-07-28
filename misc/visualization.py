from torch import Tensor
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import colors
import matplotlib.pyplot as plt


def get_density_img(
    gt_density_map: Tensor, pred_density_map: Tensor, max_val: float = 2
) -> plt.Figure:
    pred_density_map = pred_density_map.cpu()
    gt_density_map = gt_density_map.cpu()

    norm = colors.Normalize(vmin=0, vmax=max_val)

    fig = plt.figure()
    fig.suptitle("Left GT, right Pred", fontsize=16)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 2),
        cbar_location="left",
        cbar_mode="single",
        cbar_size="7%",
        cbar_pad=0.15,
    )

    # add imgs
    ax1, ax2 = grid
    im1 = ax1.imshow(gt_density_map, norm=norm)
    ax2.imshow(pred_density_map, norm=norm)

    ax1.set_axis_off()
    ax2.set_axis_off()

    cbar = ax1.cax.colorbar(
        im1, ticks=[0, 0.5, 1, 1.5, 2]
    )  # TODO check best ticks, maybe dynamic
    ax1.cax.yaxis.set_ticks_position("left")
    ax1.cax.yaxis.set_label_position("left")
    cbar.ax.tick_params(labelsize=10)
    plt.close()

    return fig
