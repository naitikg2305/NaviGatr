from matplotlib import pyplot as plt
import numpy as np
from io import BytesIO

def get_depth_color_map(depth: np.ndarray):
    inverse_depth = 1 / depth

    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    im = ax.imshow(inverse_depth, cmap="turbo", vmin=min_invdepth_vizu, vmax=max_invdepth_vizu)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Depth (m)")

    tick_vals = np.linspace(min_invdepth_vizu, max_invdepth_vizu, 10)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{1/x:.1f}" for x in tick_vals])

    buffer = BytesIO()
    plt.savefig(buffer, format="jpg", dpi=300, bbox_inches="tight")
    buffer.seek(0)

    plt.close(fig)

    return buffer
