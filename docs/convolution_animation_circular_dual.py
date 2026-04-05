import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# ----------- Data सेट -----------
T1 = [1, 2, 3, 4, 5, 0, 0]  # Zero-padded for circular convolution
kernel1 = [0, 0, 0, 0, 1, 2, 3]

T2 = [6, 7, 8, 9, 10, 0, 0]  # Another signal for dual animation
kernel2 = [0, 0, 0, 0, 1, 2, 3]

def prepare(T, kernel):
    periodic_T = T * 2
    num_frames = len(T)
    start_offset = 1

    results = []
    for f in range(num_frames):
        s = sum(
            kernel[i] * periodic_T[i + f + start_offset]
            for i in range(len(kernel))
        )
        results.append(s)

    return periodic_T, results, num_frames, start_offset

periodic_T1, results1, num_frames, start_offset = prepare(T1, kernel1)
periodic_T2, results2, _, _ = prepare(T2, kernel2)

# ----------- Figure (2 rows) -----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
# --- Divider line between subplots ---
divider = plt.Line2D(
    [0.05, 0.95],   # left to right (figure coords)
    [0.525, 0.525],     # vertical position (middle)
    transform=fig.transFigure,
    color='black',
    linewidth=1
)
fig.add_artist(divider)


def draw(ax, T, kernel, periodic_T, all_results, frame, title, period_label, output_label):
    ax.clear()
    ax.set_xlim(-1, len(periodic_T) + 1)
    ax.set_ylim(-3.5, 4)
    ax.axis('off')

    # --- Sliding Kernel ---
    for i, val in enumerate(kernel):
        x_pos = i + frame + start_offset
        rect = patches.Rectangle((x_pos, 1.5), 1, 1,
                                 edgecolor='black', facecolor='lightgreen')
        ax.add_patch(rect)
        ax.text(x_pos + 0.5, 2.0, str(val),
                ha='center', va='center', fontsize=12, fontweight='bold')

    # --- Periodic Input ---
    for i, val in enumerate(periodic_T):
        is_active = (frame + start_offset) <= i < (frame + start_offset + len(kernel))
        color = '#82caff' if is_active else 'lightblue'

        rect = patches.Rectangle((i, 0), 1, 1,
                                 edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.5, str(val),
                ha='center', va='center', fontsize=12, fontweight='bold')

    # --- Period arrow ---
    ax.annotate('', xy=(len(T), -0.3), xytext=(2*len(T), -0.3),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax.text(len(T) + len(T)/2, -0.6, period_label,
            ha='center', color='blue', fontsize=10, fontweight='bold')

    # --- Output ---
    ax.text(-0.2, -1.5, output_label, va='center',
            ha='right', fontsize=10, fontweight='bold')

    for i in range(len(all_results)):
        val_text = str(all_results[i]) if i <= frame else ""
        face_color = 'orange' if i == frame else 'white'
        alpha = 0.9 if i == frame else 0.4 if i < frame else 0.1

        rect = patches.Rectangle((i, -2), 1, 1,
                                 edgecolor='black',
                                 facecolor=face_color,
                                 alpha=alpha)
        ax.add_patch(rect)
        ax.text(i + 0.5, -1.5, val_text,
                ha='center', va='center', fontsize=12, fontweight='bold')

    # Connector
    ax.annotate('', xy=(frame + 0.5, -1),
                xytext=(frame + start_offset + len(kernel)/2, 0),
                arrowprops=dict(arrowstyle="->",
                                color='orange',
                                linestyle='--',
                                lw=1.5,
                                alpha=0.6))

    # Calculation text
    current_sum = all_results[frame]
    terms = [
        f"({kernel[i]}×{periodic_T[i + frame + start_offset]})"
        for i in range(len(kernel))
    ]
    calc_text = f"{' + '.join(terms)} = {current_sum}"

    ax.text(len(periodic_T)/2, -3.2, calc_text,
            ha='center', fontsize=9, family='monospace',
            bbox=dict(facecolor='white', alpha=0.5))

    ax.set_title(title, fontsize=14, pad=0.25)


# ----------- Animation -----------
def update(frame):
    draw(ax1, T1, kernel1, periodic_T1, results1, frame,
         "Circular Convolution 1", "One Period (T1)", "Output1:")

    draw(ax2, T2, kernel2, periodic_T2, results2, frame,
         "Circular Convolution 2", "One Period (T2)", "Output2:")






ani = FuncAnimation(fig, update, frames=num_frames,
                    interval=1500, repeat=True)

plt.tight_layout()
ani.save('dual_convolution.gif', writer='pillow')