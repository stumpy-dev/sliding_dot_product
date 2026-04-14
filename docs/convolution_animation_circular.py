import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# 1. Data Setup
T = [1, 2, 3, 4, 5]

# Periodic extension (2 periods), shifted to start at value 2
periodic_T = (T * 2)[1:]  # [2,3,4,5,1,2,3,4,5]

kernel = [0, 0, 1, 2, 3]

num_frames = 5
start_offset = 0  # periodic_T already starts from value 2

# Pre-calculate results
all_results = []
for f in range(num_frames):
    current_sum = sum(
        kernel[i] * periodic_T[i + f + start_offset]
        for i in range(len(kernel))
    )
    all_results.append(current_sum)

fig, ax = plt.subplots(figsize=(12, 7))

def update(frame):
    ax.clear()
    ax.set_xlim(-1, len(periodic_T) + 1)
    ax.set_ylim(-3.5, 4)
    ax.axis('off')

    # --- Draw Sliding Kernel (Top Array) ---
    for i, val in enumerate(kernel):
        x_pos = i + frame + start_offset
        rect = patches.Rectangle((x_pos, 1.5), 1, 1,
                                 edgecolor='black', facecolor='lightgreen')
        ax.add_patch(rect)
        ax.text(x_pos + 0.5, 2.0, str(val),
                ha='center', va='center', fontsize=14, fontweight='bold')

    # --- Annotation for Kernel (moves with frame) ---
    kernel_start = frame + start_offset
    kernel_end = kernel_start + len(kernel)
    ax.annotate('', xy=(kernel_start, 2.85), xytext=(kernel_end, 2.85),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    ax.text(kernel_start + len(kernel) / 2, 3.15, "Flipped Qr",
            ha='center', color='green', fontsize=12, fontweight='bold')

    # --- Draw Periodic Input ---
    for i, val in enumerate(periodic_T):
        is_active = (frame + start_offset) <= i < (frame + start_offset + len(kernel))

        color = '#82caff' if is_active else 'lightblue'

        rect = patches.Rectangle((i, 0), 1, 1,
                                 edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.5, str(val),
                ha='center', va='center', fontsize=14, fontweight='bold')

    # --- Per-cell correspondence lines (Kernel -> Input) ---
    for i in range(len(kernel)):
        x = frame + start_offset + i + 0.5
        ax.plot([x, x], [1.5, 1.0],
                color='orange', linestyle=':', lw=1.5, alpha=0.8)

    # --- Annotation for Periodicity ---
    ax.annotate('', xy=(0, -0.3), xytext=(len(T), -0.3),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax.text(len(T) / 2, -0.6, "One Period (T)",
            ha='center', color='blue', fontsize=12, fontweight='bold')

    # --- Draw Output ---
    ax.text(-0.2, -1.5, "Output:", va='center',
            ha='right', fontsize=12, fontweight='bold')

    for i in range(num_frames):
        val_text = str(all_results[i]) if i <= frame else ""
        face_color = 'orange' if i == frame else 'white'
        alpha = 0.9 if i == frame else 0.4 if i < frame else 0.1

        rect = patches.Rectangle((i, -2), 1, 1,
                                 edgecolor='black',
                                 facecolor=face_color,
                                 alpha=alpha)
        ax.add_patch(rect)
        ax.text(i + 0.5, -1.5, val_text,
                ha='center', va='center', fontsize=14, fontweight='bold')

    # Connector line
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
    calc_text = f"Step {frame}: " + " + ".join(terms) + f" = {current_sum}"

    ax.text(len(periodic_T)/2, -3.2, calc_text,
            ha='center', fontsize=11, family='monospace',
            bbox=dict(facecolor='white', alpha=0.5))

    plt.title(f"Circular Convolution\n Between T and Qr={kernel[::-1]}", fontsize=16, pad=20)

ani = FuncAnimation(fig, update, frames=num_frames,
                    interval=1500, repeat=True)

ani.save('convolution_annotated_circular.gif', writer='pillow')