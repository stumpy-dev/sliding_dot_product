import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# 1. Data Setup
T = [1, 2, 3, 4, 5]
kernel = [1, 2, 3]

N = len(T)
K = len(kernel)
num_frames = N - K + 1  # valid convolution

# Pre-calculate results (valid convolution)
all_results = []
for f in range(num_frames):
    current_sum = sum(kernel[i] * T[i + f] for i in range(K))
    all_results.append(current_sum)

fig, ax = plt.subplots(figsize=(10, 6))

def update(frame):
    ax.clear()
    ax.set_xlim(-1, N + 1)
    ax.set_ylim(-3.5, 4)
    ax.axis('off')

    # --- Draw Sliding Kernel (Top Array) ---
    for i, val in enumerate(kernel):
        x_pos = i + frame
        rect = patches.Rectangle((x_pos, 1.5), 1, 1,
                                 edgecolor='black', facecolor='lightgreen')
        ax.add_patch(rect)
        ax.text(x_pos + 0.5, 2.0, str(val),
                ha='center', va='center', fontsize=14, fontweight='bold')

    # --- Draw Input Array (Middle) ---
    for i, val in enumerate(T):
        is_active = frame <= i < frame + K

        color = '#82caff' if is_active else 'lightblue'

        rect = patches.Rectangle((i, 0), 1, 1,
                                 edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.5, str(val),
                ha='center', va='center', fontsize=14, fontweight='bold')

    # --- Annotation for T ---
    ax.annotate('', xy=(0, -0.3), xytext=(N, -0.3),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax.text(N / 2, -0.6, "Input Array (T)",
            ha='center', color='blue', fontsize=12, fontweight='bold')

    # --- Draw Output Array (Bottom) ---
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
                xytext=(frame + K/2, 0),
                arrowprops=dict(arrowstyle="->",
                                color='orange',
                                linestyle='--',
                                lw=1.5,
                                alpha=0.6))

    # Calculation Text
    current_sum = all_results[frame]
    terms = [f"({kernel[i]}×{T[i+frame]})" for i in range(K)]
    calc_text = f"Step {frame}: " + " + ".join(terms) + f" = {current_sum}"

    ax.text(N / 2, -3.2, calc_text,
            ha='center', fontsize=11, family='monospace',
            bbox=dict(facecolor='white', alpha=0.5))

    plt.title("Valid Convolution Visualizer", fontsize=16, pad=20)

ani = FuncAnimation(fig, update, frames=num_frames,
                    interval=1500, repeat=True)

ani.save('valid_convolution.gif', writer='pillow')