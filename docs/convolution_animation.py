import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# 1. Data Setup
T = [0, 1, 2, 3, 4]
padded_T = [0, 0, 0, 1, 2, 3, 4, 0, 0] # Manually padded for clarity
kernel = [2, 1, 0]
results = [] # To store the convolution sums

fig, ax = plt.subplots(figsize=(12, 6))

def update(frame):
    ax.clear()
    ax.set_xlim(-1, len(padded_T) + 1)
    ax.set_ylim(-2, 4)
    ax.axis('off')

    # Draw Bottom Array (Padded T)
    for i, val in enumerate(padded_T):
        color = 'lightblue' if 2 <= i <= 6 else '#eeeeee'
        rect = patches.Rectangle((i, 0), 1, 1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.5, str(val), ha='center', va='center', fontsize=14, fontweight='bold')

    # Draw Sliding Kernel (Top Array)
    current_sum = 0
    for i, val in enumerate(kernel):
        x_pos = i + frame
        rect = patches.Rectangle((x_pos, 1.5), 1, 1, edgecolor='black', facecolor='lightgreen')
        ax.add_patch(rect)
        ax.text(x_pos + 0.5, 2.0, str(val), ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Calculate multiplication for overlapping cells
        current_sum += val * padded_T[x_pos]

    # Visual alignment line for the '0' cell
    ax.plot([2.5 + frame, 2.5 + frame], [1, 1.5], color='red', linestyle='--', alpha=0.6)

    # Show the Calculation
    calc_text = f"Calculation: "
    terms = [f"({kernel[i]} × {padded_T[i+frame]})" for i in range(len(kernel))]
    calc_text += " + ".join(terms) + f" = {current_sum}"
    
    ax.text(len(padded_T)/2, -0.8, calc_text, ha='center', fontsize=12, family='monospace', 
            bbox=dict(facecolor='white', alpha=0.5))
    
    plt.title(f"1D Convolution Step {frame}", fontsize=16)

# 7 frames allows the kernel to slide across the relevant parts of the array
ani = FuncAnimation(fig, update, frames=7, interval=1500, repeat=True)
_ = ani.save('convolution_demo.gif', writer='pillow')