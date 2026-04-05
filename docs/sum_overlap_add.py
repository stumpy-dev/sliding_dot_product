import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# --- Configuration ---
arr1 = [3, 8, 14, 20, 26, 14, 5]
arr2 = [18, 33, 44, 50, 56, 29, 10]
overlap = 2
shift = len(arr1) - overlap 

# --- Timing Configuration ---
fps = 15
pause_start_frames = 11      # Frames 0-10: Initial pause
slide_frames = 25            # Frames 11-35: Sliding
pause_mid_frames = 10        # Frames 36-45: Pause before merge
merge_frames = 15            # Frames 46-60: Merging
final_pause_seconds = 3      # <-- ADJUST THIS: Pause duration before loop
final_pause_frames = int(fps * final_pause_seconds)  # 45 frames for 3 seconds

total_frames = 61 + final_pause_frames  # Total animation length

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.set_xlim(-1, len(arr1) + len(arr2) - overlap + 2)
ax.set_ylim(-1.5, 3.5)
ax.axis('off') 

boxes1, texts1 = [], []
boxes2, texts2 = [], []

# Draw Top Array
for i, val in enumerate(arr1):
    rect = patches.Rectangle((i, 1), 1, 1, facecolor='yellow', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    txt = ax.text(i + 0.5, 1.5, str(val), ha='center', va='center', fontsize=16, fontweight='bold')
    boxes1.append(rect)
    texts1.append(txt)

# Label for top array
output1_label = ax.text(-0.7, 1.5, 'Output1', ha='right', va='center', fontsize=13, fontweight='bold')

# Top annotation: "m - 1"
start_x = len(arr1) - overlap
end_x = len(arr1)
center_x = (start_x + end_x) / 2.0
arrow_y = 2.2
ax.annotate('', xy=(start_x, arrow_y), xytext=(end_x, arrow_y),
            arrowprops=dict(arrowstyle='<->', lw=1.5))
ax.text(center_x, arrow_y + 0.2, 'm - 1', ha='center', va='bottom', fontsize=14, fontstyle='italic')

# Draw Bottom Array
for i, val in enumerate(arr2):
    rect = patches.Rectangle((i, 0), 1, 1, facecolor='yellow', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    txt = ax.text(i + 0.5, 0.5, str(val), ha='center', va='center', fontsize=16, fontweight='bold')
    boxes2.append(rect)
    texts2.append(txt)

# Label for bottom array
output2_label = ax.text(-0.7, 0.5, 'Output2', ha='right', va='center', fontsize=13, fontweight='bold')

# Bottom annotation: "slice(m-1, n)" (initially hidden)
slice_start, slice_end = 2, 10
slice_center = (slice_start + slice_end) / 2.0
slice_y = 0.5
slice_arrow = ax.annotate('', xy=(slice_start, slice_y), xytext=(slice_end, slice_y),
                          arrowprops=dict(arrowstyle='<->', lw=2, color='darkgreen'),
                          visible=False)
slice_text = ax.text(slice_center, slice_y - 0.15, 'slice(m-1, n)\nValid Convolution Region', 
                     ha='center', va='top', fontsize=14, fontstyle='italic', 
                     fontweight='bold', color='darkgreen', visible=False)

def animate(frame):
    # Default: hide slice annotation
    slice_arrow.set_visible(False)
    slice_text.set_visible(False)
    
    # Show/hide output labels: visible until merge starts
    if frame < pause_start_frames + slide_frames + pause_mid_frames:
        output1_label.set_visible(True)
        output2_label.set_visible(True)
    else:
        output1_label.set_visible(False)
        output2_label.set_visible(False)
    
    # Frame 0-10: Pause at start
    if frame < pause_start_frames:
        pass  # Static initial state
    
    # Frame 11-35: Slide bottom array
    elif pause_start_frames <= frame < pause_start_frames + slide_frames:
        progress = (frame - pause_start_frames) / slide_frames
        current_shift = progress * shift
        for i in range(len(arr2)):
            boxes2[i].set_x(i + current_shift)
            texts2[i].set_position((i + current_shift + 0.5, 0.5))
            
    # Frame 36-45: Pause before merge
    elif pause_start_frames + slide_frames <= frame < pause_start_frames + slide_frames + pause_mid_frames:
        for i in range(len(arr2)):
            boxes2[i].set_x(i + shift)
            texts2[i].set_position((i + shift + 0.5, 0.5))
            
    # Frame 46-60: Move up and merge
    elif pause_start_frames + slide_frames + pause_mid_frames <= frame < 61:
        progress = (frame - (pause_start_frames + slide_frames + pause_mid_frames)) / merge_frames
        current_y = progress * 1.0 
        for i in range(len(arr2)):
            boxes2[i].set_y(current_y)
            texts2[i].set_position((i + shift + 0.5, current_y + 0.5))
            
        # Merge completion happens at frame 60
        if frame == 60:
            for i in range(len(arr2)):
                if i < overlap:
                    sum_val = arr1[-overlap+i] + arr2[i]
                    texts1[-overlap+i].set_text(str(sum_val))
                    boxes1[-overlap+i].set_facecolor('gold') 
                    boxes2[i].set_alpha(0)
                    texts2[i].set_alpha(0)
            slice_arrow.set_visible(True)
            slice_text.set_visible(True)
    
    # Frame 61+: Extended pause (shows final state until end)
    else:
        # Keep everything in final merged state
        for i in range(len(arr2)):
            boxes2[i].set_x(i + shift)
            boxes2[i].set_y(1 if i >= overlap else 0)  # Merged position or hidden
            texts2[i].set_position((i + shift + 0.5, 1.5 if i >= overlap else 0.5))
        slice_arrow.set_visible(True)
        slice_text.set_visible(True)

    return boxes1 + texts1 + boxes2 + texts2 + [slice_arrow, slice_text, output1_label, output2_label]

# Create animation with extended frames
ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=1000//fps, blit=False)

# Save as GIF (loop is automatic in GIF format)
output_file = 'array_merge_annotated.gif'
ani.save(output_file, writer='pillow', fps=fps)
print(f"Animation saved with {final_pause_seconds}s end pause! Total frames: {total_frames}")