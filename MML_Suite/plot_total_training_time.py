import json
import numpy as np
import matplotlib.pyplot as plt

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load all metrics
pretrained_multimodal = load_metrics('experiments_output/UTT_FUSION_MOSI_Pretrained_Encoders/metrics/1/epoch_metrics.json')
baseline_multimodal = load_metrics('experiments_output/UTT_FUSION_MOSI_Multimodal_Training/metrics/1/epoch_metrics.json')

# Load encoder pretraining metrics
video_encoder = load_metrics('experiments_output/MOSI_Video_Encoder_Pretrain/metrics/1/train_metrics.json')
audio_encoder = load_metrics('experiments_output/MOSI_Audio_Encoder_Pretrain/metrics/1/train_metrics.json')
text_encoder = load_metrics('experiments_output/MOSI_Text_Encoder_Pretrain/metrics/1/train_metrics.json')

# Calculate total times
def get_total_time(metrics):
    total_time = 0
    for epoch_data in metrics:
        if isinstance(epoch_data, dict) and 'train' in epoch_data:
            total_time += float(epoch_data['train']['timing']['total_time'])
            if 'validation' in epoch_data:
                total_time += float(epoch_data['validation']['timing']['total_time'])
    return total_time

# Get times for each component
video_time = get_total_time(video_encoder)
audio_time = get_total_time(audio_encoder)
text_time = get_total_time(text_encoder)
pretrained_multimodal_time = get_total_time(pretrained_multimodal)
baseline_multimodal_time = get_total_time(baseline_multimodal)

# Total time for pretrained approach
total_pretrained_time = video_time + audio_time + text_time + pretrained_multimodal_time

# Create bar plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars
bar_width = 0.35
baseline_bar = ax.bar(0, baseline_multimodal_time, bar_width, label='Baseline Training', color='red')

# Stacked bar for pretrained approach
bottom = 0
colors = ['lightblue', 'lightgreen', 'lightyellow', 'blue']
labels = ['Video Encoder Pretraining', 'Audio Encoder Pretraining', 
          'Text Encoder Pretraining', 'Multimodal Training']
times = [video_time, audio_time, text_time, pretrained_multimodal_time]

pretrained_bars = []
for time, color, label in zip(times, colors, labels):
    bar = ax.bar(bar_width, time, bar_width, bottom=bottom, color=color, label=label)
    pretrained_bars.append(bar)
    bottom += time

# Customize plot
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Total Training Time Comparison', fontsize=14, pad=20)
ax.set_xticks([0, bar_width])
ax.set_xticklabels(['Baseline Model', 'Pretrained Model'])

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2.,
                f'{height:.1f}s',
                ha='center', va='center', color='black', fontsize=10)

add_labels([baseline_bar])
for bar in pretrained_bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2.,
            f'{height:.1f}s',
            ha='center', va='center', color='black', fontsize=10)

# Add total time labels at the top
ax.text(0, baseline_multimodal_time * 1.05, 
        f'Total: {baseline_multimodal_time:.1f}s\n({baseline_multimodal_time/60:.1f}m)', 
        ha='center', va='bottom')
ax.text(bar_width, total_pretrained_time * 1.05, 
        f'Total: {total_pretrained_time:.1f}s\n({total_pretrained_time/60:.1f}m)', 
        ha='center', va='bottom')

# Adjust legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('total_time_comparison.png', dpi=300, bbox_inches='tight')

# Print detailed statistics
print("\nDetailed time statistics:")
print("\nBaseline Model:")
print(f"Total training time: {baseline_multimodal_time:.2f} seconds ({baseline_multimodal_time/60:.2f} minutes)")

print("\nPretrained Model:")
print(f"Video encoder pretraining: {video_time:.2f} seconds ({video_time/60:.2f} minutes)")
print(f"Audio encoder pretraining: {audio_time:.2f} seconds ({audio_time/60:.2f} minutes)")
print(f"Text encoder pretraining: {text_time:.2f} seconds ({text_time/60:.2f} minutes)")
print(f"Multimodal training: {pretrained_multimodal_time:.2f} seconds ({pretrained_multimodal_time/60:.2f} minutes)")
print(f"Total time: {total_pretrained_time:.2f} seconds ({total_pretrained_time/60:.2f} minutes)")

print(f"\nTime difference: {abs(total_pretrained_time - baseline_multimodal_time):.2f} seconds "
      f"({abs(total_pretrained_time - baseline_multimodal_time)/60:.2f} minutes)")
print(f"Ratio (Pretrained/Baseline): {total_pretrained_time/baseline_multimodal_time:.2f}x") 