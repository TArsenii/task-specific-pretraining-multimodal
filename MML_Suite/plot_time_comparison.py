import json
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load metrics
pretrained = load_metrics('experiments_output/AVMNIST_ResNet_Pretrained_Training/metrics/1/epoch_metrics.json')
baseline = load_metrics('experiments_output/AVMNIST_ResNet_Training/metrics/1/epoch_metrics.json')

# Create figure
plt.figure(figsize=(12, 6))

# Extract time data
pretrained_times = [float(d['train']['timing']['total_time']) + float(d['validation']['timing']['total_time']) for d in pretrained]
baseline_times = [float(d['train']['timing']['total_time']) + float(d['validation']['timing']['total_time']) for d in baseline]

# Calculate cumulative time
pretrained_cumulative = np.cumsum(pretrained_times)
baseline_cumulative = np.cumsum(baseline_times)

# Plot cumulative time
plt.plot(range(1, len(pretrained) + 1), pretrained_cumulative, 'o-', 
         color='blue', label='With Pretraining', linewidth=2, markersize=8)
plt.plot(range(1, len(baseline) + 1), baseline_cumulative, 'o-', 
         color='red', label='Without Pretraining', linewidth=2, markersize=8)

plt.title('Cumulative Training Time Comparison', fontsize=14, pad=10)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Total Time (seconds)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Add time per epoch as text annotations
for i in range(len(pretrained)):
    if i % 2 == 0:  # добавляем аннотации через одну эпоху для читаемости
        plt.annotate(f'{pretrained_times[i]:.1f}s', 
                    (i+1, pretrained_cumulative[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    color='blue',
                    fontsize=8)
        
for i in range(len(baseline)):
    if i % 2 == 0:
        plt.annotate(f'{baseline_times[i]:.1f}s', 
                    (i+1, baseline_cumulative[i]), 
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center',
                    color='red',
                    fontsize=8)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('time_comparison.png', dpi=300, bbox_inches='tight')
print("Time comparison plot saved as 'time_comparison.png'")

# Print statistics
print("\nTraining time statistics:")
print(f"With Pretraining:")
print(f"  Total time: {pretrained_cumulative[-1]:.2f} seconds ({pretrained_cumulative[-1]/60:.2f} minutes)")
print(f"  Average time per epoch: {np.mean(pretrained_times):.2f} seconds")
print(f"  Number of epochs: {len(pretrained)}")
print(f"  Training time per epoch: {np.mean([float(d['train']['timing']['total_time']) for d in pretrained]):.2f} seconds")
print(f"  Validation time per epoch: {np.mean([float(d['validation']['timing']['total_time']) for d in pretrained]):.2f} seconds")

print(f"\nWithout Pretraining:")
print(f"  Total time: {baseline_cumulative[-1]:.2f} seconds ({baseline_cumulative[-1]/60:.2f} minutes)")
print(f"  Average time per epoch: {np.mean(baseline_times):.2f} seconds")
print(f"  Number of epochs: {len(baseline)}")
print(f"  Training time per epoch: {np.mean([float(d['train']['timing']['total_time']) for d in baseline]):.2f} seconds")
print(f"  Validation time per epoch: {np.mean([float(d['validation']['timing']['total_time']) for d in baseline]):.2f} seconds")

print(f"\nTime difference: {abs(pretrained_cumulative[-1] - baseline_cumulative[-1]):.2f} seconds") 