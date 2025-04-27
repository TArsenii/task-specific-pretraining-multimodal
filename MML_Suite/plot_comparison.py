import json
import matplotlib.pyplot as plt
import os

# Set style
plt.style.use('default')

# Load data function
def load_metrics(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load metrics
pretrained = load_metrics('experiments_output/AVMNIST_ResNet_Pretrained_Training/metrics/1/train_metrics.json')
baseline = load_metrics('experiments_output/AVMNIST_ResNet_Training/metrics/1/train_metrics.json')

# Extract data
pretrained_acc = [d['accuracy_AI'] for d in pretrained]
pretrained_loss = [d['loss'] for d in pretrained]
baseline_acc = [d['accuracy_AI'] for d in baseline]
baseline_loss = [d['loss'] for d in baseline]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy plot
ax1.plot(range(1, len(pretrained_acc) + 1), pretrained_acc, 'o-', color='blue', label='With Pretraining', linewidth=2, markersize=6)
ax1.plot(range(1, len(baseline_acc) + 1), baseline_acc, 'o-', color='red', label='Without Pretraining', linewidth=2, markersize=6)
ax1.set_title('Accuracy Comparison', fontsize=12, pad=10)
ax1.set_xlabel('Epoch', fontsize=10)
ax1.set_ylabel('Accuracy', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=10)
ax1.set_ylim(0.5, 1.0)  # Устанавливаем диапазон для accuracy

# Loss plot
ax2.plot(range(1, len(pretrained_loss) + 1), pretrained_loss, 'o-', color='blue', label='With Pretraining', linewidth=2, markersize=6)
ax2.plot(range(1, len(baseline_loss) + 1), baseline_loss, 'o-', color='red', label='Without Pretraining', linewidth=2, markersize=6)
ax2.set_title('Loss Comparison', fontsize=12, pad=10)
ax2.set_xlabel('Epoch', fontsize=10)
ax2.set_ylabel('Loss', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=10)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'model_comparison.png' in the current directory:", end=' ')
print(os.getcwd()) 