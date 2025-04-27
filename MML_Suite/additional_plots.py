import json
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load metrics
pretrained = load_metrics('experiments_output/AVMNIST_ResNet_Pretrained_Training/metrics/1/train_metrics.json')
baseline = load_metrics('experiments_output/AVMNIST_ResNet_Training/metrics/1/train_metrics.json')

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

# 1. Loss and Accuracy
ax1.plot(range(1, len(pretrained) + 1), [d['loss'] for d in pretrained], 'o-', color='blue', 
         label='Loss (With Pretraining)', linewidth=2, markersize=6)
ax1.plot(range(1, len(baseline) + 1), [d['loss'] for d in baseline], 'o-', color='red', 
         label='Loss (Without Pretraining)', linewidth=2, markersize=6)

ax1_2 = ax1.twinx()  # создаем вторую ось Y
ax1_2.plot(range(1, len(pretrained) + 1), [d['accuracy_AI'] for d in pretrained], 's-', color='lightblue',
          label='Accuracy (With Pretraining)', linewidth=2, markersize=6)
ax1_2.plot(range(1, len(baseline) + 1), [d['accuracy_AI'] for d in baseline], 's-', color='lightcoral',
          label='Accuracy (Without Pretraining)', linewidth=2, markersize=6)

ax1.set_title('Loss and Accuracy over Epochs', fontsize=14, pad=10)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12, color='black')
ax1_2.set_ylabel('Accuracy', fontsize=12, color='black')
ax1.grid(True, linestyle='--', alpha=0.7)

# Объединяем легенды с обеих осей
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='center right')

# 2. F1 Scores (macro, micro, weighted)
ax2.plot(range(1, len(pretrained) + 1), [d['f1_macro_AI'] for d in pretrained], 'o-', 
         label='Macro (With Pretraining)', color='blue', linewidth=2, markersize=6)
ax2.plot(range(1, len(pretrained) + 1), [d['f1_micro_AI'] for d in pretrained], 's-', 
         label='Micro (With Pretraining)', color='lightblue', linewidth=2, markersize=6)
ax2.plot(range(1, len(pretrained) + 1), [d['f1_weighted_AI'] for d in pretrained], '^-', 
         label='Weighted (With Pretraining)', color='darkblue', linewidth=2, markersize=6)

ax2.plot(range(1, len(baseline) + 1), [d['f1_macro_AI'] for d in baseline], 'o-', 
         label='Macro (Without Pretraining)', color='red', linewidth=2, markersize=6)
ax2.plot(range(1, len(baseline) + 1), [d['f1_micro_AI'] for d in baseline], 's-', 
         label='Micro (Without Pretraining)', color='lightcoral', linewidth=2, markersize=6)
ax2.plot(range(1, len(baseline) + 1), [d['f1_weighted_AI'] for d in baseline], '^-', 
         label='Weighted (Without Pretraining)', color='darkred', linewidth=2, markersize=6)

ax2.set_title('F1 Scores Comparison', fontsize=14, pad=10)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('F1 Score', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=10)

# 3. Precision
ax3.plot(range(1, len(pretrained) + 1), [d['precision_macro_AI'] for d in pretrained], 'o-', 
         label='Macro (With Pretraining)', color='blue', linewidth=2, markersize=6)
ax3.plot(range(1, len(pretrained) + 1), [d['precision_micro_AI'] for d in pretrained], 's-', 
         label='Micro (With Pretraining)', color='lightblue', linewidth=2, markersize=6)
ax3.plot(range(1, len(pretrained) + 1), [d['precision_weighted_AI'] for d in pretrained], '^-', 
         label='Weighted (With Pretraining)', color='darkblue', linewidth=2, markersize=6)

ax3.plot(range(1, len(baseline) + 1), [d['precision_macro_AI'] for d in baseline], 'o-', 
         label='Macro (Without Pretraining)', color='red', linewidth=2, markersize=6)
ax3.plot(range(1, len(baseline) + 1), [d['precision_micro_AI'] for d in baseline], 's-', 
         label='Micro (Without Pretraining)', color='lightcoral', linewidth=2, markersize=6)
ax3.plot(range(1, len(baseline) + 1), [d['precision_weighted_AI'] for d in baseline], '^-', 
         label='Weighted (Without Pretraining)', color='darkred', linewidth=2, markersize=6)

ax3.set_title('Precision Comparison', fontsize=14, pad=10)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Precision', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend(fontsize=10)

# 4. Recall
ax4.plot(range(1, len(pretrained) + 1), [d['recall_macro_AI'] for d in pretrained], 'o-', 
         label='Macro (With Pretraining)', color='blue', linewidth=2, markersize=6)
ax4.plot(range(1, len(pretrained) + 1), [d['recall_micro_AI'] for d in pretrained], 's-', 
         label='Micro (With Pretraining)', color='lightblue', linewidth=2, markersize=6)
ax4.plot(range(1, len(pretrained) + 1), [d['recall_weighted_AI'] for d in pretrained], '^-', 
         label='Weighted (With Pretraining)', color='darkblue', linewidth=2, markersize=6)

ax4.plot(range(1, len(baseline) + 1), [d['recall_macro_AI'] for d in baseline], 'o-', 
         label='Macro (Without Pretraining)', color='red', linewidth=2, markersize=6)
ax4.plot(range(1, len(baseline) + 1), [d['recall_micro_AI'] for d in baseline], 's-', 
         label='Micro (Without Pretraining)', color='lightcoral', linewidth=2, markersize=6)
ax4.plot(range(1, len(baseline) + 1), [d['recall_weighted_AI'] for d in baseline], '^-', 
         label='Weighted (Without Pretraining)', color='darkred', linewidth=2, markersize=6)

ax4.set_title('Recall Comparison', fontsize=14, pad=10)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Recall', fontsize=12)
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.legend(fontsize=10)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('model_comparison_metrics.png', dpi=300, bbox_inches='tight')
print("Additional metrics plot saved as 'model_comparison_metrics.png'") 