import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['axes.grid'] = True

# Create directory for saving plots
output_dir = Path('plots/avmnist/comparison')
output_dir.mkdir(parents=True, exist_ok=True)

def load_metrics_data(metrics_dir):
    """
    Load metrics from epoch_metrics.json and test_metrics.json files
    
    Args:
        metrics_dir: Path to the metrics directory (e.g., "experiments_output/AVMNIST_LeNet_Training/metrics/1/")
    """
    epoch_metrics_path = Path(metrics_dir) / "epoch_metrics.json"
    test_metrics_path = Path(metrics_dir) / "test_metrics.json"
    
    if not os.path.exists(epoch_metrics_path):
        print(f"Error: File {epoch_metrics_path} not found")
        return None
    
    try:
        # Load epoch metrics
        with open(epoch_metrics_path, 'r') as f:
            data = json.load(f)
        
        # Dictionaries for storing metrics by epoch
        epochs = []
        train_metrics = {}
        train_times = []
        validation_times = []
        test_metrics = {}

        # Process epoch data
        for item in data:
            if 'epoch' in item:
                # Add epoch number
                epochs.append(item['epoch'])
                
                # Execution time
                if 'train' in item and 'timing' in item['train']:
                    train_times.append(item['train']['timing']['total_time'])
                
                if 'validation' in item and 'timing' in item['validation']:
                    validation_times.append(item['validation']['timing']['total_time'])
                
                # Process training metrics
                if 'train' in item:
                    # Process metrics for AI modality only
                    if 'AI' in item['train']:
                        for metric_name, value in item['train']['AI'].items():
                            metric_key = f"{metric_name}"
                            if metric_key not in train_metrics:
                                train_metrics[metric_key] = []
                            train_metrics[metric_key].append(value)
                    
                    # Process loss metric
                    if 'loss' in item['train']:
                        if 'loss' not in train_metrics:
                            train_metrics['loss'] = []
                        train_metrics['loss'].append(item['train']['loss'])
        
        # Load test metrics if file exists
        if os.path.exists(test_metrics_path):
            with open(test_metrics_path, 'r') as f:
                test_data = json.load(f)
                
            # Process test metrics - take the first item as it contains all metrics
            if test_data and len(test_data) > 0:
                test_item = test_data[0]
                
                # Extract metrics with AI suffix
                for key, value in test_item.items():
                    if key.endswith('_AI') or key == 'loss':
                        # Remove _AI suffix for consistency with training metrics
                        clean_key = key.replace('_AI', '') if key != 'loss' else key
                        test_metrics[clean_key] = value

        return {
            'epochs': epochs,
            'train_metrics': train_metrics,
            'train_times': train_times,
            'validation_times': validation_times,
            'test_metrics': test_metrics
        }
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_metric_comparison(metric_name, data_sets, labels, colors, output_path):
    """
    Create comparison plot for a specific metric across different models
    """
    plt.figure(figsize=(12, 6))
    
    for dataset, label, color in zip(data_sets, labels, colors):
        if not dataset:
            continue
            
        # Check if train metric exists
        if 'train_metrics' in dataset and metric_name in dataset['train_metrics']:
            plt.plot(dataset['epochs'], dataset['train_metrics'][metric_name], 
                     color=color, linestyle='-', marker='o', label=label)
    
    plt.title(f"{metric_name}", fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path / f"{metric_name}.png", dpi=300)
    plt.close()

def plot_test_metrics_comparison(data_sets, labels, colors, output_path):
    """
    Create bar charts for comparing test metrics
    """
    # Get all unique test metrics from all datasets
    all_test_metrics = set()
    for dataset in data_sets:
        if dataset and 'test_metrics' in dataset and dataset['test_metrics']:
            all_test_metrics.update(dataset['test_metrics'].keys())
    
    # Create separate plot for each metric
    for metric_name in all_test_metrics:
        plt.figure(figsize=(10, 6))
        
        # Collect metric values for each model
        metric_values = []
        valid_labels = []
        valid_colors = []
        
        for dataset, label, color in zip(data_sets, labels, colors):
            if (dataset and 'test_metrics' in dataset and 
                dataset['test_metrics'] and 
                metric_name in dataset['test_metrics']):
                metric_values.append(dataset['test_metrics'][metric_name])
                valid_labels.append(label)
                valid_colors.append(color)
        
        if not metric_values:
            continue
            
        # Create bar chart
        x = np.arange(len(valid_labels))
        bars = plt.bar(x, metric_values, color=valid_colors, width=0.5)
        
        # Add values above bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.title(f'Test {metric_name}', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.xticks(x, valid_labels)
        
        plt.tight_layout()
        plt.savefig(output_path / f"test_{metric_name}.png", dpi=300)
        plt.close()

def plot_training_time_comparison(data_sets, labels, colors, output_path):
    """
    Create comparison plot for training time per epoch
    """
    plt.figure(figsize=(12, 6))
    
    for dataset, label, color in zip(data_sets, labels, colors):
        if not dataset or 'train_times' not in dataset:
            continue
            
        plt.plot(dataset['epochs'], dataset['train_times'], 
                 color=color, linestyle='-', marker='o', label=label)
    
    plt.title('Training Time per Epoch Comparison', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path / "training_time_comparison.png", dpi=300)
    plt.close()

def plot_total_time_comparison(data_sets, labels, colors, output_path):
    """
    Create bar chart comparing total training time
    """
    plt.figure(figsize=(12, 6))
    
    # Collect total time data
    total_times = []
    valid_labels = []
    valid_colors = []
    
    for dataset, label, color in zip(data_sets, labels, colors):
        if dataset and 'train_times' in dataset:
            total_time = sum(dataset['train_times'])
            total_times.append(total_time)
            valid_labels.append(label)
            valid_colors.append(color)
    
    if not total_times:
        return
        
    # Create bar chart
    x = np.arange(len(valid_labels))
    bars = plt.bar(x, total_times, color=valid_colors, width=0.5)
    
    # Add values above bars
    for bar, value in zip(bars, total_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
               f'{value:.2f} sec',
               ha='center', va='bottom', fontsize=10)
    
    plt.title('Total Training Time Comparison', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Total Time (seconds)', fontsize=12)
    plt.xticks(x, valid_labels)
    
    plt.tight_layout()
    plt.savefig(output_path / "total_training_time_comparison.png", dpi=300)
    plt.close()

def plot_phase_time_comparison(data_sets, labels, colors, output_path):
    """
    Create grouped bar chart comparing time by phase (training, validation)
    """
    plt.figure(figsize=(12, 8))
    
    # Collect time data for each phase
    train_times = []
    val_times = []
    valid_labels = []
    
    for dataset, label in zip(data_sets, labels):
        if dataset and 'train_times' in dataset and 'validation_times' in dataset:
            train_times.append(sum(dataset['train_times']))
            val_times.append(sum(dataset['validation_times']))
            valid_labels.append(label)
    
    if not train_times:
        return
        
    # Create grouped histogram
    phases = ['Training', 'Validation']
    x = np.arange(len(valid_labels))
    width = 0.35  # bar width
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # First group - training time
    train_bars = ax.bar(x - width/2, train_times, width, label='Training', color='steelblue')
    
    # Second group - validation time
    val_bars = ax.bar(x + width/2, val_times, width, label='Validation', color='seagreen')
    
    # Add value labels above each bar
    for bars, values in zip([train_bars, val_bars], [train_times, val_times]):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{value:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    ax.set_title('Time Comparison by Phase', fontsize=14)
    ax.set_ylabel('Total Time (seconds)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "phase_time_comparison.png", dpi=300)
    plt.close()
        
def main():
    # Paths to metric directories
    experiment_dirs = [
        "experiments_output/AVMNIST_LeNet_Training/metrics/1/",
        "experiments_output/AVMNIST_LeNet_Baseline/metrics/1/"
    ]
    
    # Experiment names for legend
    experiment_labels = [
        "LeNet with pretrained encoders",
        "LeNet baseline"
    ]
    
    # Colors for plots
    experiment_colors = [
        "darkblue",
        "crimson"
    ]
    
    # Load data for all experiments
    datasets = []
    valid_labels = []
    valid_colors = []
    
    for dir_path, label, color in zip(experiment_dirs, experiment_labels, experiment_colors):
        data = load_metrics_data(dir_path)
        if data:
            datasets.append(data)
            valid_labels.append(label)
            valid_colors.append(color)
        else:
            print(f"Skipping experiment {label} due to data loading error")
    
    if not datasets:
        print("Could not load data for any experiment")
        return
    
    # Determine all available metrics
    all_metrics = set()
    for dataset in datasets:
        # Collect all metrics from training data
        if 'train_metrics' in dataset:
            all_metrics.update(dataset['train_metrics'].keys())
    
    # Create plots for each metric
    for metric_name in all_metrics:
        plot_metric_comparison(metric_name, datasets, valid_labels, valid_colors, output_dir)
    
    # Create comparison plots for test metrics
    plot_test_metrics_comparison(datasets, valid_labels, valid_colors, output_dir)
    
    # Create time comparison plots
    plot_training_time_comparison(datasets, valid_labels, valid_colors, output_dir)
    plot_total_time_comparison(datasets, valid_labels, valid_colors, output_dir)
    plot_phase_time_comparison(datasets, valid_labels, valid_colors, output_dir)
    
    print(f"Plots saved to: {output_dir}")

if __name__ == "__main__":
    main() 