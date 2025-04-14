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
output_dir = Path('plots/mosi/missing_data')
output_dir.mkdir(parents=True, exist_ok=True)

def load_metrics_data(metrics_dir):
    """
    Load metrics from epoch_metrics.json and test_metrics.json files
    
    Args:
        metrics_dir: Path to the metrics directory
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
                
                # Process training metrics - for multimodal (ATV) only
                if 'train' in item and 'ATV' in item['train']:
                    for metric_name, value in item['train']['ATV'].items():
                        metric_key = f"{metric_name}_ATV"
                        if metric_key not in train_metrics:
                            train_metrics[metric_key] = []
                        train_metrics[metric_key].append(value)
                
                # Process single modality metrics (A, T, V)
                for modality in ['A', 'T', 'V']:
                    if 'train' in item and modality in item['train']:
                        for metric_name, value in item['train'][modality].items():
                            metric_key = f"{metric_name}_{modality}"
                            if metric_key not in train_metrics:
                                train_metrics[metric_key] = []
                            train_metrics[metric_key].append(value)
                
                # Process loss metric
                if 'train' in item and 'loss' in item['train']:
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
                
                # Extract all metrics
                for key, value in test_item.items():
                    if key != 'index' and key != 'split':
                        test_metrics[key] = value

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

def plot_metric_comparison(metric_name, modality, data_sets, labels, colors, output_path):
    """
    Create comparison plot for a specific metric across different models
    """
    plt.figure(figsize=(12, 6))
    
    full_metric_name = f"{metric_name}_{modality}"
    
    for dataset, label, color in zip(data_sets, labels, colors):
        if not dataset:
            continue
            
        # Check and plot train metric
        if 'train_metrics' in dataset and full_metric_name in dataset['train_metrics']:
            plt.plot(dataset['epochs'], dataset['train_metrics'][full_metric_name], 
                     color=color, linestyle='-', label=label)
    
    plt.title(f"{metric_name} - {modality}", fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path / f"{metric_name}_{modality}.png", dpi=300)
    plt.close()

def plot_loss_comparison(data_sets, labels, colors, output_path):
    """
    Create comparison plot for loss
    """
    plt.figure(figsize=(12, 6))
    
    for dataset, label, color in zip(data_sets, labels, colors):
        if not dataset:
            continue
            
        # Check and plot train loss
        if 'train_metrics' in dataset and 'loss' in dataset['train_metrics']:
            plt.plot(dataset['epochs'], dataset['train_metrics']['loss'], 
                     color=color, linestyle='-', label=label)
    
    plt.title("Loss", fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path / "loss.png", dpi=300)
    plt.close()

def plot_test_metrics_comparison(data_sets, labels, colors, output_path):
    """
    Create bar charts for comparing test metrics, grouped by modality
    """
    # Group test metrics by modality
    modality_metrics = {'A': [], 'T': [], 'V': [], 'ATV': [], 'other': []}
    
    # Get all unique test metrics from all datasets
    all_test_metrics = set()
    for dataset in data_sets:
        if dataset and 'test_metrics' in dataset and dataset['test_metrics']:
            all_test_metrics.update(dataset['test_metrics'].keys())
    
    # Group metrics by modality
    for metric_name in all_test_metrics:
        if metric_name.endswith('_A'):
            modality_metrics['A'].append(metric_name)
        elif metric_name.endswith('_T'):
            modality_metrics['T'].append(metric_name)
        elif metric_name.endswith('_V'):
            modality_metrics['V'].append(metric_name)
        elif metric_name.endswith('_ATV'):
            modality_metrics['ATV'].append(metric_name)
        else:
            modality_metrics['other'].append(metric_name)
    
    # Create separate plots for each modality
    for modality, metrics in modality_metrics.items():
        if not metrics:
            continue
            
        for metric_name in metrics:
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
            plt.xticks(x, valid_labels, rotation=45, ha='right')
            
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
                 color=color, linestyle='-', label=label)
    
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
    plt.xticks(x, valid_labels, rotation=45, ha='right')
    
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
    ax.set_xticklabels(valid_labels, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "phase_time_comparison.png", dpi=300)
    plt.close()

def plot_modality_efficiency(data_sets, labels, colors, output_path):
    """
    Create bar chart showing performance across different modalities (A, T, V, ATV)
    Uses the mae metric (mean absolute error) for MOSI dataset
    """
    plt.figure(figsize=(12, 8))
    
    # Get modalities and metric name
    modalities = ['A', 'T', 'V', 'ATV']
    metric_name = 'mae'  # For MOSI we use MAE (Mean Absolute Error)
    
    # For each dataset, collect metrics for each modality
    for idx, (dataset, label, color) in enumerate(zip(data_sets, labels, colors)):
        if not dataset or 'test_metrics' not in dataset:
            continue
            
        # Extract metrics for each modality
        values = []
        for modality in modalities:
            metric_key = f"{metric_name}_{modality}"
            if metric_key in dataset['test_metrics']:
                values.append(dataset['test_metrics'][metric_key])
            else:
                values.append(0)  # If modality not available
        
        # Position bars for this dataset
        x = np.arange(len(modalities))
        width = 0.35  # width of the bars
        offset = width * (idx - 0.5 * (len(data_sets) - 1))
        
        bars = plt.bar(x + offset, values, width, label=label, color=color)
        
        # Add values above bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.title(f'Performance Comparison by Modality ({metric_name})', fontsize=14)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(np.arange(len(modalities)), modalities)
    plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / "modality_performance.png", dpi=300)
    plt.close()

def plot_missing_data_comparison(data_sets, labels, colors, output_path):
    """
    Create grouped bar chart comparing performance across different missing data percentages
    Uses mae metric (mean absolute error) for MOSI dataset
    """
    plt.figure(figsize=(14, 8))
    
    # Group experiments by modality
    modality_groups = {
        'Audio': {'20%': None, '90%': None, 'index': 0},
        'Text': {'20%': None, '90%': None, 'index': 1},
        'Video': {'20%': None, '90%': None, 'index': 2}
    }
    
    # Find baseline model index
    baseline_idx = None
    baseline_value = None
    for idx, label in enumerate(labels):
        if label == "Baseline":
            baseline_idx = idx
            break
    
    # Get baseline value
    if baseline_idx is not None and data_sets[baseline_idx] and 'test_metrics' in data_sets[baseline_idx]:
        baseline_value = data_sets[baseline_idx]['test_metrics'].get('mae_ATV', 0)
    
    # Assign models to groups
    for idx, label in enumerate(labels):
        if "Missing Audio" in label:
            percentage = "90%" if "90" in label else "20%"
            modality_groups['Audio'][percentage] = idx
        elif "Missing Text" in label:
            percentage = "90%" if "90" in label else "20%"
            modality_groups['Text'][percentage] = idx
        elif "Missing Video" in label:
            percentage = "90%" if "90" in label else "20%"
            modality_groups['Video'][percentage] = idx
    
    # Collect performance values
    perf_values = []
    x_labels = []
    x_colors = []
    
    for modality, data in modality_groups.items():
        for percentage in ['20%', '90%']:
            idx = data[percentage]
            if idx is not None and data_sets[idx] and 'test_metrics' in data_sets[idx]:
                value = data_sets[idx]['test_metrics'].get('mae_ATV', 0)
                perf_values.append(value)
                x_labels.append(f"{modality} {percentage}")
                x_colors.append(colors[idx])
    
    # Create bar chart
    x = np.arange(len(x_labels))
    bars = plt.bar(x, perf_values, color=x_colors, width=0.6)
    
    # Add baseline line if available
    if baseline_value is not None:
        plt.axhline(y=baseline_value, color='black', linestyle='--', 
                   label=f'Baseline MAE: {baseline_value:.4f}')
    
    # Add values above bars
    for bar, value in zip(bars, perf_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.4f}',
               ha='center', va='bottom', fontsize=10)
    
    plt.title('Performance with Missing Data (MAE)', fontsize=14)
    plt.ylabel('MAE (Lower is better)', fontsize=12)
    plt.xticks(x, x_labels, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "missing_data_comparison.png", dpi=300)
    plt.close()

def main():
    # Paths to metric directories
    experiment_dirs = [
        "experiments_output/UTT_FUSION_MOSI_Multimodal_Training/metrics/1/",
        "experiments_output/UTT_FUSION_MOSI_Pretrained_Encoders_Missing_Audio_20/metrics/1/",
        "experiments_output/UTT_FUSION_MOSI_Pretrained_Encoders_Missing_Audio_90/metrics/1/",
        "experiments_output/UTT_FUSION_MOSI_Pretrained_Encoders_Missing_Text_20/metrics/1/",
        "experiments_output/UTT_FUSION_MOSI_Pretrained_Encoders_Missing_Text_90/metrics/1/",
        "experiments_output/UTT_FUSION_MOSI_Pretrained_Encoders_Missing_Video_20/metrics/1/",
        "experiments_output/UTT_FUSION_MOSI_Pretrained_Encoders_Missing_Video_90/metrics/1/"
    ]
    
    # Experiment names for legend
    experiment_labels = [
        "Baseline",
        "Missing Audio 20%",
        "Missing Audio 90%",
        "Missing Text 20%",
        "Missing Text 90%",
        "Missing Video 20%",
        "Missing Video 90%"
    ]
    
    # Colors for plots
    experiment_colors = [
        "darkblue",
        "lightseagreen",
        "teal",
        "lightcoral",
        "crimson",
        "mediumpurple",
        "darkviolet"
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
    
    # Extract metrics and modalities
    train_metrics = set()
    modalities = set()
    
    for dataset in datasets:
        if 'train_metrics' in dataset:
            for metric_key in dataset['train_metrics'].keys():
                if '_' in metric_key:
                    # Split the metric name and modality
                    parts = metric_key.split('_')
                    modality = parts[-1]
                    metric = '_'.join(parts[:-1])
                    train_metrics.add(metric)
                    modalities.add(modality)
                elif metric_key != 'loss':  # Skip loss as it's handled separately
                    train_metrics.add(metric_key)
    
    # Create plots for loss comparison
    plot_loss_comparison(datasets, valid_labels, valid_colors, output_dir)
    
    # Create plots for each metric and modality combination
    for metric in train_metrics:
        for modality in modalities:
            if modality in ['A', 'T', 'V', 'ATV']:  # Process only valid modalities
                plot_metric_comparison(metric, modality, datasets, valid_labels, valid_colors, output_dir)
    
    # Create comparison plots for test metrics
    plot_test_metrics_comparison(datasets, valid_labels, valid_colors, output_dir)
    
    # Create time comparison plots
    plot_training_time_comparison(datasets, valid_labels, valid_colors, output_dir)
    plot_total_time_comparison(datasets, valid_labels, valid_colors, output_dir)
    plot_phase_time_comparison(datasets, valid_labels, valid_colors, output_dir)
    
    # Create modality efficiency comparison
    plot_modality_efficiency(datasets, valid_labels, valid_colors, output_dir)
    
    # Create missing data comparison
    plot_missing_data_comparison(datasets, valid_labels, valid_colors, output_dir)
    
    print(f"Plots saved to: {output_dir}")

if __name__ == "__main__":
    main() 