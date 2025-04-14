import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_theme()
plt.rcParams['figure.figsize'] = [12, 6]

# Создаем директорию для сохранения результатов
output_dir = Path('C:/projects/final_dev/MML_Suite/MML_Suite/plots/mosi/comparison')
output_dir.mkdir(parents=True, exist_ok=True)

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Словарь для хранения последних значений для каждой эпохи
    epoch_data = {}
    test_time = None
    test_metrics = None
    
    for item in data:
        if 'epoch' in item:
            epoch = item['epoch']
            # Сохраняем только последнее значение для каждой эпохи
            epoch_data[epoch] = {
                'train': {
                    'loss': item['train'].get('loss', None),
                    'timing': item['train'].get('timing', None)
                },
                'validation': {
                    'loss': item['validation'].get('loss', None),
                    'timing': item['validation'].get('timing', None)
                }
            }
        elif 'test' in item:
            test_time = item['test']['timing'].get('total_time', None)
            test_metrics = {
                'loss': item['test'].get('loss', None)
            }
    
    # Сортируем эпохи и создаем списки
    sorted_epochs = sorted(epoch_data.keys())
    train_metrics = {'loss': []}
    train_times = []
    val_times = []
    
    for epoch in sorted_epochs:
        train_value = epoch_data[epoch]['train'].get('loss', None)
        train_metrics['loss'].append(train_value if train_value is not None else np.nan)
        
        train_time = epoch_data[epoch]['train']['timing']['total_time'] if epoch_data[epoch]['train']['timing'] else np.nan
        val_time = epoch_data[epoch]['validation']['timing']['total_time'] if epoch_data[epoch]['validation']['timing'] else np.nan
        train_times.append(train_time)
        val_times.append(val_time)
    
    return sorted_epochs, train_metrics, train_times, val_times, test_time, test_metrics

def plot_loss_comparison(data_sets, labels, colors, output_path):
    plt.figure()
    
    for (epochs, train_metrics, _, _, _, _), label, color in zip(data_sets, labels, colors):
        plt.plot(epochs, train_metrics['loss'], color=color, linestyle='-', label=f'{label}')
    
    plt.title('Comparison of Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(output_path / 'comparison_loss.png')
    plt.close()

def plot_timing_comparison(data_sets, labels, colors, output_path):
    plt.figure()
    
    for (epochs, _, train_times, _, _, _), label, color in zip(data_sets, labels, colors):
        plt.plot(epochs, train_times, color=color, linestyle='-', label=f'{label}')
    
    plt.title('Comparison of Epoch Execution Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(output_path / 'comparison_epoch_times.png')
    plt.close()

def plot_total_time_comparison(data_sets, labels, colors, output_path):
    plt.figure(figsize=(14, 8))
    
    # Собираем данные о времени для каждой фазы
    train_times = []
    val_times = []
    test_times = []
    
    for _, _, train_time_list, val_time_list, test_time, _ in data_sets:
        train_times.append(sum(filter(lambda x: not np.isnan(x), train_time_list)))
        val_times.append(sum(filter(lambda x: not np.isnan(x), val_time_list)))
        test_times.append(test_time if test_time is not None else 0)
    
    # Создаем группированную гистограмму
    phases = ['Train', 'Validation', 'Test']
    x = np.arange(len(phases))
    width = 0.2  # ширина столбца для 2 групп
    
    # Создаем столбцы и добавляем подписи значений
    for i, (label, color) in enumerate(zip(labels, colors)):
        offset = (i - 0.5) * width
        bars = plt.bar(x + offset, [train_times[i], val_times[i], test_times[i]], 
                width, label=label, color=color)
        
        # Добавляем подписи значений над каждым столбцом
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10)
    
    plt.title('Total Time Comparison by Phase')
    plt.xlabel('Phase')
    plt.ylabel('Total Time (seconds)')
    plt.xticks(x, phases)
    plt.legend()
    
    # Увеличиваем верхнюю границу графика, чтобы подписи не обрезались
    plt.ylim(0, max(max(train_times), max(val_times), max(test_times)) * 1.2)
    
    plt.savefig(output_path / 'total_time_comparison.png')
    plt.close()

def plot_test_loss_comparison(data_sets, labels, colors, output_path):
    plt.figure(figsize=(10, 6))
    
    # Собираем тестовые метрики
    test_losses = []
    
    for _, _, _, _, _, test_metrics in data_sets:
        test_loss = test_metrics['loss'] if test_metrics and 'loss' in test_metrics else np.nan
        test_losses.append(test_loss)
    
    # Создаем столбчатую диаграмму
    x = np.arange(len(labels))
    width = 0.5
    
    bars = plt.bar(x, test_losses, width, color=colors)
    
    # Добавляем подписи значений над каждым столбцом
    for bar, loss in zip(bars, test_losses):
        if not np.isnan(loss):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{loss:.4f}',
                    ha='center', va='bottom', fontsize=10)
    
    plt.title('Test Loss Comparison')
    plt.xlabel('Model')
    plt.ylabel('Loss')
    plt.xticks(x, labels)
    plt.grid(axis='y')
    
    plt.savefig(output_path / 'test_loss_comparison.png')
    plt.close()

def main():
    # Загружаем данные для обоих экспериментов
    base_path = Path('experiments_output')
    
    try:
        # Multimodal Training
        multimodal_training_data = load_data(base_path / 'UTT_FUSION_MOSI_Multimodal_Training/metrics/1/epoch_metrics.json')
        print("Успешно загружены данные мультимодального обучения")
    except Exception as e:
        print(f"Ошибка при загрузке данных мультимодального обучения: {e}")
        multimodal_training_data = None
    
    try:
        # Pretrained Encoders
        pretrained_encoders_data = load_data(base_path / 'UTT_FUSION_MOSI_Pretrained_Encoders/metrics/1/epoch_metrics.json')
        print("Успешно загружены данные предобученных энкодеров")
    except Exception as e:
        print(f"Ошибка при загрузке данных предобученных энкодеров: {e}")
        pretrained_encoders_data = None
    
    data_sets = []
    if multimodal_training_data:
        data_sets.append(multimodal_training_data)
    if pretrained_encoders_data:
        data_sets.append(pretrained_encoders_data)
    
    if not data_sets:
        print("Не удалось загрузить данные ни для одной модели. Проверьте пути к файлам.")
        return
    
    labels = ['Multimodal Training', 'Pretrained Encoders'][:len(data_sets)]
    colors = ['blue', 'red'][:len(data_sets)]
    
    # Строим график потерь
    plot_loss_comparison(data_sets, labels, colors, output_dir)
    
    # Строим график времени выполнения по эпохам
    plot_timing_comparison(data_sets, labels, colors, output_dir)
    
    # Строим график общего времени по фазам
    plot_total_time_comparison(data_sets, labels, colors, output_dir)
    
    # Строим график сравнения тестовых потерь
    plot_test_loss_comparison(data_sets, labels, colors, output_dir)
    
    print(f"Графики сохранены в: {output_dir}")

if __name__ == "__main__":
    main() 