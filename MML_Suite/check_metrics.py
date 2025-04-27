import json

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load metrics
pretrained = load_metrics('experiments_output/AVMNIST_ResNet_Pretrained_Training/metrics/1/train_metrics.json')
baseline = load_metrics('experiments_output/AVMNIST_ResNet_Training/metrics/1/train_metrics.json')

# Print first epoch data for inspection
print("Keys in pretrained metrics (first epoch):")
print(json.dumps(pretrained[0], indent=2))

print("\nKeys in baseline metrics (first epoch):")
print(json.dumps(baseline[0], indent=2)) 