# Task-Specific Pre-Training for Multimodal Models

## Investigating the Impact of Task-Specific Pre-Trained Encoders on Late-Fusion Multimodal Model Performance

[![Academic Research](https://img.shields.io/badge/Type-Academic%20Research-blue.svg)](https://github.com/TArsenii/task-specific-pretraining-multimodal)
[![UCD](https://img.shields.io/badge/Institution-University%20College%20Dublin-green.svg)](https://www.ucd.ie/)
[![Final Year Project](https://img.shields.io/badge/Project-Final%20Year%20Project-orange.svg)](https://github.com/TArsenii/task-specific-pretraining-multimodal)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Research Question:** *"Does task-specific pre-training of encoders significantly improve late-fusion multimodal model performance compared to training from scratch?"*

## 🎯 **Project Overview**

This repository presents a comprehensive Final Year Project investigating the impact of task-specific pre-trained encoders on late-fusion multimodal model performance. The research extends the MML_Suite framework to conduct systematic experiments across three diverse datasets, providing insights into optimal multimodal learning strategies.

### **Key Research Contributions:**
- **Original Research Question** - systematic comparison of pre-training vs training from scratch
- **Multi-Dataset Validation** - AVMNIST, MMIMDB, and MOSI datasets
- **Comprehensive Analysis** - 66-page academic report with statistical significance testing
- **Practical Guidelines** - decision framework for when to use pre-trained encoders
- **Framework Extension** - significant contributions to open-source MML_Suite

## 🔬 **Experimental Results**

### **AVMNIST Dataset (Audio-Visual Digit Classification):**
- **11.1% accuracy improvement** with pre-trained encoders (0.9515 vs 0.8567)
- **46.6% loss reduction** in first epoch (0.2706 vs 0.5070)
- **75% faster convergence** - 2 epochs vs 8 epochs to reach 99% accuracy
- **Computational trade-off analysis** - 128.3% higher total cost but 20.1% faster main training

### **MMIMDB Dataset (Movie Genre Classification):**
- **F1_samples improvement:** IT: 0.5632→0.5878, T: 0.4443→0.4967
- **Training time optimization:** 2426s → 1698s main training phase
- **Modality-specific insights:** Text encoders benefit more from pre-training than image encoders
- **Missing modality robustness:** Comprehensive evaluation across different modality combinations

### **MOSI Dataset (Multimodal Sentiment Analysis):**
- **Trimodal implementation:** Audio + Video + Text sentiment analysis
- **Sequential data handling:** Variable-length sequence support
- **Dual-task capability:** Both classification and regression approaches
- **7 modality patterns:** Complete missing modality scenario evaluation

## 🚀 **Technical Implementation**

### **Extended MML_Suite Framework:**
This project significantly extends the original [MML_Suite](https://github.com/jmg049/MML_Suite) by Jack Geraghty with:

- **Novel Dataset Implementations:** MMIMDB and MOSI integration
- **Temporal Modeling:** Sequence-based multimodal learning support
- **Robust Evaluation Pipeline:** Missing modality scenario handling
- **Comprehensive Monitoring:** Training dynamics and convergence analysis
- **Pre-Training Pipeline:** Task-specific encoder initialization

### **Architecture Innovations:**
- **Modality-Specific Encoders:** ResNet18 (audio) + ResNet34 (image) optimization
- **Late-Fusion Strategy:** Systematic comparison of fusion approaches
- **Pre-Training Pipeline:** Individual modality encoder pre-training
- **Statistical Analysis:** Significance testing and confidence intervals

## 📊 **Key Findings**

### **When to Use Pre-Trained Encoders:**
✅ **High-performance requirements** - 11.1% accuracy improvement  
✅ **Shared encoder scenarios** - amortize pre-training cost  
✅ **Fast adaptation needs** - 75% faster convergence  
✅ **Text-heavy multimodal tasks** - stronger benefits for text encoders  

### **When Training from Scratch May Be Better:**
❌ **Severely constrained computational budgets** - 128.3% higher total cost  
❌ **Simple tasks** - diminishing returns from pre-training  
❌ **Highly domain-specific data** - pre-training may not transfer well  

## 🛠 **Getting Started**

### **Installation:**
```bash
# Clone the repository
git clone https://github.com/TArsenii/task-specific-pretraining-multimodal.git
cd task-specific-pretraining-multimodal

# Install dependencies
cd MML_Suite
poetry install
# or
pip install .
```

### **Running Experiments:**

#### **Pre-train Individual Encoders:**
```bash
python train_monomodal.py --config ./configs/avmnist_pretrain.yaml --run_id 1
```

#### **Train Multimodal Model:**
```bash
python train_multimodal.py --config ./configs/avmnist_multimodal.yaml --run_id 1
```

#### **Additional CLI Arguments:**
- `--skip-train` - Skips the training phase
- `--skip-test` - Skips the testing phase
- `--dry-run` - Performs a dry run and stops just before training
- `--disable-monitoring` - Force monitoring off, overrides config file

## ⚙️ **Configuration System**

The project uses a hierarchical YAML-based configuration system designed for reproducible multimodal experiments:

### **Configuration Structure:**
- `experiment`: General experiment settings (device, seed, debug mode)
- `data`: Dataset and dataloader configuration with missing pattern support
- `model`: Model architecture and pre-trained path specifications
- `logging`: Comprehensive logging paths and settings
- `metrics`: Evaluation metrics with batch/epoch level tracking
- `training`: Optimizer, scheduler, and early stopping configuration
- `monitoring`: Gradient, activation, and weight tracking settings

### **Example Configuration:**
```yaml
experiment:
  name: "avmnist_pretrained"
  device: "cuda"
  seed: 42

data:
  datasets:
    train:
      dataset: "avmnist"
      batch_size: 32
      missing_patterns:
        modalities:
          audio:
            missing_rate: 0.2
          image:
            missing_rate: 0.1

model:
  name: "avmnist_late_fusion"
  pretrained_path: "./weights/pretrained_encoders"

training:
  epochs: 50
  optimizer:
    name: "adam"
    lr: 0.001
  early_stopping: true
  early_stopping_patience: 10
```

## 📈 **Research Impact**

### **Theoretical Contributions:**
- **Systematic pre-training analysis** across multiple multimodal domains
- **Convergence rate optimization** through transfer learning
- **Modality-specific pre-training benefits** quantification
- **Computational cost-benefit analysis** for practical deployment

### **Practical Applications:**
- **Social Media Analysis** - text + image content understanding
- **Medical Diagnostics** - patient records + medical imaging
- **Autonomous Systems** - multi-sensor data fusion
- **Sentiment Analysis** - audio + video + text integration

## 🎓 **Academic Context**

- **Institution:** University College Dublin (UCD)
- **Degree:** BSc (Hons) Computer Science
- **Supervisor:** Professor Fatemeh Golpayegani
- **Academic Year:** 2024-2025
- **Report:** [20204701FYP.pdf](./20204701FYP.pdf) - 66-page comprehensive analysis

## 📄 **Repository Structure**

```
├── MML_Suite/                 # Extended multimodal learning framework
│   ├── models/               # Model implementations (AVMNIST, MMIMDB, MOSI)
│   │   ├── avmnist.py       # Audio-visual digit classification
│   │   ├── mmimdb.py        # Movie genre classification
│   │   └── cmams.py         # Cross-modal association models
│   ├── data/                # Dataset handling and preprocessing
│   ├── configs/             # Experiment configurations
│   ├── metrics/             # Evaluation metrics and analysis
│   └── results_processing/  # Analysis and visualization tools
├── 20204701FYP.pdf         # Complete 66-page academic report
├── README.md               # This file
├── LICENSE                 # MIT License
└── pyproject.toml          # Poetry dependency management
```

## 🔧 **Adding Custom Models/Datasets**

### **Models:**
1. Implement required functions:
   - `train_step(self, batch, optimizer, loss_functions, device, metric_recorder, **kwargs)`
   - `validation_step(self, batch, loss_functions, device, metric_recorder, **kwargs)`

2. Add to [resolvers.py](./MML_Suite/config/resolvers.py) in `resolve_model_name` function

### **Datasets:**
1. Implement as standard PyTorch dataset
2. Add to [resolvers.py](./MML_Suite/config/resolvers.py) in `resolve_dataset_name` function

## 📚 **Citation**

If you use this work in your research, please cite:

```bibtex
@thesis{troitskii2025taskspecific,
  title={Investigating the Impact of Task-Specific Pre-Trained Encoders on Late-Fusion Multimodal Model Performance},
  author={Troitskii, Arsenii},
  year={2025},
  school={University College Dublin},
  type={Final Year Project},
  supervisor={Golpayegani, Fatemeh}
}
```

## 🤝 **Acknowledgments**

- **Supervisor:** Professor Fatemeh Golpayegani (UCD School of Computer Science)
- **Base Framework:** [MML_Suite](https://github.com/jmg049/MML_Suite) by Jack Geraghty (jmg049)
- **Institution:** University College Dublin

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note:** This project represents independent research conducted as part of a Final Year Project at University College Dublin. The work extends the MML_Suite framework with novel experimental designs and comprehensive analysis across multiple multimodal learning domains, contributing both theoretical insights and practical guidelines for the multimodal machine learning community.
