# Bridging Research-Grade Models and Consumer Wearables: A Personalized Federated Learning Approach to Stress Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Author:** Vasiliki Anna Vaitsopoulou  
> **Institution:** City College, University of York Europe Campus  
> **Project Type:** Individual BSc Dissertation Project  
> **Timeline:** November 2025 - May 2026

## 📋 Overview

This project explores how Federated Learning (FL) combined with personalization techniques enables privacy-preserving stress prediction models trained on biometric data from wearables. The work bridges the gap between research-grade physiological datasets and consumer-level wearable devices with limited sensing capabilities.

### Key Objectives

- Train a baseline global model on the WESAD dataset (research-grade sensors)
- Implement a federated learning pipeline treating each subject as a separate client
- Develop personalization techniques for cross-sensor adaptation
- Validate the approach using consumer-grade Fitbit Charge 6 data
- Demonstrate privacy-accuracy tradeoffs in real-world deployment

## 🏗️ Project Structure

```
.
├── data/                          # Data directory (gitignored)
│   ├── raw/                       # Raw WESAD dataset
│   ├── processed/                 # Preprocessed features
│   └── fitbit/                    # Personal Fitbit data
├── src/                           # Source code
│   ├── data/                      # Data loading and preprocessing
│   │   ├── wesad_loader.py
│   │   ├── fitbit_loader.py
│   │   └── preprocessing.py
│   ├── models/                    # Neural Network Architectures
│   │   ├── stress_net.py          # The core architecture  
│   │   └── transfer_heads.py      # Personalization layers (frozen/unfrozen parts)
│   ├── federated/                 # FL System Logic
│   │   ├── client.py              # Virtual client logic
│   │   ├── server.py              # Aggregation logic
│   │   └── strategies.py          # FedAvg, FedProx, etc.
│   ├── training/                  # Training loops
│   │   ├── train_baseline.py
│   │   ├── train_federated.py
│   │   └── train_personalized.py
│   ├── evaluation/                # Evaluation metrics and plots
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── utils/                     # Utility functions
│       ├── config.py
│       └── logging_utils.py
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_experiments.ipynb
│   ├── 03_federated_learning.ipynb
│   └── 04_personalization.ipynb
├── configs/                       # Configuration files
│   ├── baseline_config.yaml
│   ├── federated_config.yaml
│   └── personalization_config.yaml
├── tests/                         # Unit tests
│   ├── test_data_loading.py
│   ├── test_models.py
│   └── test_federated.py
├── scripts/                       # Standalone scripts
│   ├── download_wesad.sh
│   ├── preprocess_data.py
│   └── run_experiment.py
├── demo/                          # Demo application
│   └── streamlit_app.py
├── docs/                          # Documentation
│   ├── literature_review.md
│   ├── methodology.md
│   └── results.md
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project configuration (black, pytest, etc.)
├── .env.example                   # Example environment variables
├── .gitignore
├── .gitattributes
├── .editorconfig
├── .pre-commit-config.yaml
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for training)
- ~10GB disk space for datasets

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vvaitsopoulou/stressfl.git
   cd federated-stress-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

5. **Download the WESAD dataset**
   ```bash
   # Visit: https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection
   # Place the dataset in data/raw/wesad/
   # Or use the download script:
   bash scripts/download_wesad.sh
   ```

6. **Preprocess the data**
   ```bash
   python scripts/preprocess_data.py --config configs/baseline_config.yaml
   ```

### Quick Start

Run the baseline model training:
```bash
python src/training/train_baseline.py --config configs/baseline_config.yaml
```

## 📊 Experiments

### Iteration 1: Baseline Model (Nov 9 - Dec 18, 2025)
- Literature review and dataset exploration
- Train baseline stress detection model on WESAD
- Evaluate on standard metrics (accuracy, F1, ROC-AUC)

**Run:**
```bash
python src/training/train_baseline.py
```

### Iteration 2: Federated Learning Pipeline (Dec 19 - Feb 2, 2026)
- Implement FedAvg algorithm
- Simulate 15 clients (one per WESAD subject)
- Evaluate privacy-accuracy tradeoffs

**Run:**
```bash
python src/training/train_federated.py --n_clients 15 --rounds 100
```

### Iteration 3: Personalization + Demo (Feb 3 - Mar 12, 2026)
- Apply transfer learning for Fitbit Charge 6 adaptation
- Fine-tune on personal data
- Develop Streamlit demo application

**Run:**
```bash
python src/training/train_personalized.py --pretrained checkpoints/federated_model.pth
streamlit run demo/streamlit_app.py
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=src
```

Run specific test categories:
```bash
pytest tests/test_models.py -v
```

## 📈 Results

Results will be documented in `docs/results.md` and include:
- Baseline model performance on WESAD
- Federated learning convergence analysis
- Personalization improvements
- Privacy-accuracy tradeoff analysis
- Cross-sensor performance comparison

## 🔧 Configuration

All experiments are configured via YAML files in `configs/`. Key parameters:

```yaml
# Example: configs/baseline_config.yaml
model:
  architecture: "lstm"
  hidden_size: 128
  num_layers: 2
  dropout: 0.3

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 50
  optimizer: "adam"

data:
  window_size: 60  # seconds
  overlap: 0.5
  features: ["eda", "bvp", "temp", "acc"]
```

## 📝 Documentation

- [Literature Review](docs/literature_review.md)
- [Methodology](docs/methodology.md)
- [Results & Analysis](docs/results.md)
- [API Documentation](docs/api_documentation.md)

## 🤝 Contributing

This is an individual dissertation project, but feedback and suggestions are welcome! If you notice any issues or have suggestions:

1. Open an issue describing the problem/suggestion
2. For code contributions, feel free to fork and submit a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- This project relies on the WESAD dataset:
```bibtex
@inproceedings{wesad2018,
  title={Introducing WESAD, a multimodal dataset for wearable stress and affect detection},
  author={Schmidt, Philip and Reiss, Attila and Duerichen, Robert and Marberger, Claus and Van Laerhoven, Kristof},
  booktitle={Proceedings of the 20th ACM International Conference on Multimodal Interaction},
  pages={400--408},
  year={2018}
}
```
- **Federated Learning:** Inspired by research from Google, OpenMined, and the FL community
- **Supervisor:** [Miltos Vafiadis]

## 📧 Contact

**Vasiliki Anna Vaitsopoulou**
- Email: vvaitsopoulou@york.citycollege.eu / vaisanna97@gmail.com
- LinkedIn: [[vasiliki-anna-vaitsopoulou](https://www.linkedin.com/in/vasiliki-anna-vaitsopoulou-b23bb8303/)]
- Project Link: https://github.com/vvaitsopoulou/stressfl
 

**Note:** This repository contains the implementation code for a Bachelor's dissertation project. The final dissertation document will be submitted separately through university channels.
