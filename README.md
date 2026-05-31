# Bridging Research-Grade Models and Consumer Wearables: A Personalized Federated Learning Approach to Stress Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

- **Author:** Vasiliki Anna Vaitsopoulou
- **Institution:** City College, University of York Europe Campus
- **Project type:** Individual BSc dissertation project
- **Timeline:** November 2025 – May 2026

This repository contains the code accompanying a BSc dissertation. The project addresses a practical question: whether a stress-detection model trained on research-grade laboratory sensors can be adapted to operate on a consumer wristband suitable for everyday use, without ever centralising participants' physiological data.

The repository is research code. It was developed across three iterations, and several components are driven by configuration files rather than a unified command-line interface. Two experiments were deliberately omitted for scope: these are noted below. Throughout, we have tried to document clearly what the system does and does not do.

## Overview

The work proceeds in three stages, each building on the previous one.

1. **A baseline on WESAD.** WESAD is a laboratory dataset providing chest (RespiBAN) and wrist (Empatica E4) recordings for fifteen subjects. We begin with a classical random-forest pipeline over hand-engineered features, followed by a CNN-LSTM trained on the raw windowed signals. Both are evaluated under a leave-one-subject-out (LOSO) protocol, which establishes the performance ceiling that the later stages aim to preserve.

2. **Federated learning, one client per subject.** Rather than training on pooled data, each of the fifteen subjects is treated as a separate client, and the raw signals never leave the client. We implemented and compared four approaches — FedAvg, FedRep, BABU, and PeFLL, the last of which learns a hypernetwork that generates per-client model weights — against the centralised baseline. The per-subject results are the most informative: several folds that the centralised model handled well (S9, S13, S14) are precisely where personalization provides the clearest benefit.

3. **Distillation to a consumer device.** A Fitbit Charge 6 offers little beyond heart rate: it has neither a chest strap nor the sensing range of a research-grade wristband. The federated model therefore serves as a teacher, and knowledge distillation is used to train a compact student that requires only the features a Fitbit can supply. The student is then evaluated against intraday Charge 6 recordings. This is the bridge referred to in the title: the federated model is the research-grade artefact, and the distilled student is the form that could plausibly run on consumer hardware.

A live demonstration dashboard (FastAPI and React) streams predictions over a WebSocket.

## Project structure

The reconstructed layout is given below: run `git ls-files` for the exact current tree.

```
.
├── data/                          # gitignored — no data is committed
│   ├── raw/                       # WESAD subject pickles (S2…S17)
│   ├── processed/                 # resampled and merged signals, windowed tensors
│   └── fitbit/                    # personal Charge 6 exports (excluded from version control)
├── notebooks/
│   ├── 01_raw_data_eda.ipynb      # the raw WESAD signals
│   ├── 02_merged_data_eda.ipynb   # signals after resampling onto a common clock
│   ├── 03_feature_analysis.ipynb  # feature distributions and leakage checks
│   └── 04_rf_evaluation.ipynb     # classical baseline results
├── src/
│   ├── data/                      # WESAD ingestion and deep-learning preprocessing
│   │   ├── wesad_metadata_extractor.py
│   │   ├── wesad_resample_merge.py
│   │   ├── wesad_dl_preprocess.py
│   │   ├── wesad_dl_windowing.py
│   │   └── wesad_dl_dataset.py
│   ├── ml_pipeline/               # classical random-forest baseline (Iteration 1)
│   │   ├── preprocessing/         # one module per signal: ecg, eda, emg, bvp, temp, acc, resp
│   │   ├── feature_selection.py   # tree, k-best, and PCA ablations
│   │   ├── wesad_windowing.py
│   │   └── ml_classifier.py
│   ├── models/
│   │   ├── wesad_cnnlstm_model.py # the deep baseline
│   │   └── wesad_fl_model.py      # shared body and swappable head used by the FL clients
│   ├── fl/                        # federated learning (Iteration 2)
│   │   ├── clients/               # client_base and client_avg / client_babu / client_rep
│   │   ├── servers/               # server_base and the corresponding aggregators
│   │   └── pefll/                 # PeFLL: dedicated client and server, hypernetwork, embedding net
│   ├── kd/                        # knowledge distillation to Fitbit (Iteration 3)
│   │   ├── teacher_wrapper.py
│   │   ├── kd_teacher_loading.py  # reconstructs a teacher from an FL or centralised checkpoint
│   │   ├── student_model.py
│   │   ├── kd_trainer.py
│   │   ├── kd_loss.py
│   │   ├── paired_data_generator.py
│   │   ├── fitbit_preprocessor.py
│   │   ├── hrv_feature_extractor.py
│   │   ├── kd_config.py
│   │   └── fitbit_evaluator.py
│   ├── training/
│   │   ├── wesad_cnnlstm_train.py
│   │   └── wesad_cnnlstm_evaluate.py
│   └── utils/
│       ├── fl_config.py           # federated hyperparameters
│       ├── metrics.py
│       ├── result_store.py        # the FoldResult format in which results are saved
│       ├── wesad_fl_dataset.py
│       └── utils.py
├── dashboard/                     # live demonstration: FastAPI backend and React/Vite frontend
├── scripts/                       # diagnostics, result loaders, and teacher-verification helpers
├── wesad_chest_configuration.json # signal and feature configuration, chest device
├── wesad_wrist_configuration.json # signal and feature configuration, wrist device
├── demo.ipynb
├── demo_colab_version.ipynb
└── README.md
```

A note on configuration: the project does not use YAML. Federated runs are configured in `src/utils/fl_config.py`, distillation in `src/kd/kd_config.py`, and the WESAD signal and feature settings in the two `*_configuration.json` files. These are the files to edit when changing an experiment.

## Installation

The project requires Python 3.10 or later and, in practice, a CUDA-capable GPU for any deep-learning training. WESAD together with the intermediate tensors requires approximately 10 GB of disk space.

```bash
git clone https://github.com/vvaitsopoulou/stressfl.git
cd stressfl

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

WESAD is not redistributable and must be obtained directly from the [UCI repository](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection); the per-subject folders should be placed in `data/raw/wesad/`. The Fitbit evaluation uses personal Charge 6 exports, which are not included in the repository for privacy reasons; the distillation and evaluation stages expect them under `data/fitbit/`.

## Reproducing the experiments

Most preprocessing and training stages can be run as modules. Only CNN-LSTM training and the classical windowing step expose a full command-line interface. The remaining stages read their settings from the configuration files described above.

**Data preparation** (extract, resample and merge, then window):

```bash
python -m src.data.wesad_metadata_extractor
python -m src.data.wesad_resample_merge
python -m src.data.wesad_dl_preprocess
python -m src.data.wesad_dl_windowing
```

**Iteration 1 — baselines.** The classical random-forest pipeline runs from `src.ml_pipeline` and the JSON configurations. The CNN-LSTM exposes explicit flags:

```bash
python -m src.training.wesad_cnnlstm_train \
    --device-type wrist --window 60 \
    --windowed-dir data/processed/windowed \
    --output-dir results/dl/wrist_baseline \
    --n-classes 2 --batch-size 32 --lr 1e-3 --max-epochs 100 --patience 15

python -m src.training.wesad_cnnlstm_evaluate
```

**Iteration 2 — federated learning.** Each run is specified in `src/utils/fl_config.py` (algorithm, number of rounds, client split). The clients and servers under `src/fl/` are driven from that configuration rather than from command-line arguments.

**Iteration 3 — distillation and Fitbit evaluation.** The paired teacher/student data are generated first, the student is then trained against the loaded teacher checkpoints, and the result is evaluated on the Fitbit data. Distillation settings are defined in `src/kd/kd_config.py`.

**The dashboard** is run separately:

```bash
cd dashboard/backend && uvicorn main:app --reload      # WebSocket and /health endpoints
cd dashboard/frontend && npm install && npm run dev     # React/Vite
```

Where a stage does not expose a command-line entry point, the analysis notebooks and the helper scripts in `scripts/` document how each was invoked end to end.

## Results

Detailed results, figures, and the per-subject analysis are reported in the dissertation rather than here, so that this document need not be revised after each run. In summary, federated personalization recovers most of the centralised LOSO performance while keeping data on-device, and the distilled student performs well enough on Fitbit-derived features to support the feasibility of the proposed bridge.

Two limitations should be stated plainly. The chest deep-learning baseline was not run to completion — only a single fold exists — and was excluded on the grounds that the literature consistently reports a small chest-versus-wrist performance gap and that the wrist is the intended deployment target. In addition, the dashboard currently uses a synthetic data source: the replay and live Fitbit feeds are stubbed rather than implemented.

## Acknowledgments

This work relies on the WESAD dataset:

```bibtex
@inproceedings{wesad2018,
  title={Introducing WESAD, a multimodal dataset for wearable stress and affect detection},
  author={Schmidt, Philip and Reiss, Attila and Duerichen, Robert and Marberger, Claus and Van Laerhoven, Kristof},
  booktitle={Proceedings of the 20th ACM International Conference on Multimodal Interaction},
  pages={400--408},
  year={2018}
}
```

The federated and distillation methods draw on the FedAvg, FedRep, BABU, and PeFLL literature. The PeFLL hypernetwork formulation in particular informed the design of the personalized variant. I thank my supervisor, Miltos Vafiadis, for guidance on the scope of the project.

## License

Released under the MIT License; see [LICENSE](LICENSE). The implementation is free to reuse. The dissertation text itself is submitted separately through the university and is not covered by this license.

## Contact

**Vasiliki Anna Vaitsopoulou**
- vvaitsopoulou@york.citycollege.eu / vaisanna97@gmail.com
- [LinkedIn](https://www.linkedin.com/in/vasiliki-anna-vaitsopoulou-b23bb8303/)
- https://github.com/vvaitsopoulou/stressfl
