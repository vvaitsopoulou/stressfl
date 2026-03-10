# CNN-LSTM Training Pipeline

# Full LOSO cross-validation training loop for the multi-branch CNN-LSTM.
# Each fold: train on 13 subjects, validate on 1, test on 1.

# Training Configuration (all values literature-justified):

# Loss:        CrossEntropyLoss with inverse-frequency class weights
#              (Prajod & André, 2022)
# Optimizer:   AdamW (lr=1e-3, weight_decay=1e-4)with
#              decoupled L2 regularisation.
#              Ablation: RMSProp (lr=1e-3, α=0.9) Ordóñez & Roggen (2016)
#              report "best results with widest tolerance to learning rate"
#              for DeepConvLSTM on wearable data.
# Scheduler:   ReduceLROnPlateau (factor=0.5, patience=5)
# Batch size:  64 (compromise: 100 from Ordóñez, 128 from Prajod,
#              smaller batches provide implicit regularisation)
# Early stop:  Patience=10 on validation LOSS (not accuracy)
#              (Moser et al., 2024: "terminated once validation loss
#              stops improving")

# Regularisation Stack :
#   1. High-overlap training / non-overlapping test (in Dataset)
#   2. On-the-fly augmentation: noise + scaling (in Dataset)
#   3. Instance Z-score normalization (in Dataset)
#   4. Dropout p=0.5 after CNN branches and classifier
#      (Ordóñez & Roggen, 2016: "p=0.5, tested p>0.5 with no benefit")
#   5. L2 weight decay via AdamW (weight_decay=1e-4)
#   6. BatchNorm after CNN and LSTM
#   7. Early stopping on val loss
#   8. Forget gate bias = 1 (in model initialization)

# Metrics

#   Primary: Macro-F1 (handles class imbalance)
#   Secondary: Per-class recall (stress recall clinically important)
#   Tertiary: Accuracy (for literature comparability)
#   Diagnostic: Train/val loss curves per fold

# References
#   Ordóñez & Roggen (2016): optimizer, learning rate, dropout, batch size
#   Hwang et al. (2018): Deep ECGNet, batch normalization
#   Prajod & André (2022): weighted loss, LOSO evaluation
#   Moser et al. (2024): early stopping, augmentation, Z-score

import json
import os
import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.wesad_dl_dataset import (
    ALL_SUBJECT_IDS,
    LABEL_MAP_3CLASS,
    LABEL_MAP_BINARY,
    build_loso_datasets,
)
from src.models.wesad_cnnlstm_model import (
    build_chest_model,
    build_wrist_model,
    print_model_summary,
)

#  Custom collate function for dict-based samples


def _collate_fn(batch):
    signal_dicts, labels = zip(*batch)
    signal_names = signal_dicts[0].keys()

    batched_signals = {}
    for name in signal_names:
        batched_signals[name] = torch.stack([d[name] for d in signal_dicts], dim=0)

    batched_labels = torch.stack(labels, dim=0)
    return batched_signals, batched_labels


#  Training configuration dataclass -to be modified for ablation sdtudies


class TrainConfig:

    def __init__(
        self,
        # Architecture
        device_type: str = "wrist",
        n_classes: int = 2,
        lstm_hidden: int = 32,
        lstm_layers: int = 2,
        n_conv_layers: int = 1,
        n_filters_override: int | None = None,
        cnn_dropout: float = 0.5,
        fc_dropout: float = 0.5,
        target_time: int | None = None,  # None → use factory default (chest=5, wrist=8)
        # Training
        optimizer: str = "adamw",  # "adamw" (default) or "rmsprop" (ablation)
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 10,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        # Data
        window_sec: float = 16.0,
        noise_std: float = 0.05,
        scale_range: tuple[float, float] = (0.9, 1.1),
        label_map: dict | None = None,
        signals_to_load: list[str] | None = None,
        # Paths
        windowed_dir: str = "data/dl_windowed/",
        output_dir: str = "results/cnnlstm/",
    ):
        self.device_type = device_type
        self.n_classes = n_classes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.n_conv_layers = n_conv_layers
        self.n_filters_override = n_filters_override
        self.cnn_dropout = cnn_dropout
        self.fc_dropout = fc_dropout
        self.target_time = target_time
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.window_sec = window_sec
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.label_map = label_map or LABEL_MAP_BINARY
        self.signals_to_load = signals_to_load
        self.windowed_dir = windowed_dir
        self.output_dir = output_dir

    def to_dict(self) -> dict:
        # Serialise config for logging.
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


#  Single-fold training


def train_one_fold(
    test_subject: int,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    """Train and evaluate one LOSO fold.

    Returns a dict with fold metrics, loss curves, and best model path.
    """
    fold_start = time.time()

    # Build datasets
    train_ds, val_ds, test_ds = build_loso_datasets(
        test_subject=test_subject,
        windowed_dir=config.windowed_dir,
        device_type=config.device_type,
        window_sec=config.window_sec,
        label_map=config.label_map,
        noise_std=config.noise_std,
        scale_range=config.scale_range,
        signals_to_load=config.signals_to_load,
    )

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        print(f"  Empty dataset in fold S{test_subject} — skipping")
        return {"subject": test_subject, "skipped": True}

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        drop_last=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        drop_last=False,
        num_workers=0,
    )

    # Build model
    build_fn = build_chest_model if config.device_type == "chest" else build_wrist_model
    build_kwargs: dict[str, Any] = {
        "n_classes": config.n_classes,
        "lstm_hidden": config.lstm_hidden,
        "lstm_layers": config.lstm_layers,
        "cnn_dropout": config.cnn_dropout,
        "fc_dropout": config.fc_dropout,
        "n_conv_layers": config.n_conv_layers,
        "n_filters_override": config.n_filters_override,
    }
    # Only override target_time if explicitly set, otherwise use the
    # factory default (chest=5, wrist=8) which is safe for each device.
    if config.target_time is not None:
        build_kwargs["target_time"] = config.target_time
    model = build_fn(**build_kwargs).to(device)

    # Loss function with class weights
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    # Default: AdamW, modern standard with decoupled weight decay.
    # Ablation: RMSProp - Ordóñez & Roggen (2016) report it "consistently
    #   offered the best results with widest tolerance to learning rate"
    #   for DeepConvLSTM on wearable data.
    # NOTE: RMSProp does not have built-in weight decay like AdamW.
    #   L2 regularisation is added via the weight_decay parameter which
    #   PyTorch's RMSProp implements as L2 penalty on the gradient.
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=config.lr,
            alpha=0.9,  # decay factor ρ=0.9 (Ordóñez & Roggen, 2016)
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {config.optimizer}. Use 'adamw' or 'rmsprop'."
        )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    fold_dir = os.path.join(config.output_dir, f"fold_S{test_subject}")
    os.makedirs(fold_dir, exist_ok=True)
    best_model_path = os.path.join(fold_dir, "best_model.pt")

    for epoch in range(config.max_epochs):
        # Train
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0

        for signals, labels in train_loader:
            # Move to device
            signals = {k: v.to(device) for k, v in signals.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = epoch_train_loss / max(n_train_batches, 1)
        train_losses.append(avg_train_loss)

        # Validate
        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for signals, labels in val_loader:
                signals = {k: v.to(device) for k, v in signals.items()}
                labels = labels.to(device)

                logits = model(signals)
                loss = criterion(logits, labels)

                epoch_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = epoch_val_loss / max(n_val_batches, 1)
        val_losses.append(avg_val_loss)

        # lR scheduler step
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best checkpoint
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"    Epoch {epoch:>3d}: train_loss={avg_train_loss:.4f}  "
                f"val_loss={avg_val_loss:.4f}  lr={current_lr:.2e}  "
                f"patience={patience_counter}/{config.patience}"
            )

        if patience_counter >= config.patience:
            print(
                f"    Early stopping at epoch {epoch} "
                f"(best val_loss={best_val_loss:.4f})"
            )
            break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = {k: v.to(device) for k, v in signals.items()}
            logits = model(signals)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    all_classes = sorted(set(config.label_map.values()))

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(
        all_labels, all_preds, average="macro", zero_division=0, labels=all_classes
    )
    f1_weighted = f1_score(
        all_labels, all_preds, average="weighted", zero_division=0, labels=all_classes
    )
    prec = precision_score(
        all_labels, all_preds, average="macro", zero_division=0, labels=all_classes
    )
    rec = recall_score(
        all_labels, all_preds, average="macro", zero_division=0, labels=all_classes
    )
    cm = confusion_matrix(all_labels, all_preds, labels=all_classes)

    fold_time = time.time() - fold_start

    print(f"\n    Results for S{test_subject}:")
    print(f"      Accuracy:      {acc:.4f}")
    print(f"      F1 (macro):    {f1_macro:.4f}")
    print(f"      F1 (weighted): {f1_weighted:.4f}")
    print(f"      Precision:     {prec:.4f}")
    print(f"      Recall:        {rec:.4f}")
    print(f"      Epochs:        {len(train_losses)}")
    print(f"      Time:          {fold_time:.1f}s")
    print(f"      Confusion matrix:\n{cm}")

    # Save loss curves
    curves_path = os.path.join(fold_dir, "loss_curves.json")
    with open(curves_path, "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

    return {
        "subject": test_subject,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm.tolist(),
        "n_epochs": len(train_losses),
        "best_val_loss": best_val_loss,
        "best_model_path": best_model_path,
        "fold_time": fold_time,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }


#  Full LOSO-CV


def run_loso(
    config: TrainConfig,
    subject_ids: list[int] | None = None,
) -> dict:
    """Run full 15-fold LOSO cross-validation.

    Returns aggregate results dict with per-fold and summary metrics.
    """
    if subject_ids is None:
        subject_ids = ALL_SUBJECT_IDS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)

    # Build and print model summary (once, for inspection)
    build_fn = build_chest_model if config.device_type == "chest" else build_wrist_model
    summary_kwargs: dict[str, Any] = {
        "n_classes": config.n_classes,
        "lstm_hidden": config.lstm_hidden,
        "lstm_layers": config.lstm_layers,
        "n_conv_layers": config.n_conv_layers,
        "n_filters_override": config.n_filters_override,
    }
    if config.target_time is not None:
        summary_kwargs["target_time"] = config.target_time
    summary_model = build_fn(**summary_kwargs)
    print_model_summary(summary_model, config.device_type)
    del summary_model

    print("=" * 60)
    print(f"  LOSO Cross-Validation: {len(subject_ids)} folds")
    print("=" * 60)

    start_time = time.time()
    fold_results = []

    # load previously completed folds:
    # On crash and restart, completed folds are loaded from disk and
    # skipped.
    progress_dir = os.path.join(config.output_dir, "_fold_results")
    os.makedirs(progress_dir, exist_ok=True)

    completed_sids = set()
    for test_sid in subject_ids:
        fold_result_path = os.path.join(progress_dir, f"fold_S{test_sid}.json")
        if os.path.exists(fold_result_path):
            with open(fold_result_path) as f:
                cached = json.load(f)
            fold_results.append(cached)
            completed_sids.add(test_sid)

    if completed_sids:
        print(
            f"\n  Resuming: {len(completed_sids)} folds already complete "
            f"({sorted(completed_sids)})"
        )
        print(f"  Remaining: {[s for s in subject_ids if s not in completed_sids]}")

    for test_sid in subject_ids:
        if test_sid in completed_sids:
            print(f"\n  Fold S{test_sid}: already complete — skipping")
            continue

        print("\n")
        print(f"  Fold: test = S{test_sid}")
        print("\n")

        result = train_one_fold(test_sid, config, device)
        fold_results.append(result)

        # Save fold result immediately (crash protection)
        fold_result_path = os.path.join(progress_dir, f"fold_S{test_sid}.json")
        with open(fold_result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"    Fold checkpoint saved → {fold_result_path}")

    # aggregate
    valid_results = [r for r in fold_results if not r.get("skipped", False)]

    if not valid_results:
        print("   No valid folds —> check data paths")
        return {"fold_results": fold_results}

    accs = [r["accuracy"] for r in valid_results]
    f1s_macro = [r["f1_macro"] for r in valid_results]
    f1s_weighted = [r["f1_weighted"] for r in valid_results]
    precs = [r["precision"] for r in valid_results]
    recs = [r["recall"] for r in valid_results]

    total_time = time.time() - start_time

    summary = {
        "n_folds": len(valid_results),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)),
        "f1_macro_mean": float(np.mean(f1s_macro)),
        "f1_macro_std": float(np.std(f1s_macro, ddof=1)),
        "f1_weighted_mean": float(np.mean(f1s_weighted)),
        "f1_weighted_std": float(np.std(f1s_weighted, ddof=1)),
        "precision_mean": float(np.mean(precs)),
        "precision_std": float(np.std(precs, ddof=1)),
        "recall_mean": float(np.mean(recs)),
        "recall_std": float(np.std(recs, ddof=1)),
        "total_time": total_time,
    }

    print("\n")
    print("  LOSO RESULTS SUMMARY")
    print("\n")
    print(
        f"  Accuracy:      {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}"
    )
    print(
        f"  F1 (macro):    {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}"
    )
    print(
        f"  F1 (weighted): {summary['f1_weighted_mean']:.4f} ± {summary['f1_weighted_std']:.4f}"
    )
    print(
        f"  Precision:     {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}"
    )
    print(
        f"  Recall:        {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}"
    )
    print(f"  Total time:    {total_time:.1f}s ({total_time/60:.1f} min)")
    print("  Note: High std motivates Federated Learning (inter-subject variability)")

    # Save results
    results = {
        "summary": summary,
        "fold_results": valid_results,
        "config": config.to_dict(),
    }

    results_path = os.path.join(config.output_dir, "loso_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {results_path}")

    return results


#  CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WESAD CNN-LSTM Training")
    parser.add_argument("--device-type", default="wrist", choices=["chest", "wrist"])
    parser.add_argument("--window", type=float, default=16.0)
    parser.add_argument("--windowed-dir", default="data/dl_windowed/")
    parser.add_argument("--output-dir", default="results/cnnlstm/")
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lstm-hidden", type=int, default=32)
    parser.add_argument("--n-conv-layers", type=int, default=1)
    parser.add_argument(
        "--n-filters",
        type=int,
        default=None,
        help="Override all branch filter counts (for ablation)",
    )
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=["adamw", "rmsprop"],
        help="Optimizer: adamw (default) or rmsprop (ablation)",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use binary classification (stress vs non-stress)",
    )

    args = parser.parse_args()

    config = TrainConfig(
        device_type=args.device_type,
        n_classes=2 if args.binary else args.n_classes,
        lstm_hidden=args.lstm_hidden,
        n_conv_layers=args.n_conv_layers,
        n_filters_override=args.n_filters,
        cnn_dropout=args.dropout,
        fc_dropout=args.dropout,
        optimizer=args.optimizer,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        window_sec=args.window,
        label_map=LABEL_MAP_BINARY if args.binary else LABEL_MAP_3CLASS,
        windowed_dir=args.windowed_dir,
        output_dir=args.output_dir,
    )

    run_loso(config)
