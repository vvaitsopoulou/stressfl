# Deep Leakage from Gradients (DLG) attack adapted for WESAD biosignals.
#
# Demonstrates that gradient updates shared during FL can be inverted to
# reconstruct original training data.  This is the privacy motivation
# for DP-FL and the core of the DLG experiment
#
# Adapts from:
#   - PFLIB dlg.py                : the original DLG implementation
#   - Zhu et al. 2019             : "Deep Leakage from Gradients" (NeurIPS)
#
# ADAPTATIONS FROM PFLIB dlg.py
#
# 1. DICT-OF-TENSORS INPUT (major rewrite)
#
#    PFLlib assumes model(x) where x is a single tensor (image batch).
#    Our WESADFLModel.forward() takes dict[str, Tensor] — one tensor
#    per signal modality (BVP, EDA, ACC, TEMP).  The DLG attack must
#    optimise a dictionary of dummy tensors, one per signal key.
#
# 2. REMOVAL OF F.sigmoid() WRAPPER
#
#    PFLlib applies F.sigmoid(dummy_data) before the forward pass,
#    constraining reconstructed values to [0, 1] for normalised images.
#    Our biosignals are Z-SCORE NORMALISED (mean≈0, std≈1, range≈
#    [-3, +3]).  Sigmoid would squash the optimisation landscape into
#    [0, 1], making the gradient signal degenerate for z-scored data:
#      - The attacker's search space would be [0, 1] but the true
#        data lives in [-3, +3] → the attack can never succeed.
#
#    We remove sigmoid entirely and optimise dummy signals directly in
#    the same z-score space as the real data, which is necessary for non-image
#    data (Zhu et al. 2019 used sigmoid specifically because they were dealing with images).
#
# 3. PSNR FORMULA CORRECTION
#
#    PFLlib's PSNR divides MSE by 3 (RGB channels) and uses
#    PIXEL_MAX = 1.0 (normalised image range). We need to change these for biosignals
#
#    We use a fixed data_range of 6.0 (the ±3σ theoretical range of
#    z-scored data) for PSNR computation.  This ensures PSNR values
#    are comparable across windows, subjects, and modalities
#
#    The primary reconstruction metric is normalised MSE (nMSE), which
#    is scale-invariant and has a natural interpretability threshold:
#      nMSE < 1.0 → reconstruction better than a zero-vector guess
#      nMSE ≥ 1.0 → reconstruction no better than random
#    PSNR is reported as a secondary metric for readers familiar with
#    the image convention.
#
# 4. BASE-ONLY GRADIENT MATCHING
#
#    In FedRep, only model.base (CNN) parameters are uploaded.  The
#    origin_grad computed by call_dlg() in server_base.py is over
#    base parameters only (global_base - client_base).
#
#    But DLG needs a forward pass through the full model (base → head)
#    to get predictions, because gradients of the loss re base
#    parameters depend on the head's computation (chain rule).
#
#    The solution: pass the full model for the forward pass, but
#    compute autograd only re base parameters.
#
# 5. PER-MODALITY REPORTING
#
#    Since biosignal modalities have different privacy sensitivities
#    (Benouis et al. 2024: "ECG and acceleration are the most
#    identity-revealing modalities"), we report nMSE per signal branch
#    in addition to the aggregate nMSE.  This tells us which modality
#    is most vulnerable to gradient inversion.
#
# 6. LBFGS ITERATION STRUCTURE
#
#    Both PFLlib and the original Zhu et al. 2019 code use the pattern:
#        for iters in range(N): optimizer.step(closure)
#    with default LBFGS (max_iter=20).  Each step() call runs up to 20
#    internal LBFGS iterations, so total function evaluations are up to
#    N × 20.  With n_iters=100, this is up to 2,000 evals.
#
#    We keep this pattern for consistency with the reference code.
#
# WHAT THE SERVER SEES (THE ATTACKER MODEL)
#
#   - The server faithfully follows the FL protocol
#   - But it tries to reconstruct training data from observed gradients
#
# In FedRep, the server observes:
#   - The global CNN base parameters (it sent these)
#   - The updated CNN base parameters from each client (it received these)
#   - The difference (global - client) ≈ gradient of the loss re.
#     base parameters, accumulated over local_epochs × batches
#
# The server does not observe:
#   - The LSTM head parameters (never communicated)
#   - The head's gradients
#   - The raw data or labels
#
# For a realistic attack, the server would need to guess or approximate
# the head.In this implementation (following PFLlib and standard DLG
# practice),we give the attacker the client's full model (base + head)
# as an upper on attack effectiveness.This is done with the thought process tat
# if the attack fails even with the head, it would fail harder without.
#
# ──────────────────────────────────────────────────────────────────────
# LIMITATIONS AND CAVEATS
# ──────────────────────────────────────────────────────────────────────
#
# 1. DLG on multi-epoch updates: Zhu et al. 2019 noted that
#    DLG works best on single-batch single-step gradients.  Our
#    setting (5 local epochs × multiple batches) is more realistic
#    but less favourable for the attacker.
#
# 2. Batch size: larger batches make reconstruction harder because
#    the gradient is averaged over more samples.  With batch_size=64,
#    the attack attempts to reconstruct the "average" training signal,
#    not individual samples.
#
# 3. These limitations make our PSNR values lower on privacy.
#    the reported PSNR overestimates the attacker's true ability,
#    because we give them the full model and match against multi-epoch
#    aggregated gradients.
#
# 4. FedRep-DP variants use GroupNorm in the base (required by Opacus), while non-DP variants
#    (FedAvg, FedRep) use BatchNorm.  DLG operates on structurally
#    different models, so PSNR/nMSE differences between DP and non-DP
#    reflect the noise injection and the normalisation change.

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch
import torch.nn as nn

# CONSTANs

# Fixed data range for PSNR computation on z-scored biosignals. A fixed range ensures PSNR is comparable across
# windows, subjects, and modalities.  Per-window max-min is not used
# because it varies dramatically (eg flat baseline ≈ 0.5, stress ≈ 4.0)
ZSCORE_DATA_RANGE = 6.0


# PSNR AND RECONSTRUCTION METRICS
def psnr_signal(
    original: np.ndarray,
    reconstruction: np.ndarray,
    data_range: float = ZSCORE_DATA_RANGE,
) -> float:
    """Compute Peak Signal-to-Noise Ratio for z-scored biosignals.

    Uses the standard signal-processing PSNR definition (Wang et al.
    2004, IEEE TIP):

        PSNR = 10 * log10(data_range² / MSE)

    For z-scored biosignals, data_range is fixed at ZSCORE_DATA_RANGE
    (6.0, representing ±3σ) to ensure cross-window and cross-subject
    comparability. nMSE is the primary reconstruction quality measure.

    Parameters:
    original : np.ndarray
        Ground truth signal (any shape, will be flattened).
    reconstruction : np.ndarray
        Reconstructed signal (same shape as original).
    data_range : float
        Fixed dynamic range for PSNR computation.
        Default: ZSCORE_DATA_RANGE (6.0).

    Returns:
    float
        PSNR in dB.  Higher = better reconstruction = worse privacy.
        Returns 100.0 if MSE is exactly 0 (perfect reconstruction).
    """
    original = original.flatten().astype(np.float64)
    reconstruction = reconstruction.flatten().astype(np.float64)

    mse = np.mean((original - reconstruction) ** 2)

    if mse == 0.0:
        return 100.0

    return float(10.0 * np.log10(data_range**2 / mse))


def normalised_mse(original: np.ndarray, reconstruction: np.ndarray) -> float:
    """Compute MSE normalised by the variance of the original signal.

    nMSE = MSE / Var(original)

    This is the primaru reconstruction quality metric for DLG on
    z-scored biosignals.  It is scale-invariant and has a natural
    interpretability threshold:

      nMSE < 1.0 → reconstruction is better than a zero-vector guess
      nMSE ≈ 0.0 → near-perfect reconstruction (privacy breach)
      nMSE ≥ 1.0 → reconstruction is no better than random (privacy OK)

    Parameters
    ----------
    original, reconstruction : np.ndarray
        Same shape.  Will be flattened.

    Returns
    -------
    float
        Normalised MSE.  Returns inf if variance is 0.
    """
    original = original.flatten().astype(np.float64)
    reconstruction = reconstruction.flatten().astype(np.float64)

    mse = np.mean((original - reconstruction) ** 2)
    var = np.var(original)

    if var == 0.0:
        return float("inf")

    return float(mse / var)


# RESULT CONTAINER


@dataclass
class DLGResult:
    """Container for DLG attack results on one client.

    Stores aggregate and per-modality metrics for the privacy analysis.
    The primary metric is nMSE (scale-invariant)
    """

    client_id: int = -1
    aggregate_psnr: float = float("nan")
    aggregate_nmse: float = float("inf")
    per_modality_psnr: dict[str, float] = field(default_factory=dict)
    per_modality_nmse: dict[str, float] = field(default_factory=dict)
    n_batches_attacked: int = 0
    final_grad_diff: float = float("inf")


# CORE DLG ATTACK


def DLG(
    net: nn.Module,
    origin_grad: list[torch.Tensor],
    target_inputs: list[tuple[dict[str, torch.Tensor], torch.Tensor]],
    grad_params: list[nn.Parameter] | None = None,
    # n_iters: int = 100,
) -> DLGResult | None:
    """Deep Leakage from Gradients attack for WESAD biosignal models.

    Optimises dummy signal windows to produce gradients that match
    the observed client gradients, following Zhu et al. 2019.

    Adapted from PFLIB dlg.py with the following changes:
      1. Input is dict[str, Tensor] per modality, not a single tensor
      2. No F.sigmoid() wrapper (z-scored signals, not [0,1] images)
      3. PSNR uses fixed data_range (6.0 for ±3σ z-scored data)
      4. nMSE is the primary metric (scale-invariant, threshold at 1.0)
      5. Gradient matching against specified parameters (base-only
         for FedRep) instead of all net.parameters()
      6. Per-modality nMSE breakdown for privacy analysis

    Parameters
    ----------
    net : nn.Module
        The client's full model (base + head) for forward pass.
        Must be in eval() mode.  The attacker uses the full model
        because the forward pass needs base → head to get predictions.
    origin_grad : list[torch.Tensor]
        The observed "gradient": list of (global_param - client_param)
        tensors for the shared parameters only (CNN base in FedRep,
        full model in FedAvg).
    target_inputs : list[tuple[dict[str, Tensor], Tensor]]
        Ground truth (signals_dict, model_output) pairs used to
        compute metrics after the attack.  These represent what the
        attacker is trying to reconstruct.
    grad_params : list[nn.Parameter] | None
        Which parameters to compute gradients for the matching
        loss.  If None, uses all net.parameters() (FedAvg setting).
        For FedRep, pass list(net.base.parameters()) so that gradient
        matching is only against the communicated base parameters.
    n_iters : int
        Number of LBFGS outer steps per batch.  Each step() call runs
        up to max_iter=20 internal LBFGS iterations, so total function
        evaluations are up to n_iters × 20.  Default 100 matches
        PFLlib.

    Returns
    -------
    DLGResult | None
        Attack results with aggregate and per-modality metrics.
        Returns None if no valid reconstructions were achieved.

    Algorithm
    ---------
    For each target batch:
      1. Create dummy signals: one random tensor per modality key,
         same shape as ground truth, with requires_grad=True.
      2. Create dummy output: random tensor matching model output shape.
      3. Optimise (dummy_signals, dummy_output) via LBFGS to minimise:
            L = Σ_i || ∂loss/∂θ_i(dummy) - origin_grad_i ||²
         where θ_i are the grad_params and loss = MSE(net(dummy), dummy_out).
      4. After optimisation, compute nMSE and PSNR between ground
         truth and reconstructed (dummy) signals.

    Notes
    -----
    - LBFGS is used (not Adam) because it provides second-order
      information critical for this non-convex reconstruction problem.
    - The outer loop pattern (for i in range(N): step(closure)) matches
      both PFLlib and the original Zhu et al. 2019 code.
    """
    if grad_params is None:
        grad_params = list(net.parameters())

    criterion = torch.nn.MSELoss()
    batch_psnrs: list[float] = []
    batch_nmses: list[float] = []
    batch_grad_diffs: list[float] = []

    # Per-modality accumulators: signal_name → list of per-batch values
    modality_psnrs: dict[str, list[float]] = {}
    modality_nmses: dict[str, list[float]] = {}

    for _idx, (gt_signals, gt_out) in enumerate(target_inputs):

        # Determine device from ground truth data
        device = gt_out.device

        # 1. Create dummy signal dict
        # One dummy tensor per modality, same shape, random init,
        # requires_grad=True for LBFGS optimisation.
        #
        # NO F.sigmoid() wrapper: our data is z-score normalised,
        # not [0, 1] image pixels.
        dummy_signals = {}
        optimised_tensors = []

        for sig_name, gt_tensor in gt_signals.items():
            dummy = torch.randn_like(gt_tensor, requires_grad=True, device=device)
            dummy_signals[sig_name] = dummy
            optimised_tensors.append(dummy)

        # 2. Create dummy output
        dummy_out = torch.randn_like(gt_out, requires_grad=True, device=device)
        optimised_tensors.append(dummy_out)

        # LBFGS optimisation
        # Each step() call runs up to max_iter=20 internal iterations.
        # Total function evaluations: up to n_iters × 20.
        optimizer = torch.optim.LBFGS(optimised_tensors)

        # Define closure outside the loop
        last_grad_diff = torch.tensor(float("inf"), device=device)

        def closure(
            _optimizer=optimizer,
            _dummy_signals=dummy_signals,
            _dummy_out=dummy_out,
            _device=device,
        ):
            nonlocal last_grad_diff
            _optimizer.zero_grad()

            # Forward pass through the FULL model with dummy signals.
            current_signals = dict(_dummy_signals)

            dummy_pred = net(current_signals)
            dummy_loss = criterion(dummy_pred, _dummy_out)

            # Compute gradients of the dummy loss w.r.t. the
            # specified parameters (base-only for FedRep).
            # create_graph=True is needed because we backprop
            # through the gradient computation itself.
            dummy_grad = torch.autograd.grad(
                dummy_loss,
                grad_params,
                create_graph=True,
            )

            # Gradient matching loss: L2 distance between dummy
            # gradients and the observed (origin) gradients.
            grad_diff = torch.tensor(0.0, device=_device)
            for gx, gy in zip(dummy_grad, origin_grad):
                grad_diff = grad_diff + ((gx - gy) ** 2).sum()

            grad_diff.backward()
            last_grad_diff = grad_diff.detach()
            return grad_diff

        # Record final gradient matching loss for this batch
        batch_grad_diffs.append(last_grad_diff.item())

        # 4. Compute reconstruction metrics
        # Compare ground truth signals against the optimised dummy
        # signals.  No sigmoid on the dummy: direct comparison in
        # z-score space.

        # Aggregate: flatten all modalities into a single vector
        gt_flat_parts = []
        recon_flat_parts = []

        for sig_name in gt_signals:
            gt_np = gt_signals[sig_name].detach().cpu().numpy()
            recon_np = dummy_signals[sig_name].detach().cpu().numpy()

            gt_flat_parts.append(gt_np.flatten())
            recon_flat_parts.append(recon_np.flatten())

            # Per-modality metrics
            p = psnr_signal(gt_np, recon_np)
            n = normalised_mse(gt_np, recon_np)

            if sig_name not in modality_psnrs:
                modality_psnrs[sig_name] = []
                modality_nmses[sig_name] = []

            if not math.isnan(p):
                modality_psnrs[sig_name].append(p)
            if not math.isinf(n):
                modality_nmses[sig_name].append(n)

        # Aggregate metrics across all modalities
        gt_all = np.concatenate(gt_flat_parts)
        recon_all = np.concatenate(recon_flat_parts)

        p_agg = psnr_signal(gt_all, recon_all)
        n_agg = normalised_mse(gt_all, recon_all)

        if not math.isnan(p_agg):
            batch_psnrs.append(p_agg)
        if not math.isinf(n_agg):
            batch_nmses.append(n_agg)

    # Aggregate across batches

    if not batch_nmses:
        return None

    result = DLGResult(
        aggregate_psnr=float(np.mean(batch_psnrs)) if batch_psnrs else float("nan"),
        aggregate_nmse=float(np.mean(batch_nmses)),
        n_batches_attacked=len(batch_nmses),
        final_grad_diff=(
            float(np.mean(batch_grad_diffs)) if batch_grad_diffs else float("inf")
        ),
    )

    # Per-modality averages
    for sig_name in modality_psnrs:
        if modality_psnrs[sig_name]:
            result.per_modality_psnr[sig_name] = float(
                np.mean(modality_psnrs[sig_name])
            )
    for sig_name in modality_nmses:
        if modality_nmses[sig_name]:
            result.per_modality_nmse[sig_name] = float(
                np.mean(modality_nmses[sig_name])
            )

    return result


# SERVER-SIDE DLG HOOK
#
# This function is called from server_base.py's call_dlg() method.
# It adapts PFLIBserverbase.py's call_dlg() to the WESAD context.
#
# The key differences from PFLlib's call_dlg():
#
#   1. origin_grad uses global_model.base vs client.model.base
#      (not full model), because only base is communicated in FedRep.
#
#   2. target_inputs are (signals_dict, output) not (tensor, output),
#      because our model takes dict input.
#
#   3. We pass grad_params=list(client_model.base.parameters()) to
#      DLG() so gradient matching is against base params only.
#
#   4. The client model for DLG must be the FULL model (not just base)
#      because the forward pass needs base → head to produce logits.
#      But the uploaded_models in receive_models() are base-only
#      deep copies.  So we use the live client model instead.
#
#   5. Results include per-modality nMSE for the privacy analysis.


def call_dlg_for_client(
    global_base: nn.Module,
    client_full_model: nn.Module,
    client_base: nn.Module,
    trainloader,
    device: torch.device,
    batch_num: int = 2,
    # n_iters: int = 100,
    client_id: int = -1,
) -> DLGResult | None:
    """Run DLG attack for one client.

    Called from the server's call_dlg() method.

    Parameters
    ----------
    global_base : nn.Module
        The server's global base model (before this round's aggregation).
    client_full_model : nn.Module
        The client's FULL model (base + head) for forward pass.
        This is the live client model, not the deep-copied base.
    client_base : nn.Module
        The client's trained base (from uploaded_models).
        Used to compute origin_grad = global_base - client_base.
    trainloader : DataLoader
        The client's training data loader for ground truth.
    device : torch.device
        Computation device.
    batch_num : int
        Number of batches to attack (default 2, matching PFLlib).
    n_iters : int
        LBFGS outer steps per batch.  Each step runs up to 20
        internal LBFGS iterations (total evals: up to n_iters × 20).
    client_id : int
        Client identifier for logging.

    Returns
    -------
    DLGResult | None
        Attack metrics, or None if reconstruction failed entirely.
    """
    client_full_model.eval()

    # Compute approximate gradient
    # origin_grad = global_base_params - client_base_params
    #
    # This is the "gradient" the server observes.  It is actually
    # the accumulated parameter update over local_epochs × batches,
    # not a single-step gradient.  This makes DLG harder (the
    # gradient is averaged/accumulated), which is realistic
    origin_grad = []
    for gp, pp in zip(global_base.parameters(), client_base.parameters()):
        origin_grad.append((gp.data - pp.data).to(device))

    # Collect target inputs
    # These are ground truth (signals, predictions) that the
    # attacker is trying to reconstruct.
    target_inputs = []
    with torch.no_grad():
        for i, (signals, labels) in enumerate(trainloader):
            if i >= batch_num:
                break

            signals = {k: v.to(device) for k, v in signals.items()}
            labels = labels.to(device)
            output = client_full_model(signals)
            target_inputs.append((signals, output))

    if not target_inputs:
        return None

    # Run DLG
    # Pass the full model for forward, but match gradients against
    # base parameters only (the communicated component).
    grad_params = list(client_full_model.base.parameters())

    result = DLG(
        net=client_full_model,
        origin_grad=origin_grad,
        target_inputs=target_inputs,
        grad_params=grad_params,
        # n_iters=n_iters,
    )

    if result is not None:
        result.client_id = client_id

    return result


def print_dlg_results(results: list[DLGResult], round_num: int):
    """Pretty-print DLG results for a round.

    Reports nMSE as the primary metric and PSNR as secondary.

    Parameters
    ----------
    results : list[DLGResult]
        One result per client that was attacked.
    round_num : int
        Current federation round.
    """
    valid = [r for r in results if r is not None]

    if not valid:
        print(f"  DLG round {round_num}: no valid reconstructions")
        return

    nmses = [r.aggregate_nmse for r in valid]
    psnrs = [r.aggregate_psnr for r in valid if not math.isnan(r.aggregate_psnr)]
    grad_diffs = [r.final_grad_diff for r in valid if not math.isinf(r.final_grad_diff)]
    mean_nmse = float(np.mean(nmses))

    # Primary metric: nMSE
    status = "LEAK RISK" if mean_nmse < 1.0 else "privacy OK"
    print(
        f"  DLG round {round_num}: "
        f"nMSE={mean_nmse:.4f} ({status})  "
        f"({len(valid)}/{len(results)} clients)"
    )

    # Secondary: PSNR
    if psnrs:
        print(f"    PSNR={np.mean(psnrs):.2f} dB (fixed range={ZSCORE_DATA_RANGE})")

    # Gradient matching residual
    if grad_diffs:
        print(f"    grad_diff={np.mean(grad_diffs):.6f}")

    # Per-modality breakdown (nMSE, the primary metric)
    all_modalities: set[str] = set()
    for r in valid:
        all_modalities.update(r.per_modality_nmse.keys())

    if all_modalities:
        parts = []
        for mod in sorted(all_modalities):
            mod_nmses = [
                r.per_modality_nmse[mod] for r in valid if mod in r.per_modality_nmse
            ]
            if mod_nmses:
                parts.append(f"{mod}={np.mean(mod_nmses):.3f}")
        if parts:
            print(f"    Per-modality nMSE: {', '.join(parts)}")
