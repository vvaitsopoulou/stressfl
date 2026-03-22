# FedAvg client: standard federated averaging with full-model training.
#
# This is the simplest FL client: it trains the FULL model (base + head)
# for local_epochs epochs each round.  No personalisation, no split.
#
# Adapts from:
#   - PFLIB clientavg.py    : the train() method is essentially just the
#                              base client's training loop (no additions)
#   - client_base.py        : our WESADClient base class
#
# FedAvg protocol (McMahan et al., 2017):
#   1. Server broadcasts full global model to all clients
#   2. Each client trains the full model locally for E epochs
#   3. Clients upload the full model to the server
#   4. Server averages all client models (weighted by sample count)
#   5. Repeat
#
# This file exists as a named class for clarity to parallel PFLlib's
# structure, even though FedAvgClient.train() is identical to the base
# WESADClient.train().  Having a separate class makes the algorithm name
# explicit in logs and configs.
#
# Literature:
#   McMahan et al. 2017: "Communication-Efficient Learning of Deep
#   Networks from Decentralized Data" : the FedAvg algorithm
#
# ──────────────────────────────────────────────────────────────────────
# TODO (other PFLIB-supported baselines, implement if time permits):
#   - fl/clients/client_prox.py   : FedProx (proximal term in loss)
#   - fl/clients/client_per.py    : FedAvg + fine-tuning
#   - fl/servers/server_prox.py   : FedProx server
#   - fl/servers/server_per.py    : FedAvg+FT server
#   - fl/servers/server_local.py  : Local-only baseline (no federation)
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

from fl.clients.client_base import WESADClient


# inherits WESADClient, adds nothing (useful for future extensibility though)
class FedAvgClient(WESADClient):
    """FedAvg client: trains the full model locally each round.

    This is functionally identical to WESADClient — the base class
    train() method already implements vanilla FedAvg local training
    (full model, local_epochs epochs, AdamW optimizer).

    The class exists for:
      1. Naming clarity in logs and server instantiation
      2. Structural parallel with PFLlib's clientAVG
      3. Future-proofing if FedAvg-specific behaviour is needed

    Mirrors PFLIB clientavg.py which is also just the base client's
    train() with no additions.
    """

    # No __init__ override needed — WESADClient.__init__ handles everything.
    # No train() override needed — WESADClient.train() is FedAvg training.
    pass
