# benchmark/benchmark.py
import sys
from pathlib import Path
import copy
import threading
from metrics.Classe_model_metrics import ModelMetrics
from metrics.Classe_EpochTimer import EpochTimer
from agents.Classe_agent import Agent
from training.scrpit_generate_agent import generate_agent
sys.path.insert(1, '')


# ──────────────────────────────────────────────────────────────
# Structure de résultat pour un run
# ──────────────────────────────────────────────────────────────

class RunResult:
    """Stocke les métriques d'un run complet pour une méthode de communication."""

    def __init__(self, method_name: str, agent_list: list, metrics: object,timer: EpochTimer):
        self.method_name   = method_name
        self.agent_list    = agent_list   # snapshot des agents après entraînement
        self.metrics       = metrics      # objet ModelMetrics loggé
        self.timer = timer
        # Pratique : on extrait directement les listes de l'agent 0 pour comparer
        self.val_losses      = copy.deepcopy(agent_list[0].val_losses)
        self.val_accuracies  = copy.deepcopy(agent_list[0].val_accuracies)
        self.train_losses    = copy.deepcopy(agent_list[0].train_losses)
        self.train_accuracies= copy.deepcopy(agent_list[0].train_accuracies)

    def __repr__(self):
        final_val = self.val_accuracies[-1] if self.val_accuracies else 0
        return f"<RunResult method={self.method_name} | final_val_acc={final_val:.2f}%>"


# ──────────────────────────────────────────────────────────────
# Entraînement d'une liste d'agents (identique à ton train_agent)
# ──────────────────────────────────────────────────────────────

def _train_agent_run(agent_list, num_epochs, graph, k, communication_fn) -> object:
    """
    Entraîne agent_list pendant num_epochs avec communication_fn.
    Retourne l'objet ModelMetrics loggé.
    """
    metrics = ModelMetrics(agent_list)

    for epoch in range(num_epochs):

        # ── Entraînement parallèle (thread par agent) ──────────────
        threads = [
            threading.Thread(target=agent.train_and_validate, args=[agent_list])
            for agent in agent_list
        ]
        for t in threads:
            t.start()

        print(f"  [Epoch {epoch + 1}/{num_epochs}] threads lancés …")

        for t in threads:
            t.join()

        # ── Distance inter-agents ──────────────────────────────────
        distances = Agent.node_weight_metric(agent_list)
        for agent, dist in zip(agent_list, distances):
            agent.total_distance_list.append(dist)

        # ── Synchronisation selon la méthode choisie ───────────────
        communication_fn(agent_list, graph, k)

        metrics.log_metrics(epoch)

    print("  ✓ Entraînement terminé\n")
    return metrics


# ──────────────────────────────────────────────────────────────
# Benchmark séquentiel
# ──────────────────────────────────────────────────────────────

def benchmark_sequential(
    communication_methods: dict,   # {"nom": fn, ...}
    num_epochs: int,
    graph,
    k: int,
    BATCH_SIZE: int,
    N_AGENT: int,
    DEVICE,
) -> list[RunResult]:
    """
    Entraîne chaque méthode l'une après l'autre.
    Les agents sont RÉINITIALISÉS entre chaque méthode pour une comparaison équitable.

    Args:
        communication_methods: dict {label: fn} — ex. {"average": synchronize_models_average}
        num_epochs:            nombre d'époques par run
        graph:                 topologie réseau (NetworkX graph)
        k:                     paramètre K pour consensu_algortyme
        BATCH_SIZE:            taille du batch
        N_AGENT:               nombre d'agents
        DEVICE:                torch.device

    Returns:
        Liste de RunResult (un par méthode)
    """
    results: list[RunResult] = []

    for method_name, communication_fn in communication_methods.items():
        print(f"\n{'='*60}")
        print(f"  RUN — méthode : {method_name}")
        print(f"{'='*60}")

        # ── Agents frais à chaque run ──────────────────────────────
        agent_list = generate_agent(BATCH_SIZE, N_AGENT, DEVICE)

        metrics = _train_agent_run(agent_list, num_epochs, graph, k, communication_fn)

        result = RunResult(
            method_name=method_name,
            agent_list=agent_list,
            metrics=metrics,
        )

        results.append(result)
        print(result)

    return results