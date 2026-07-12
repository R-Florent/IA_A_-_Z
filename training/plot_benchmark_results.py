# benchmark/plot_benchmark.py
import copy
import matplotlib.pyplot as plt


class RunResult:
    """Stocke les métriques d'un run complet pour une méthode de communication."""

    def __init__(self, method_name: str, agent_list: list, metrics: object):
        self.method_name   = method_name
        self.agent_list    = agent_list   # snapshot des agents après entraînement
        self.metrics       = metrics      # objet ModelMetrics loggé
        # Pratique : on extrait directement les listes de l'agent 0 pour comparer
        self.val_losses      = copy.deepcopy(agent_list[0].val_losses)
        self.val_accuracies  = copy.deepcopy(agent_list[0].val_accuracies)
        self.train_losses    = copy.deepcopy(agent_list[0].train_losses)
        self.train_accuracies= copy.deepcopy(agent_list[0].train_accuracies)

    def __repr__(self):
        final_val = self.val_accuracies[-1] if self.val_accuracies else 0
        return f"<RunResult method={self.method_name} | final_val_acc={final_val:.2f}%>"

def plot_benchmark_results(results: list[RunResult], metric: str = "val_loss"):
    """
    Trace une courbe par méthode de communication sur le même graphique.

    Args:
        results: liste de RunResult retournée par benchmark_sequential / benchmark_parallel
        metric:  "val_loss" | "val_accuracy" | "train_loss" | "train_accuracy"
    """
    metric_map = {
        "val_loss"       : ("val_losses",       "Validation Loss",     "Loss"),
        "val_accuracy"   : ("val_accuracies",    "Validation Accuracy", "Accuracy (%)"),
        "train_loss"     : ("train_losses",      "Train Loss",          "Loss"),
        "train_accuracy" : ("train_accuracies",  "Train Accuracy",      "Accuracy (%)"),
    }

    if metric not in metric_map:
        raise ValueError(f"metric doit être parmi {list(metric_map.keys())}")

    attr, title, ylabel = metric_map[metric]

    plt.figure(figsize=(10, 6))

    for result in results:
        values = getattr(result, attr)
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=result.method_name, marker="o", markersize=3)

    plt.title(f"Comparaison des méthodes — {title} (Agent 0)")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_all_benchmark_metrics(results: list[RunResult]):
    """
    Trace les 4 métriques en une seule figure (2x2).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Benchmark des méthodes de communication décentralisée", fontsize=14)

    metrics_cfg = [
        ("val_losses",       "Validation Loss",     "Loss",         axes[0, 0]),
        ("val_accuracies",   "Validation Accuracy", "Accuracy (%)", axes[0, 1]),
        ("train_losses",     "Train Loss",          "Loss",         axes[1, 0]),
        ("train_accuracies", "Train Accuracy",      "Accuracy (%)", axes[1, 1]),
    ]

    for attr, title, ylabel, ax in metrics_cfg:
        for result in results:
            values = getattr(result, attr)
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, label=result.method_name, marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_benchmark_summary(results: list[RunResult]):
    """
    Affiche un tableau récapitulatif final dans la console.
    """
    print("\n" + "="*70)
    print(f"{'Méthode':<30} {'Val Loss':>10} {'Val Acc':>10} {'Train Acc':>10}")
    print("="*70)

    for r in results:
        val_loss = r.val_losses[-1]      if r.val_losses      else float("nan")
        val_acc  = r.val_accuracies[-1]  if r.val_accuracies  else float("nan")
        train_acc= r.train_accuracies[-1]if r.train_accuracies else float("nan")
        print(f"{r.method_name:<30} {val_loss:>10.4f} {val_acc:>9.2f}% {train_acc:>9.2f}%")

    print("="*70)



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
        agent_list = genereate_agent(BATCH_SIZE, N_AGENT, DEVICE)

        metrics = _train_agent_run(agent_list, num_epochs, graph, k, communication_fn)

        result = RunResult(
            method_name=method_name,
            agent_list=agent_list,
            metrics=metrics,
        )
        results.append(result)
        print(result)

    return results