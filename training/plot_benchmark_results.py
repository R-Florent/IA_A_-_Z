# benchmark/plot_benchmark.py
import copy
import matplotlib.pyplot as plt
import numpy as np


class RunResult:
    """Stocke les métriques d'un run complet pour une méthode de communication."""

    def __init__(self, method_name: str, agent_list: list, metrics: object):
        self.epochs_int = int
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



def plot_communication_comparison(results: list[RunResult]):
    """
    Compare the communication cost of all decentralized FL methods.

    Two plots are generated:
        1. Number of messages exchanged at each epoch.
        2. Cumulative number of messages exchanged over training.

    Args:
        results: list of RunResult objects. Each result must contain
                 a CommunicationCostMetric in `result.comm_cost`.
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    fig.suptitle(
        "Communication Cost Benchmark",
        fontsize=14
    )

    for result in results:

        comm = result.comm_cost
        snapshots = comm.snapshots

        # Messages exchanged during each epoch
        messages_per_epoch = [
            snapshot.n_messages
            for snapshot in snapshots
        ]

        # Cumulative number of messages
        cumulative_messages = np.cumsum(messages_per_epoch)

        epochs = range(1, len(messages_per_epoch) + 1)

        # ─────────────────────────────────────────────
        # Messages per epoch
        # ─────────────────────────────────────────────

        axes[0].plot(
            epochs,
            messages_per_epoch,
            label=result.method_name,
            marker="o",
            markersize=3
        )

        # ─────────────────────────────────────────────
        # Cumulative messages
        # ─────────────────────────────────────────────

        axes[1].plot(
            epochs,
            cumulative_messages,
            label=result.method_name,
            marker="o",
            markersize=3
        )

    # ─────────────────────────────────────────────────
    # Plot 1
    # ─────────────────────────────────────────────────

    axes[0].set_title("Messages per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Number of Messages")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # ─────────────────────────────────────────────────
    # Plot 2
    # ─────────────────────────────────────────────────

    axes[1].set_title("Cumulative Communication Messages")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cumulative Number of Messages")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()