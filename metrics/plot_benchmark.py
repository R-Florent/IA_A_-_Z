import matplotlib.pyplot as plt
from training.benchmark import RunResult


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