
import copy
from metrics.Classe_EpochTimer import EpochTimer


class RunResult:
    """Stocke les métriques d'un run complet pour une méthode de communication."""

    def __init__(self, method_name: str, agent_list: list, metrics: object, timer: EpochTimer ,comm_cost: "CommunicationCost"):
        self.method_name   = method_name
        self.agent_list    = agent_list   # snapshot des agents après entraînement
        self.metrics       = metrics      # objet ModelMetrics loggé
        self.timer         = timer
        self.comm_cost     = comm_cost
        # Pratique : on extrait directement les listes de l'agent 0 pour comparer
        self.val_losses      = copy.deepcopy(agent_list[0].val_losses)
        self.val_accuracies  = copy.deepcopy(agent_list[0].val_accuracies)
        self.train_losses    = copy.deepcopy(agent_list[0].train_losses)
        self.train_accuracies= copy.deepcopy(agent_list[0].train_accuracies)


    def __repr__(self) -> str:
        """
        CORRECTION : ajout du résumé timer dans le repr
        pour visualiser immédiatement les perfs de chaque méthode.

            <RunResult method=X | final_val_acc=92.30%
             | total=45.2s | mean=9.04s/ep
             | fastest=ep3(8.1s) | slowest=ep1(11.2s)>
        """
        final_val = self.val_accuracies[-1] if self.val_accuracies else 0.0
        final_loss = self.val_losses[-1] if self.val_losses else 0.0
        fast_i, fast_t = self.timer.fastest_epoch
        slow_i, slow_t = self.timer.slowest_epoch

        return (
            f"\n{'═' * 55}\n"
            f"  RunResult : {self.method_name}\n"
            f"{'─' * 55}\n"
            f"  Accuracy  finale : {final_val:.2f}%\n"
            f"  Val Loss  finale : {final_loss:.4f}\n"
            f"  Temps total      : {self.timer.total_time:.2f}s\n"
            f"  Temps moyen/ép.  : {self.timer.mean_epoch_time:.2f}s\n"
            f"  Époque la + rapide : ép.{fast_i} ({fast_t:.2f}s)\n"
            f"  Époque la + lente  : ép.{slow_i} ({slow_t:.2f}s)\n"
            f"{'═' * 55}"
        )
