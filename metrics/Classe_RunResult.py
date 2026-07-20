
import copy
from metrics.Classe_EpochTimer import EpochTimer


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

