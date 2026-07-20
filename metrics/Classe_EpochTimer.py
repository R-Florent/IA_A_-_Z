# metrics/Classe_EpochTimer.py

import time

# ══════════════════════════════════════════════════════════════════════════════
# CALCUL TIME —  epoque + cumulatif
# ══════════════════════════════════════════════════════════════════════════════

class EpochTimer:
    """
    Chronomètre pour mesurer le temps de chaque époque.

    Usage :
        timer = EpochTimer()
        timer.start()
        ... entraînement ...
        elapsed = timer.stop()          # enregistre et retourne le temps
        print(timer.epoch_times)        # liste des temps par époque
        print(timer.cumulative_times)   # liste des temps cumulatifs
    """

    def __init__(self):
        self.epoch_times      : list[float] = []   # durée de chaque époque (s)
        self.cumulative_times : list[float] = []   # temps cumulatif (s)
        self._start_time      : float | None = None

    def start(self):
        """Démarre le chrono au début d'une époque."""
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """
        Arrête le chrono, enregistre le temps écoulé.
        Retourne la durée de l'époque en secondes.
        """
        if self._start_time is None:
            raise RuntimeError("EpochTimer.stop() appelé sans start() préalable.")

        elapsed = time.perf_counter() - self._start_time
        self.epoch_times.append(elapsed)

        cumul = (self.cumulative_times[-1] if self.cumulative_times else 0.0) + elapsed
        self.cumulative_times.append(cumul)

        self._start_time = None
        return elapsed

    def reset(self):
        """Remet le timer à zéro (nouveau run)."""
        self.epoch_times      = []
        self.cumulative_times = []
        self._start_time      = None
