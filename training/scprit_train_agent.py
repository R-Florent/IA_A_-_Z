import threading
from metrics.Classe_model_metrics import ModelMetrics
from synchronize_weight import*
import copy
import threading
from agents.Classe_agent import Agent
from metrics.Classe_model_metrics import ModelMetrics
from metrics.metrics import (log_weight_std,log_distance_to_mean,EpochTimer)


def train_agent(agent_list,num_epochs,graph,k,communication_model):

    metrics = ModelMetrics(agent_list)

    for epoch in range(num_epochs):

        threads_list = []

        # THREADS
        for agent in agent_list:
            thread = threading.Thread(
                target=agent.train_and_validate,
                args=[agent_list]
            )
            threads_list.append(thread)

        # Start des deux modèles
        for thread in threads_list:
            thread.start()

        print(f"Epoch {epoch+1}/{num_epochs}")

        # Attendre la fin
        for thread in threads_list:
            thread.join()

        distances = Agent.node_weight_metric(agent_list)

        for agent, distance in zip(agent_list, distances):
            agent.total_distance_list.append(distance)

        communication_model(agent_list,graph,k,epoch,num_epochs)

        metrics.log_metrics(epoch)

    print("Les modèles ont fini l'entraînement")
    return metrics


def train_agent_chrono(agent_list, num_epochs, graph, k, communication_fn) -> tuple[object, EpochTimer]:

    metrics = ModelMetrics(agent_list)
    timer = EpochTimer()        # ⏱
    for epoch in range(num_epochs):

        elapsed = timer.start()
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
        communication_fn(agent_list,graph,k,epoch,num_epochs)

        elapsed = timer.stop()      # ⏱

        #metrics.print_all_metrics(epoch)
        metrics.log_metrics(epoch)

    print("Les modèles ont fini l'entraînement")
    return metrics, timer