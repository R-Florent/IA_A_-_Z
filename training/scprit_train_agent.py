import threading
from metrics.Classe_model_metrics import ModelMetrics
from synchronize_weight import*

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

        communication_model(agent_list,graph,k)

        metrics.log_metrics(epoch)

    print("Les modèles ont fini l'entraînement")
    return metrics