import copy
from agents.Classe_agent import Agent
from topologies.NetworkTopology import NetworkTopology


# In main.py or training.py
def synchronize_with_topology(agent_list, graph):
    """
    Synchronize all agents according to network topology
    Only neighbors communicate
    """
    for agent in agent_list:
        neighbors_ids = NetworkTopology.get_neighbors(graph, agent.id)

        for neighbor_id in neighbors_ids:
            neighbor = agent_list[neighbor_id]
            # Average weights between neighbors
            my_weights = agent.model.state_dict()
            neighbor_weights = neighbor.model.state_dict()

            averaged = {
                key: (my_weights[key] + neighbor_weights[key]) / 2
                for key in my_weights
            }

            agent.model.load_state_dict(averaged)

def consensus_step(agent_list, graph):

    new_weights = {}

    for agent in agent_list:

        neighbors = NetworkTopology.get_neighbors(graph, agent.id)

        current = agent.model.state_dict()

        avg = {
            k: current[k].clone()
            for k in current
        }

        count = 1

        for n in neighbors:
            neighbor_weights = agent_list[n].model.state_dict()

            for k in avg:
                avg[k] += neighbor_weights[k]

            count += 1

        for k in avg:
            avg[k] /= count

        new_weights[agent.id] = avg

    # Mise à jour simultanée
    for agent in agent_list:
        agent.model.load_state_dict(new_weights[agent.id])

def consensu_algortyme(agent_list, graph, K=5):

    r = {}
    for agent_id, agent in enumerate(agent_list):
        r[agent_id] = copy.deepcopy(agent.model.state_dict())

    for k in range(K):

        r_new = {}

        for agent_id, agent in enumerate(agent_list):

            participants = [agent.id] + list(graph.neighbors(agent.id))

            avg_state = copy.deepcopy(agent_list[participants[0]].model.state_dict())

            for key in avg_state:

                for participant_id  in participants[1:]:

                    avg_state[key] += r[participant_id][key]

                avg_state[key] /= len(participants)

            r_new[agent_id] = avg_state

        r = r_new

        for agent_id, agent in enumerate(agent_list):
            agent.model.load_state_dict(r[agent_id])

        distances = Agent.node_weight_metric(agent_list)
        avg_distance = sum(distances) / len(distances)
        print(f"    → Distance moyenne après iter {k+1}: {avg_distance:.6f}")

    return r

def synchronize_models_average(agent_list):

    # récupérer les poids du premier modèle
    avg_state_dict = copy.deepcopy(agent_list[0].model.state_dict())

    # somme des poids
    for key in avg_state_dict:

        for i in range(1, len(agent_list)):
            avg_state_dict[key] += agent_list[i].model.state_dict()[key]

        # moyenne
        avg_state_dict[key] = avg_state_dict[key] / len(agent_list)

    # appliquer les poids moyens à tous les agents
    for agent in agent_list:
        agent.model.load_state_dict(avg_state_dict)

def synchronize_models_cycle(agent_list):
    state_dict_list = []

    for agent in agent_list:
        state_dict = copy.deepcopy(agent.model.state_dict())
        state_dict_list.append(state_dict)

    for e,agent in enumerate(agent_list):
        agent.model.load_state_dict(state_dict_list[(e+1) % len(agent_list)])


    # Charger les nouveaux poid

