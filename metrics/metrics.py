import copy
import matplotlib.pyplot as plt
import torch

def node_weight_metric(agent_list):

    avg_state_dict = copy.deepcopy(
        agent_list[0].model.state_dict()
    )

    # moyenne
    for key in avg_state_dict:

        for i in range(1, len(agent_list)):
            avg_state_dict[key] += (
                agent_list[i].model.state_dict()[key]
            )

        avg_state_dict[key] /= len(agent_list)

    # différences
    for agent_id, agent in enumerate(agent_list):

        print(f"\nAgent {agent_id}")

        total_distance = 0

        for key in avg_state_dict:

            diff = (
                avg_state_dict[key]
                - agent.model.state_dict()[key]
            )

            distance = torch.norm(diff).item()

            total_distance += distance

            print(
                f"{key:<30} "
                f"L2 distance = {distance:.6f}"
            )

        print(
            f"Distance totale = {total_distance:.6f}"
        )


def plot_all_agents_metrics(agent_list):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ===== LOSS =====
    for agent in agent_list:

        train_epochs = range(1, len(agent.train_losses) + 1)
        val_epochs = range(1, len(agent.loss_validate_list) + 1)

        axes[0].plot(
            train_epochs,
            agent.train_losses,
            label=f"Agent {agent.id} Train"
        )

        axes[0].plot(
            val_epochs,
            agent.loss_validate_list,
            linestyle='--',
            label=f"Agent {agent.id} Val"
        )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # ===== ACCURACY =====
    for agent in agent_list:

        train_epochs = range(1, len(agent.train_accuracies) + 1)
        val_epochs = range(1, len(agent.accuracy_validate_list) + 1)

        axes[1].plot(
            train_epochs,
            agent.train_accuracies,
            label=f"Agent {agent.id} Train"
        )

        axes[1].plot(
            val_epochs,
            agent.accuracy_validate_list,
            linestyle='--',
            label=f"Agent {agent.id} Val"
        )

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_all_agent_node_weight_metric(agent_list):

    for agent in agent_list:
        plt.plot(
            agent.total_distance_list,
            label=f"Agent {agent.id}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Total Distance")
    plt.title("total_distance_node_weight")
    plt.legend()
    plt.show()


