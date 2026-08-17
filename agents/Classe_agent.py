import threading
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import copy
import time

class Agent:
    def __init__(self,
                 agent_id,
                 net,
                 learning_rate,
                 momentum,
                 trainloader,
                 testloader,
                 device,
                 neighbors):

        self.id = agent_id

        self.device = device

        self.model = net.to(device)

        self.trainloader = trainloader

        self.testloader = testloader

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum
        )

        self.neighbors = []

        # Metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.loss_validate_list = []        # list of validate loss use outise the méthode of the classe for doing metric evry were we whant
        self.accuracy_validate_list = []    #list of validate accuracy use outise the méthode of the classe for doing metric evry were we whant
        self.total_distance_list = []       #list of the distance bewteen avrage of the node weight and curent node weight of this agent
        self.avg_state_dict_list = []       #list of the distance avrage of the node weight of evry agent
        self.weight_std_list = []           # Std des poids de CET agent par époque
        self.epoch_compute_times = []


    def train_one_epoch(self):

        print(f"Agent {self.id} start")
        self.model.train()

        correct = 0
        total = 0

        running_loss = 0.0

        for inputs, labels in self.trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Loss train
        avg_train_loss = running_loss / len(self.trainloader)
        avg_train_accuracies = 100 * correct / total

        print(f"Model{self.id}"f" - Acc:{avg_train_accuracies} % - Loss:{avg_train_loss} %")
        return avg_train_loss,avg_train_accuracies


    def validate_one_epoch(self):

        self.model.eval()

        running_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for inputs, labels in self.testloader:

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(self.testloader)

        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def train_and_validate(self,agent_list):

        try:
            val_loss, val_acc = self.validate_one_epoch()

            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            start = time.perf_counter()
            train_loss, train_acc = self.train_one_epoch()
            elapsed = time.perf_counter() - start
            self.epoch_compute_times.append(elapsed)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            return train_loss, train_acc, val_loss, val_acc
            print(f"Agent {self.id} OK")

        except Exception as e:

            print(f"Erreur agent {self.id} : {e}")

    @staticmethod
    def node_weight_metric(agent_list: list) -> list[float]:
        """
        Calcule la distance euclidienne (norme L2 globale) entre chaque agent
        et le modèle moyen de tous les agents.
        Args:
            agent_list: liste de tous les agents

        Returns:
            list[float]: distance L2 globale de chaque agent au modèle moyen
        """

        """
         Step 1: Compute the average model across all agents.
         A deep copy is used to create an independent accumulator so that
         the original model parameters remain unchanged.
        """
        
        avg_state_dict = copy.deepcopy(agent_list[0].model.state_dict())

        for key in avg_state_dict:
            for i in range(1, len(agent_list)):
                avg_state_dict[key] = avg_state_dict[key] + \
                                      agent_list[i].model.state_dict()[key]
            # Division par le nombre total d'agents
            avg_state_dict[key] = avg_state_dict[key] / len(agent_list)

        """
        # Step 2: Compute the global L2 distance between each agent and the
        # average model.
        #
        # The distance is computed as
        #
        #     sqrt( Σ ||Δ_layer||² )
        #
        # which is equivalent to the Euclidean norm of the concatenated
        # parameter vector.
        """

        total_distance_list = []

        for agent in agent_list:

            agent_state = agent.model.state_dict()
            sum_of_squared_norms = 0.0

            for key in avg_state_dict:
                # Différence param par param pour cette couche
                diff = avg_state_dict[key] - agent_state[key]

                # ‖diff_layer‖₂² — on accumule le CARRÉ
                squared_norm = torch.norm(diff).item() ** 2
                sum_of_squared_norms += squared_norm

            # Racine carrée finale → vraie norme L2 dans l'espace global
            total_distance = sum_of_squared_norms ** 0.5
            total_distance_list.append(total_distance)

        return total_distance_list