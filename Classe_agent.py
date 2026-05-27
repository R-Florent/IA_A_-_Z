import threading
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch


class Agent:
    def __init__(self,
                 agent_id,
                 net,
                 learning_rate,
                 momentum,
                 trainloader,
                 testloader,
                 device):

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

        self.train_losses.append(avg_train_loss)
        self.train_accuracies.append(avg_train_accuracies)

        return avg_train_loss,avg_train_accuracies


    def validate(self):

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

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)

        return avg_loss, accuracy

    def train_and_validate(self):

        try:

            train_loss, train_acc = self.train_one_epoch()

            val_loss, val_acc = self.validate()

            print(f"Agent {self.id} OK")

        except Exception as e:

            print(f"Erreur agent {self.id} : {e}")

    def val_one_epoch(self):
        thread = threading.Thread(
            target=self.train_one_epoch,
            args=(self.model, self.trainloader, self.optimizer)
        )

    def communicate(self, other_agent):

        my_weights = self.model.state_dict()
        other_weights = other_agent.model.state_dict()

        averaged_weights = {}

        for key in my_weights:
            averaged_weights[key] = (my_weights[key] + other_weights[key]) / 2

        self.model.load_state_dict(averaged_weights)
        other_agent.model.load_state_dict(averaged_weights)

    def plot_metrics(self):

        epochs = range(1, len(self.train_losses)+1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # LOSS
        axes[0].plot(epochs, self.train_losses, label="Train Loss")
        axes[0].plot(epochs, self.val_losses, label="Validation Loss")

        axes[0].set_title(f"{self.id} Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        # ACCURACY
        axes[1].plot(epochs, self.val_accuracies, label="Val accuracies")
        axes[1].plot(epochs, self.train_accuracies, label="Train accuracies")


        axes[1].set_title(f"{self.id} Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()

        #plt.tight_layout()
        plt.show()