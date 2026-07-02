import threading
import numpy as np

from metrics.metrics import*
from synchronize_weight import*

class TraininMultitAgent():


    def __init__(self,num_epochs,num_agents,synchronization_func):
        self.num_epochs = num_epochs
        self.num_agents = num_agents
        self.synchronization_func = synchronization_func


    def create_list_agents(self,num_agents,):

        for e in range(num_agents):
            trainloader = DataLoader(
                datasets_list[e],
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            trainloaders_list.append(trainloader)

            net = Net().to(DEVICE)
            # net = AlexNet(len(classes)).to(DEVICE)

            net_list.append(net)

            agent = Agent(
                agent_id=e,
                net=net_list[e],
                learning_rate=0.001,
                momentum=0.9,
                trainloader=trainloaders_list[e],
                testloader=testloader,
                device=DEVICE,
                neighbors=agent_list
            )
            agent_list.append(agent)