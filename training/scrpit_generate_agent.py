import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import random_split, DataLoader

from models.model import Net,AlexNet
from metrics.metrics import *
from synchronize_weight import*


def genereate_agent(BATCH_SIZE,N_AGENT,DEVICE):

    transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_size = len(trainset)

    split = train_size // 9
    remainder = train_size % N_AGENT
    print(remainder)

    # Crée les parts
    splits = [split] * N_AGENT
    splits[0] += remainder  # Ajoute le reste au premier agent

    print(f"Train size: {train_size}, Splits: {splits}, Sum: {sum(splits)}")

    trainloaders_list = []
    net_list = []
    agent_list = []
    split = train_size // N_AGENT

    datasets_list = random_split(trainset, [split] * N_AGENT)#split dataset in N agent pefectly

    for e in range(N_AGENT):

        trainloader = DataLoader(
            datasets_list[e],
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        trainloaders_list.append(trainloader)

        net = Net().to(DEVICE)
        #net = AlexNet(len(classes)).to(DEVICE)

        net_list.append(net)

        agent = Agent(
            agent_id=e,
            net=net_list[e],
            learning_rate=0.001,
            momentum=0.9,
            trainloader=trainloaders_list[e],
            testloader=testloader,
            device=DEVICE,
            neighbors = agent_list
        )
        agent_list.append(agent)

    return agent_list