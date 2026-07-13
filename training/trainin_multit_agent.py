import threading
import numpy as np
import
from metrics.metrics import*
from synchronize_weight import*
from scrpit_generate_agent import genereate_agent
from topologies.NetworkTopology import NetworkTopology

class TraininMultitAgent():


    def __init__(self,num_epochs,num_agents,synchronization_func):
        self.num_epochs = num_epochs
        self.num_agents = num_agents
        self.synchronization_func = synchronization_func


    def train(self):


