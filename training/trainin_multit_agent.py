import threading
import numpy as np

from metrics.metrics import*
from synchronize_weight import*

class TraininMultitAgent():


    def __init__(self,num_epochs,num_agents,synchronization_func):
        self.num_epochs = num_epochs
        self.num_agents = num_agents
        self.synchronization_func = synchronization_func


