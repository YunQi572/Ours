import torch
import torch.nn
import numpy as np
import torch.nn.functional as f

class BaseServer:
    def __init__(self, args, message_pool):
        self.args = args
        # self.clients = clients
        # self.model = model
        # self.data = data
        self.message_pool = message_pool

    def run(self):
        pass

    def communicate(self):
        pass

    def aggregate(self):
        pass

    def global_evaluate(self):
        pass


class BaseClient():
    def __init__(self, args, client_id, data):
        self.args = args
        self.client_id = client_id
        # self.model = model
        self.data = data
    
    def train(self):
        pass