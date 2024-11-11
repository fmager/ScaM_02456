import unittest
import torch
import numpy as np

from src.ScaM import tracker
from src.ScaM import metrics

class TestScaM(unittest.TestCase):
    def setUp(self):
        self.model = AutoEncoder()
        self.input = torch.randn(20, 5*5)
        rand_label = np.random.randint(0, 5, 20)
        rand_letter = np.random.choice(['X', 'Y', 'Z'], 20)
        self.label = {'A': rand_label, 'B': rand_letter}
        self.cust_sel_fn = [lambda x: x, lambda x: x+1]
        self.tracker = tracker.model_tracker(self.model, ["encoder", "decoder"], cust_select_fn=self.cust_sel_fn)

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(5*5, 4*4),
            torch.nn.ReLU(),
            torch.nn.Linear(4*4, 3*3),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3*3, 4*4),
            torch.nn.ReLU(),
            torch.nn.Linear(4*4, 5*5),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    unittest.main()