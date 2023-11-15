from nn.painn import PaiNN
import schnetpack.nn as snn
from torch.optim import AdamW

def Trainer():

    def __init__(self, config, model):
        self.config = config
        self.model = model

        self.optim = AdamW(lr=5e-4, weight_decay=0)

        # TODO different objectives/setup (e.g. learning rate, loss functions) for different experiments
    
    def run(self):
        # TODO simple training loop
        pass


if __name__ == '__main__':

    painn = PaiNN(
        n_atom_basis=       128,
        n_interactions=     3,
        radial_basis=       snn.radial.GaussianRBF(n_rbf=20, cutoff=5.),
        cutoff_fn=          snn.cutoff.CosineCutoff(cutoff=5.)
    )

    print(painn)