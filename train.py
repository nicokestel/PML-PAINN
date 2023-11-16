from nn.painn import PaiNN

from torch.optim import AdamW
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn

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

    experiment = 'qm9_atomwise'

    cutoff = 5.
    n_atom_basis = 128
    n_interactions = 3

    pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    painn = PaiNN(
        n_atom_basis=       128,
        n_interactions=     3,
        radial_basis=       spk.nn.radial.GaussianRBF(n_rbf=20, cutoff=5.),
        cutoff_fn=          spk.nn.cutoff.CosineCutoff(cutoff=5.)
    )
    pred_U0 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=QM9.U0)

    nnpot = spk.model.NeuralNetworkPotential(
        representation=painn,
        input_modules=[pairwise_distance],
        output_modules=[pred_U0],
        postprocessors=[trn.CastTo64(), trn.AddOffsets(QM9.U0, add_mean=True, add_atomrefs=True)]
    )

    print(painn)