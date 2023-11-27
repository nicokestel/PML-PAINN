import os

from data_loader import load_data

import torch
import torch.nn.functional as F

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.datasets.md17 import MD17
from schnetpack.units import convert_units


def run(path_to_model, path_to_data_dir, molecule='ethanol'):
    """ Evaluates the specified MLP baseline model on MD17.

    Args:
        path_to_model: where the model is stored.
        path_to_data_dir: where the data is stored.
        molecule: (default: 'ethanol').
    """

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model_path = os.path.join(path_to_model)
    best_model = torch.load(model_path, map_location=device)
    best_model.eval()

    print(best_model)

    # load dataset
    bs = 10
    dataset = load_data('md17',
                        molecule=molecule,
                        transformations=[
                            trn.ASENeighborList(cutoff=5.),
                            trn.RemoveOffsets(
                                MD17.energy, remove_mean=True, remove_atomrefs=False),
                            trn.CastTo32()
                        ],
                        n_train=950,  # 950
                        n_val=50,  # 50
                        batch_size=bs,
                        work_dir=path_to_data_dir)

    n_atoms = dataset.train_dataset[0]['_n_atoms'].item()
    n_test = i = 100

    # Compute MAE + MSE
    forces_avg_mae = 0.0
    forces_avg_mse = 0.0
    for _, b in enumerate(dataset.test_dataloader()):
        if i == 0:
            break

        batch = b['_positions'].view(bs, n_atoms, -1).clone().detach().to(device)
        cdist = torch.cdist(batch, batch).view(bs, -1).to(device)
        pred = best_model(cdist).cpu()

        # Copnvert units
        # results['energy'] *= convert_units("kcal/mol", "eV")

        # magnitude of forces
        f_mag = torch.linalg.norm(b['forces'].view(bs, n_atoms, 3), dim=-1)

        # MAE
        forces_avg_mae += F.l1_loss(pred, f_mag)

        # MSE
        forces_avg_mse += F.mse_loss(pred, f_mag)

        i -= 1

    forces_avg_mae /= n_test
    forces_avg_mse /= n_test

    print('MAE on forces: {:.3f}'.format(
        forces_avg_mae))
    print('MSE on forces: {:.3f}'.format(
        forces_avg_mse))
