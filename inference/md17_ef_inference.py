import os
import copy

from data_loader import load_data

import torch
import torch.nn.functional as F


import schnetpack as spk
import schnetpack.transform as trn

from schnetpack.datasets.md17 import MD17

from schnetpack.units import convert_units


def run(path_to_model, path_to_data_dir, molecule='ethanol'):
    """ Evaluates the specified PaiNN model on MD17.

        Args:
            path_to_model: where the model is stored.
            path_to_data_dir: where the data is stored.
            molecule: (default: 'ethanol').
    """

    # set device
    device = torch.device("cpu")

    # load model
    model_path = os.path.join(path_to_model)
    best_model = torch.load(model_path, map_location=device)
    best_model.eval()

    # load MD17 data
    dataset = load_data('md17',
                        molecule=molecule,
                        transformations=[
                            trn.ASENeighborList(cutoff=5.),
                            trn.RemoveOffsets(MD17.energy, remove_mean=True, remove_atomrefs=False),
                            trn.CastTo32()
                        ],
                        n_train=950,  # 950
                        n_val=50,  # 50
                        batch_size=10,
                        work_dir=path_to_data_dir)

   
    n_test = i = 100

    # compute MAE
    energy_avg_mae, forces_avg_mae = 0.0, 0.0
    energy_avg_mse, forces_avg_mse = 0.0, 0.0
    for _, batch in enumerate(dataset.test_dataloader()):
        if i == 0:
            break

        results = best_model(copy.deepcopy(batch))

        # Convert units
        # results['energy'] *= convert_units("kcal/mol", "eV")

        # MAE
        energy_avg_mae += F.l1_loss(results['energy'], batch['energy'])
        forces_avg_mae += F.l1_loss(results['forces'], batch['forces'])

        # MSE
        energy_avg_mse += F.mse_loss(results['energy'], batch['energy'])
        forces_avg_mse += F.mse_loss(results['forces'], batch['forces'])

        i -= 1


    energy_avg_mae /= n_test
    forces_avg_mae /= n_test
    energy_avg_mse /= n_test
    forces_avg_mse /= n_test

    print('MAE on energy: {:.3f}'.format(
        energy_avg_mae))
    print('MAE on forces: {:.3f}'.format(
        forces_avg_mae))
    print('MSE on energy: {:.3f}'.format(
        energy_avg_mse))
    print('MSE on forces: {:.3f}'.format(
        forces_avg_mse))
