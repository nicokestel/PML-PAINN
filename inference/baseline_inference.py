import os
import copy

from data_loader import load_data

import torch
import torch.nn.functional as F

import numpy as np
from ase import Atoms

import schnetpack as spk
import schnetpack.transform as trn

from schnetpack.datasets.md17 import MD17


def run(path_to_model, path_to_data_dir, molecule='ethanol'):
    # set device
    device = torch.device("cpu")

    # load model
    model_path = os.path.join(path_to_model)
    best_model = torch.load(model_path, map_location=device)
    best_model.eval()

    print(best_model)

    # load dataset
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
                        batch_size=10,
                        work_dir=path_to_data_dir)

    # print(dataset.test_idx)

    # create atoms object from dataset
    # splits = random_split(dataset.test_dataset, [1/3, 1/3, 1/3])
    n_test = i = 100
    # splits = dataset.test_dataset[:n_test//3], dataset.test_dataset[n_test//3 : 2*n_test//3], dataset.test_dataset[2*n_test//3:]

    # compute MAE + MSE
    forces_avg_mae = 0.0
    forces_avg_mse = 0.0
    for _, b in enumerate(dataset.test_dataloader()):
        if i == 0:
            break

        batch = b['forces'].view(10, -1).clone().detach().to(device)
        pred = best_model(batch)

        # convert units
        # forces stays same
        # results['energy'] *= convert_units("kcal/mol", "eV")

        # MAE
        forces_avg_mae += F.l1_loss(pred, batch)

        # MSE
        forces_avg_mse += F.mse_loss(pred, batch)

        i -= 1

    forces_avg_mae /= n_test
    forces_avg_mse /= n_test

    print('MAE on forces: {:.3f}'.format(
        forces_avg_mae))
    print('MSE on forces: {:.3f}'.format(
        forces_avg_mse))
