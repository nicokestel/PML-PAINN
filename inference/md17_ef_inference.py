import os
import copy

from data_loader import load_data

import numpy as np

import torch
import torch.nn.functional as F

import schnetpack as spk
import schnetpack.transform as trn

from schnetpack.datasets.md17 import MD17

from schnetpack.units import convert_units

from ase import Atoms

import matplotlib.pyplot as plt


def run(model, path_to_data_dir, molecule='ethanol'):
    """ Evaluates the specified PaiNN model on MD17.

        Args:
            model: path to where the model is stored or model object.
            path_to_data_dir: where the data is stored.
            molecule: (default: 'ethanol').

        Returns:
            test set metrics, i.e. MAE
    """

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    if type(model) is str:
        model_path = os.path.join(model)
        best_model = torch.load(model_path, map_location=device)
    else:
        best_model = model
    best_model.eval()

    # remove add offsets postprocessor
    nnp = [module for module in best_model.modules() if isinstance(module, spk.model.NeuralNetworkPotential)][0]
    nnp.postprocessors.__delattr__('1')

    # load MD17 data
    bs = 10
    dataset = load_data('md17',
                        molecule=molecule,
                        transformations=[
                            trn.ASENeighborList(cutoff=5.),
                            trn.RemoveOffsets(MD17.energy, remove_mean=True, remove_atomrefs=False),
                            trn.CastTo32()
                        ],
                        n_train=950,  # 950
                        n_val=50,  # 50
                        batch_size=bs,
                        work_dir=path_to_data_dir)

    # compute MAE
    metrics = {'energy': 0.0, 'forces': 0.0}

    # use 10000 batches for testing
    n_batches = 10000

    print('length test set:', n_batches * bs)
    
    #dataset.test_dataset = dataset.test_dataset[:n_batches*bs]
    for i, batch in enumerate(dataset.test_dataloader()):
        
        # keep terminal active and pipe alive
        if i % 1000 == 0:
            print(i)
        if i >= n_batches:
            break


        b = {k: v.to(device) for k, v in batch.items()}
        results = best_model(b)
        results = {k: v.to('cpu') for k, v in results.items()}

        # Convert units [DON'T!! WE REPORT IN KCAL/MOL]
        #results['energy'] *= convert_units("kcal/mol", "eV")
        #results['forces'] *= convert_units("kcal/mol", "eV")
        
        # MAE
        metrics['energy'] += F.l1_loss(results['energy'], batch['energy'])
        metrics['forces'] += F.l1_loss(results['forces'], batch['forces'])

    metrics['energy'] = round(metrics['energy'].item() / n_batches, 5)
    metrics['forces'] = round(metrics['forces'].item() / n_batches, 5)

    print('MAE of energies in [kcal/mol] and forces in [kcal/mol/Angstrom]')
    print(metrics)

    return metrics


def val_mean_std(model, path_to_data_dir, molecule='ethanol'):
    """ Predicts test set labels of the specified PaiNN model on MD17.

        Args:
            model: path to where the model is stored or model object.
            path_to_data_dir: where the data is stored.
            molecule: (default: 'ethanol').
            sample_idx: (default: 0)

        Returns:
            predicted energies and forces for sample with index `sample_idx`
            confidence measure
    """

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    if type(model) is str:
        model_path = os.path.join(model)
        best_model = torch.load(model_path, map_location=device)
    else:
        best_model = model
    best_model.eval()

    # remove add offsets postprocessor [ONLY REMOVE WHEN MEASURING ERRORS]
    nnp = [module for module in best_model.modules() if isinstance(module, spk.model.NeuralNetworkPotential)][0]
    nnp.postprocessors.__delattr__('1')

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
    
    # set up converter
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32, device=device
    )

    #structure = dataset.test_dataset[sample_idx]
    e_maes, f_maes = [], []

    for structure in dataset.val_dataset:
        # create atoms object from dataset
        atoms = Atoms(
            numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
        )

        # convert atoms to SchNetPack inputs and perform prediction
        inputs = converter(atoms)
        b = {k: v.to(device) for k, v in inputs.items()}
        results = best_model(b)
        results = {k: v.to('cpu') for k, v in results.items()}

        # energy and forces mae
        e_mae = torch.abs(results['energy'] - structure['energy'])
        e_maes.append(e_mae.item())
        f_mae = torch.mean(torch.abs((results['forces'] - structure['forces'])))
        f_maes.append(f_mae.item())
    
    #print(f_maes)
    import json
    import numpy as np
    maes_dict = {'energy_maes': e_maes,
                 'energy_mean': np.mean(e_maes),
                 'energy_std': np.std(e_maes),
                 'forces_maes': f_maes,
                 'forces_mean': np.mean(f_maes),
                 'forces_std': np.std(f_maes)}
    with open(os.path.join(path_to_data_dir, molecule + '_ef_val_maes.json'), 'w') as maes_file:
        json.dump(maes_dict, maes_file, indent=4)


    return maes_dict


def predict(model,
            path_to_data_dir,
            molecule='ethanol',
            sample_idx=0,
            ref_stats={'energy_mean': 0.0,
                       'energy_std': 1.0,
                       'forces_mean': 0.0,
                       'forces_std': 1.0}):
    """ Predicts forces of a test set sample of the specified PaiNN model on MD17.

        Args:
            model: path to where the model is stored or model object.
            path_to_data_dir: where the data is stored.
            molecule: (default: 'ethanol').
            sample_idx: (default: 0)
            ref_stats: dictionary with expected forces mae and standard deviation.

        Returns:
            predicted energy and forces for sample with index `sample_idx`
            confidence measure
    """
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    if type(model) is str:
        model_path = os.path.join(model)
        best_model = torch.load(model_path, map_location=device)
    else:
        best_model = model
    best_model.eval()

    # remove add offsets postprocessor [ONLY REMOVE WHEN MEASURING ERRORS]
    nnp = [module for module in best_model.modules() if isinstance(module, spk.model.NeuralNetworkPotential)][0]
    nnp.postprocessors.__delattr__('1')

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
    
    # set up converter
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32, device=device
    )

    structure = dataset.test_dataset[sample_idx]
    # create atoms object from dataset
    atoms = Atoms(
        numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
    )

    # convert atoms to SchNetPack inputs and perform prediction
    inputs = converter(atoms)
    b = {k: v.to(device) for k, v in inputs.items()}
    results = best_model(b)
    results = {k: v.to('cpu') for k, v in results.items()}

    # forces mae + confidence measure (:= how many standard deviations from expected error)
    f_mae = torch.mean(torch.abs((results['forces'] - structure['forces']))).item()

    forces_cm = (f_mae - ref_stats['forces_mean']) / (ref_stats['forces_std'] + 1e-8)  # +eps for numerical stability


    return results, forces_cm

