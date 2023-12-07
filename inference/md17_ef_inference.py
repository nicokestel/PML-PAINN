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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model_path = os.path.join(path_to_model)
    best_model = torch.load(model_path, map_location=device)
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
    split_metrics   = {'energy': 0.0, 'forces': 0.0}
    metrics         = {'energy': 0.0, 'forces': 0.0}  # averaged over splits

    n_splits = 3
    n_test_batches_per_split = len(dataset.test_dataset) // bs // n_splits
    print('length   test set:', len(dataset.test_dataset))
    print('batches per split:', n_test_batches_per_split)
    
    for i, batch in enumerate(dataset.test_dataloader(), 1):

        if i % 100 == 0:
            print('batch', i)
        
        b = {k: v.to(device) for k, v in batch.items()}
        results = best_model(b)
        results = {k: v.to('cpu') for k, v in results.items()}

        # Convert units [DON'T!! WE REPORT IN KCAL/MOL]
        #results['energy'] *= convert_units("kcal/mol", "eV")
        #results['forces'] *= convert_units("kcal/mol", "eV")
        
        # MAE
        split_metrics['energy'] += F.l1_loss(results['energy'], batch['energy'])
        split_metrics['forces'] += F.l1_loss(results['forces'], batch['forces'])

        if i % n_test_batches_per_split == 0:
            metrics['energy'] += split_metrics['energy'] / n_test_batches_per_split
            metrics['forces'] += split_metrics['forces'] / n_test_batches_per_split

    metrics['energy'] /= n_splits
    metrics['forces'] /= n_splits

    print('MAE\nenergy in [kcal/mol] and forces in [kcal/mol/Angstrom]')
    print(metrics)
