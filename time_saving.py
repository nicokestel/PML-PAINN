# for each molecule
#   for t in human_time
#       for s in testsample
#           _, cm = model(s)
#           if not cm in 95/90/80%
#               human(s) -> add t to total time
#           else
#               pass
#       -> time/samples = total time / #testsamples
#       -> relative time saved = time/samples / human_time*#testsamples
#       store in json

import glob
import sys
import os
from data_loader import load_data
from schnetpack.datasets.md17 import MD17
import schnetpack.transform as trn
import torch
import json
from inference import md17_ef_inference
import matplotlib.pyplot as plt
import numpy as np

def calc_time_savings(molecules=['aspirin'],
                      n_testsamples=100,
                      ref_stats=None,
                      work_dir='./md17_ef'):

    results = {mol: {'0.95': 0.0, '0.90': 0.0, '0.80': 0.0} for mol in molecules}

    for molecule in molecules:
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
                        work_dir=work_dir)
        
        # load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.load(os.path.join(work_dir, f'best_inference_model_{molecule}'), map_location=device)
        model.eval()

        # load reference stats
        with open(os.path.join(work_dir, molecule+'_ef_val_maes.json'), 'r') as fp:
            ref_stats = json.load(fp)

        for i in range(n_testsamples):
            _, cm = md17_ef_inference.predict(model, dataset, molecule=molecule, sample_idx=i, ref_stats=ref_stats)
            if abs(cm) > 1.28:  # 80% considered correct
                results[molecule]['0.80'] += 1
            if abs(cm) > 1.645:  # 90% considered correct
                results[molecule]['0.90'] += 1
            if abs(cm) > 1.96:  # 95% considered correct
                results[molecule]['0.95'] += 1

        results[molecule]['0.80'] = 1 - (results[molecule]['0.80'] / n_testsamples)
        results[molecule]['0.90'] = 1 - (results[molecule]['0.90'] / n_testsamples)
        results[molecule]['0.95'] = 1 - (results[molecule]['0.95'] / n_testsamples)

    return results
            
            
MOLECULES = ['aspirin', 'ethanol', 'naphthalene', 'salicylic_acid', 'toluene', 'uracil']

if __name__ == '__main__':

    if sys.argv[1] == 'calc':
        results = calc_time_savings(molecules=MOLECULES, n_testsamples=10000)

        with open('md17_ef/time_saving.json', 'w') as fp:
            json.dump(results, fp, indent=4)
        print(results)
    elif sys.argv[1] == 'plot':
        with open('md17_ef/time_saving.json', 'r') as fp:
            results = json.load(fp)
        
        plt.figure(figsize=(7, 7))
        plt.title('Empirical vs Analytical Time-Savings')
        plt.xticks([0.80, 0.90, 0.95], [80, 90, 95])
        plt.yticks([0.80, 0.90, 0.95], [80, 90, 95])
        plt.gca().set_aspect('equal')
        for v in [0.80, 0.90, 0.95]:
            plt.axvline(v, c='#E0833D' if v==0.95 else 'gray', linestyle='--')
            plt.axhline(v, c='#E0833D' if v==0.95 else 'gray', linestyle='--')
        for mol in ['ethanol', 'uracil', 'toluene', 'naphthalene', 'aspirin', 'salicylic_acid']:
            mol_dict = results[mol]
            analytical_ts = [float(ts) for ts in list(mol_dict.keys())]
            empirical_ts = [float(ts) for ts in list(mol_dict.values())]

            print(analytical_ts, empirical_ts)

            plt.plot(analytical_ts, empirical_ts, label=mol, color='#163B4E')

        plt.xlabel('Analytical time-saving [%]')
        plt.ylabel('Empirical time-saving [%]')
        plt.legend()
        #plt.show()
        plt.savefig('visuals/time_saving.svg', dpi=300, bbox_inches='tight')