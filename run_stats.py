"""
Examplary calls:

python run_inference.py PML-PAINN/md17_ef/best_inference_model_aspirin PML-PAINN/md17_ef/ aspirin
    - starts testing the PaiNN model on MD17 aspirin energies and forces.

python run_inference.py PML-PAINN/md17_ef/best_inference_model_aspirin PML-PAINN/md17_ef/ all
    - starts testing all PaiNN models on MD17 aspirin, ethanol, naphthalene, salicylic acid, toluene and uracil energies and forces.

python run_inference.py PML-PAINN/md17_ef/ablation_lv4_model_aspirin PML-PAINN/md17_ef/ aspirin
    - starts testing the PaiNN model without vector features on MD17 aspirin energies and forces.
"""
import sys
import os
import glob
import json
from inference import md17_ef_inference


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print(
            'USAGE: python run_stats.py <path-to-models> <path-to-data-dir>')
        sys.exit(1)


    #molecules = [sys.argv[3]] if len(sys.argv) >= 4 else ['ethanol']
    #if molecules == ['all']:
    #    molecules = ['aspirin', 'ethanol', 'naphthalene', 'salicylic_acid', 'toluene', 'uracil']
    #    models = glob.glob('md17_ef/best_*')
    #else:
    #   models = [sys.argv[1]]
        
    paths_to_models = glob.glob(os.path.join(sys.argv[1], 'best_*'))
    paths_to_models = [ptm for ptm in paths_to_models if ptm[-1] != 'f']  # remove models trained on forces only
    path_to_data_dir = sys.argv[2]
    molecules = ['aspirin', 'ethanol', 'naphthalene', 'salicylic_acid', 'toluene', 'uracil'] #sys.argv[3]
    #sample_idx = int(sys.argv[4])

    for molecule in molecules:
        for path_to_model in paths_to_models:
            if molecule in path_to_model:
                print('running evaluation on {} with {}.'.format(molecule, path_to_model))
            
                maes_dict = md17_ef_inference.val_mean_std(
                    path_to_model, path_to_data_dir, molecule=molecule)


        """import matplotlib.pyplot as plt
        from statistics import NormalDist
        import numpy as np
        import seaborn as sns

        f_maes = np.load('aspirin_forces_maes.npy')
        npdf = NormalDist.from_samples(f_maes)

        print(npdf.mean, npdf.stdev)
        print(f_maes.max(), f_maes.min(), f_maes.mean(), f_maes.std())
        #plt.hist(f_maes, bins=200, normed=True)
        sns.histplot(f_maes, kde=True)
        #plt.axvline(1.94 * f_maes.std())
        #plt.axvline(-1.94 * f_maes.std())
        plt.show()

        #print(eval)"""