"""
Examplary calls:

python run_predict.py md17_ef/best_inference_model_aspirin md17_ef/ aspirin 0
    - predicts MAE of first aspirin test sample 
"""
import sys
import os
import glob
import json
from inference import md17_ef_inference


if __name__ == '__main__':

    if len(sys.argv) < 5:
        print(
            'USAGE: python run_predict.py <path-to-model> <path-to-data-dir> <molecule> <sample_idx>')
        sys.exit(1)


    #molecules = [sys.argv[3]] if len(sys.argv) >= 4 else ['ethanol']
    #if molecules == ['all']:
    #    molecules = ['aspirin', 'ethanol', 'naphthalene', 'salicylic_acid', 'toluene', 'uracil']
    #    models = glob.glob('md17_ef/best_*')
    #else:
    #   models = [sys.argv[1]]
        
    path_to_model = sys.argv[1]
    path_to_data_dir = sys.argv[2]
    molecule = sys.argv[3]
    sample_idx = int(sys.argv[4])

    
    print('predicting {} sample {} with {}.'.format(molecule, sample_idx, path_to_model))

    # load reference stats
    with open(os.path.join(path_to_data_dir, molecule + '_ef_val_maes.json'), 'r') as fp:
        ref_stats = json.load(fp)

    pred, cm = md17_ef_inference.predict(
        path_to_model, path_to_data_dir, molecule=molecule, ref_stats=ref_stats)
    
    print(pred)
    print(f'difference of prediction to expected error: {cm} std.')


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