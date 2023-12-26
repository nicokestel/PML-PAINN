"""
Examplary calls:

python run_inference.py PML-PAINN/md17_ef/best_inference_model_aspirin PML-PAINN/md17_ef/ aspirin
    - starts testing the PaiNN model on MD17 energies and forces for the Aspirin molecule.

python run_inference.py PML-PAINN/mlp_baseline/mlp PML-PAINN/md17_ef/ aspirin
    - starts testing the MLP baseline model on a custom regression task for MD17 forces for the Aspirin molecule.
"""
import sys
import os
import glob
import json
from inference import md17_ef_inference
from inference import baseline_inference


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print(
            'USAGE: python run_inference.py <path-to-model> <path-to-data-dir> [molecule]')
        sys.exit(1)

    if 'baseline' in sys.argv[1]:
        path_to_model = sys.argv[1]
        path_to_data_dir = sys.argv[2]
        molecule = sys.argv[3] if len(sys.argv) >= 4 else 'ethanol'
        baseline_inference.run(
            path_to_model, path_to_data_dir, molecule=molecule)

    else:
        molecules = [sys.argv[3]] if len(sys.argv) >= 4 else ['ethanol']
        if molecules == ['*']:
            models = glob.glob('best_*')
        else:
            models = [sys.argv[1]]
        path_to_data_dir = sys.argv[2]

        for molecule in molecules:
            for path_to_model in models:
                if molecule in path_to_model:

                    metrics = md17_ef_inference.run(
                        path_to_model, path_to_data_dir, molecule=molecule)
        
                    # save test metrics to `test_results.json`
                    test_results_file_path = os.path.join(path_to_data_dir, 'test_results.json')
                    if os.path.exists(test_results_file_path):
                        with open(test_results_file_path, 'r') as file:
                            test_results = json.load(file)

                        with open(test_results_file_path, 'w') as file:
                            model_name = path_to_model.split('/')[-1]
                            test_results[model_name] = metrics
                            file.write(json.dumps(test_results, indent=4))

                    else:
                        test_results = {}
                        with open(test_results_file_path, 'w') as file:
                            model_name = path_to_model.split('/')[-1]
                            test_results[model_name] = metrics
                            file.write(json.dumps(test_results, indent=4))
