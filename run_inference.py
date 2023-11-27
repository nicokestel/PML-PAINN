"""
Examplary calls:

python run_inference.py PML-PAINN/md17_ef/best_inference_model_aspirin PML-PAINN/md17_ef/ aspirin
    - starts testing the PaiNN model on MD17 energies and forces for the Aspirin molecule.

python run_inference.py PML-PAINN/mlp_baseline/mlp PML-PAINN/md17_ef/ aspirin
    - starts testing the MLP baseline model on a custom regression task for MD17 forces for the Aspirin molecule.
"""
import sys
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
        path_to_model = sys.argv[1]
        path_to_data_dir = sys.argv[2]
        molecule = sys.argv[3] if len(sys.argv) >= 4 else 'ethanol'
        md17_ef_inference.run(
            path_to_model, path_to_data_dir, molecule=molecule)
