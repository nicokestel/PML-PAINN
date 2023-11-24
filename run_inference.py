import sys
from inference import md17_ef_inference


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('USAGE: python run_inference.py <path-to-model> <path-to-data-dir> [molecule]')
        sys.exit(1)

    path_to_model = sys.argv[1]
    path_to_data_dir = sys.argv[2]
    molecule = sys.argv[3] if len(sys.argv) >= 4 else 'ethanol'
    md17_ef_inference.run(path_to_model, path_to_data_dir, molecule=molecule)
