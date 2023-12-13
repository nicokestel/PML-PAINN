"""
Examplary calls:

python run_training.py baseline aspirin
    - starts training and validating the MLP baseline model on the Aspirin molecule from the MD17 dataset.

python run_training.py md_17_ef uracil
    - starts training and validating the PaiNN model on MD17 energies and forces for the Uracil molecule.

python run_training.py qm9 energy_U0
    - starts training and validating the PaiNN model on QM9 for predicting the QM9.U0 property.
"""
import sys
from experiments import *


SUPPORTED_EXPERIMENTS = ['qm9', 'md17_f', 'md17_ef', 'baseline']


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('[ERROR] Provide experiment to execute! Choose one of {}'.format(SUPPORTED_EXPERIMENTS))
        sys.exit(1)

    expmt = sys.argv[1]
    if expmt == 'md17_ef':
        molecule = sys.argv[2] if len(sys.argv) >= 3 else 'ethanol'
        md17_ef.run(molecule)
    elif expmt == 'md17_f':
        molecule = sys.argv[2] if len(sys.argv) >= 3 else 'ethanol'
        md17_ef.run(molecule, train_on_forces_only=True)
    elif expmt == 'qm9':
        prop = sys.argv[2] if len(sys.argv) >= 3 else 'energy_U0'
        qm9.run(prop)
    elif expmt == 'baseline':
        molecule = sys.argv[2] if len(sys.argv) >= 3 else 'ethanol'
        baseline.run(molecule)
    else:
        print('[ERROR] experiment <{}> not supported! Choose one of {}'.format(expmt, SUPPORTED_EXPERIMENTS))

