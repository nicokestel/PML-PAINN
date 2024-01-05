"""
Examplary calls:

python run_training.py md_17_ef uracil
    - starts training and validating the PaiNN model on MD17 energies and forces for the Uracil molecule.

python run_training.py md17_ef_ablation 4
    - starts training and validating the ablated PaiNN model on MD17 aspirin energies and forces.
"""
import sys
from experiments import *


SUPPORTED_EXPERIMENTS = ['qm9', 'md17_f', 'md17_ef', 'baseline', 'md17_ef_ablation']


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('[ERROR] Provide experiment to execute! Choose one of {}'.format(SUPPORTED_EXPERIMENTS))
        sys.exit(1)

    expmt = sys.argv[1]
    if expmt == 'md17_ef':
        for molecule in sys.argv[2:]:
            print(f'running MD17 training on energies & forces for molecule={molecule}.')
            md17_ef.run(molecule)
    elif expmt == 'md17_f':
        for molecule in sys.argv[2:]:
            print(f'running MD17 training on forces for molecule={molecule}.')
            md17_ef.run(molecule, train_on_forces_only=True)
    elif expmt == 'qm9':
        prop = sys.argv[2] if len(sys.argv) >= 3 else 'energy_U0'
        qm9.run(prop)
    elif expmt == 'baseline':
        molecule = sys.argv[2] if len(sys.argv) >= 3 else 'ethanol'
        baseline.run(molecule)
    elif expmt == 'md17_ef_ablation':
        md17_ef_ablation.run(ablation_level=int(sys.argv[2]))
    else:
        print('[ERROR] experiment <{}> not supported! Choose one of {}'.format(expmt, SUPPORTED_EXPERIMENTS))

