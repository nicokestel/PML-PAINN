import sys
from experiments import *


SUPPORTED_EXPERIMENTS = ['qm9', 'md17_ef']


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('[ERROR] Provide experiment to execute! Choose one of {}'.format(SUPPORTED_EXPERIMENTS))
        sys.exit(1)

    expmt = sys.argv[1]
    if expmt == 'md17_ef':
        molecule = sys.argv[2] if len(sys.argv) >= 3 else 'ethanol'
        md17_ef.run(molecule)
    elif expmt == 'qm9':
        prop = sys.argv[2] if len(sys.argv) >= 3 else 'energy_U0'
        qm9.run(prop)
    else:
        print('[ERROR] experiment <{}> not supported! Choose one of {}'.format(expmt, SUPPORTED_EXPERIMENTS))

