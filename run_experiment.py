import sys
from experiments import *


SUPPORTED_EXPERIMENTS = ['qm9_atomwise', 'qm9_dipole', 'md17_ef']


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('[ERROR] Provide experiment to execute! Choose one of {}'.format(SUPPORTED_EXPERIMENTS))
        sys.exit(1)

    expmt = sys.argv[1]
    if expmt == 'md17_ef':
        molecule = sys.argv[2] if len(sys.argv) >= 3 else 'ethanol'
        expmt_scr = globals()[sys.argv[1]]
        expmt_scr.run(molecule)
    elif expmt.startsWith('qm9'):
        expmt_scr = globals()[sys.argv[1]]
        expmt_scr.run()
