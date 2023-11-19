import sys
from experiments import *


SUPPORTED_EXPERIMENTS = ['qm9_atomwise', 'qm9_dipole', 'md17']


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('[ERROR] Provide experiment to execute! Choose one of {}'.format(SUPPORTED_EXPERIMENTS))
        sys.exit(1)

    expmt = globals()[sys.argv[1]]
    expmt.run()
