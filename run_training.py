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
        prop = sys.argv[2] or 'U0'
        if hasattr(QM9, prop):
            if prop != 'mu' and prop != 'r2':
                # atomwise
                qm9_atomwise.run(prop)
            else:
                # not atomwise
                if prop == 'mu':
                    qm9_dipole.run(prop)
                elif prop == 'r2':
                    print('QM9 r2')
        else:
            print('[ERROR] QM9 property <{}> not supported!'.format(expmt))
    else:
        print('[ERROR] experiment <{}> not supported! Choose one of {}'.format(expmt, SUPPORTED_EXPERIMENTS))

