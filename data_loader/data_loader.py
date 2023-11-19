import os
import sys

from schnetpack.datasets.qm9 import QM9


SUPPORTED_DATASETS = ['qm9', 'md17']


def load_data(dataset,
              transformations=None,
              n_train=None,
              n_val=None,
              batch_size=100,
              work_dir=None):
    
    if dataset.lower() == 'qm9':
        return __load_qm9_data(transformations=transformations,
                               n_train=n_train,
                               n_val=n_val,
                               batch_size=batch_size,
                               work_dir=work_dir)
    
    elif dataset.lower() == 'md17':
        pass

    else:
        print('[ERROR] Dataset \'{}\' not supported! Choose one of {}.'.format(dataset, SUPPORTED_DATASETS))
        sys.exit(1)


def __load_qm9_data(transformations=None,
                    n_train=None,
                    n_val=None,
                    batch_size=100,
                    work_dir=None):
    
    # prepare working directory
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # QM9 data
    os.system('rm ' + os.path.join(work_dir, "split.npz"))
    qm9data = QM9(
        './qm9.db',
        batch_size=batch_size,
        num_train=n_train,
        num_val=n_val,
        transforms=transformations,
        property_units={QM9.U0: 'eV', QM9.mu: 'Debye'},
        num_workers=1,
        split_file=os.path.join(work_dir, "split.npz"),
        pin_memory=False, # set to false, when not using a GPU
        load_properties=[QM9.U0, QM9.mu], #only load U0 property, i.e. inner energy at 0K
    )
    qm9data.prepare_data()
    qm9data.setup()

    return qm9data