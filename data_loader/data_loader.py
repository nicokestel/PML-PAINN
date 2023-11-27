import os
import sys

from schnetpack.datasets.qm9 import QM9
from schnetpack.datasets.md17 import MD17


SUPPORTED_DATASETS = ['qm9', 'md17']


def load_data(dataset,
              molecule='ethanol',
              transformations=None,
              n_train=None,
              n_val=None,
              batch_size=100,
              work_dir=None):
    """Loads the specified dataset from SchNetPack.

    Args:
        dataset: dataset to load ('qm9' or 'md17').
        molecule: if dataset='md17', specifies which molecule structures to load (default: 'ethanol').
        transformations: list of transformation to apply to each sample in the dataset (default: None).
        n_train: number of samples in train split (default: None).
        n_val: number of samples in validation split (default: None).
        batch_size: number of samples per batch (default: 10).
        work_dir: the directory to store the dataset in (default: 'None).

    Returns:
        the dataset object
    """

    if dataset.lower() == 'qm9':
        return __load_qm9_data(transformations=transformations,
                               n_train=n_train,
                               n_val=n_val,
                               batch_size=batch_size,
                               work_dir=work_dir)

    elif dataset.lower() == 'md17':
        return __load_md17_data(molecule=molecule,
                                transformations=transformations,
                                n_train=n_train,
                                n_val=n_val,
                                batch_size=batch_size,
                                work_dir=work_dir)

    else:
        print('[ERROR] Dataset \'{}\' not supported! Choose one of {}.'.format(dataset, SUPPORTED_DATASETS))
        sys.exit(1)


def __load_qm9_data(transformations=None,
                    n_train=None,
                    n_val=None,
                    batch_size=100,
                    work_dir=None):
    """Loads the QM9 dataset from SchNetPack.

    Args:
        transformations: list of transformation to apply to each sample in the dataset (default: None).
        n_train: number of samples in train split (default: None).
        n_val: number of samples in validation split (default: None).
        batch_size: number of samples per batch (default: 100).
        work_dir: the directory to store the dataset in (default: 'None).

    Returns:
        QM9 dataset object
    """

    # prepare working directory
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # QM9 data
    # os.system('rm ' + os.path.join(work_dir, "split.npz"))
    qm9data = QM9(
        os.path.join(work_dir, 'qm9.db'),
        batch_size=batch_size,
        num_train=n_train,
        num_val=n_val,
        transforms=transformations,
        property_units={
            QM9.mu: "Debye",
            #QM9.alpha: "Angstrom Angstrom Angstrom",
            QM9.homo: "eV",
            QM9.lumo: "eV",
            QM9.gap: "eV",
            #QM9.r2: "Angstrom Angstrom",
            QM9.zpve: "eV",
            QM9.U0: "eV",
            QM9.U: "eV",
            QM9.H: "eV",
            QM9.G: "eV",
            #QM9.Cv: "cal/mol/K"
        },
        remove_uncharacterized=False,
        num_workers=0,
        split_file=os.path.join(work_dir, "split.npz"),
        pin_memory=True # set to false, when not using a GPU
        # load_properties=[QM9.U0, QM9.mu], #only load U0 property, i.e. inner energy at 0K
    )
    qm9data.prepare_data()
    qm9data.setup()

    return qm9data


def __load_md17_data(molecule='ethanol',
                     transformations=None,
                     n_train=None,
                     n_val=None,
                     batch_size=10,
                     work_dir=None):
    """Loads the MD17 dataset from SchNetPack.

    Args:
        molecule: molecule structures to load (default: 'ethanol').
        transformations: list of transformation to apply to each sample in the dataset (default: None).
        n_train: number of samples in train split (default: None).
        n_val: number of samples in validation split (default: None).
        batch_size: number of samples per batch (default: 100).
        work_dir: the directory to store the dataset in (default: 'None).

    Returns:
        MD17 dataset object
    """

     # prepare working directory
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # MD17 data
    # os.system('rm ' + os.path.join(work_dir, molecule + "_split.npz"))
    md17data = MD17(
        os.path.join(work_dir, molecule + '.db'),
        molecule=molecule,
        batch_size=batch_size,
        num_train=n_train,
        num_val=n_val,
        property_units={MD17.energy:"kcal/mol", MD17.forces:"kcal/mol/Ang"},
        transforms=transformations,
        num_workers=3,
        split_file=os.path.join(work_dir, molecule + "_split.npz"),
        pin_memory=True # set to false, when not using a GPU
        # load_properties=[MD17.energy, MD17.forces]
    )
    md17data.prepare_data()
    md17data.setup()

    return md17data
