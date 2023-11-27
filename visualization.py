import numpy as np
import matplotlib.pyplot as plt


from ase.visualize.plot import plot_atoms
from ase.db import connect
from schnetpack.datasets import AtomsDataModule
from schnetpack.transform import ASENeighborList
import schnetpack.properties as structure


def show_molecule(dataset, data_id=1, outfile=None, title=None, xlabel=None, ylabel=None, **kwargs):

    # plt.grid(True)
    atoms = connect(dataset.datapath).get(data_id).toatoms()
    plot_atoms(atoms)

    # Set the title and labels.
    if title:
        plt.title(atoms.symbols)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    # Hide the x and y axes.
    plt.axis('off')

    # Save the figure to a file if specified.
    if outfile is not None:
        plt.savefig(outfile, format='svg')
    else:
        plt.show()




def explore_dataset(dataset, batch_size, num_train, num_val, num_test, transforms=None):
    """
    Explore a molecular dataset using SchNetPack's AtomsDataModule.
    Adapted from the SchNetPack tutorial: https://schnetpack.readthedocs.io/en/latest/tutorials/tutorial_01_preparing_data.html


    This function creates an instance of the AtomsDataModule for the given dataset, prepares and sets up the data module,
    and prints statistics about the dataset, including the number of reference calculations and available properties.
    It also loads and prints an example data point from the dataset.

    Parameters:
    - dataset: The SchNetPack dataset class (e.g., QM9).
    - batch_size: Batch size for the data module.
    - num_train: Number of training samples.
    - num_val: Number of validation samples.
    - num_test: Number of test samples.
    - transforms: List of transformations to be applied to the dataset.

    Example usage for QM9 dataset:
    explore_dataset(QM9, batch_size=10, num_train=110000, num_val=10000, num_test=13885, transforms=[ASENeighborList(cutoff=5.)])

    """
    # Create an instance of the AtomsDataModule for the given dataset
    data_module = AtomsDataModule(
        dataset=dataset,
        batch_size=batch_size,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        transforms=transforms
    )

    # Prepare and set up the data module
    data_module.prepare_data()
    data_module.setup()

    # Print dataset statistics
    print(f'Number of reference calculations: {len(data_module.dataset)}')
    print(f'Number of train data: {len(data_module.train_dataset)}')
    print(f'Number of validation data: {len(data_module.val_dataset)}')
    print(f'Number of test data: {len(data_module.test_dataset)}')

    print('Available properties:')
    for prop in data_module.dataset.available_properties:
        print('-', prop)

    # Load an example data point
    example = data_module.dataset[0]
    print('\nProperties:')
    for key, value in example.items():
        print('-', key, ':', value.shape)

