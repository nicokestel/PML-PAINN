import sys
from experiments import md17_ef
from inference import md17_ef_inference


def train_apply(method='painn',
                dataset='md17',
                molecule='ethanol',
                train_on_forces_only=False,
                params=None):
    """Loads `dataset`, trains the algorithm in `method` and
    returns predictions on test set.

    Args:
        method: which algorithm/architecture to train (default: painn). Only `painn`supported for now.
        dataset: which dataset to train on and predict test labels (default: md17)
        molecule: when dataset='md17', decides which on which molecule data to load (default: ethanol).
        train_on_forces_only: when dataset='md17', decides if method is only trained on MD17 forces or energies and forces (default: False).

    Returns:
        predictions on test set.
    """

    if dataset == 'md17':
        # loads data
        # trains algorithm
        model = md17_ef.run(molecule=molecule,
                            train_on_forces_only=train_on_forces_only)

        # return predictions on test set
        test_pred = md17_ef_inference.predict(model=model,
                                              path_to_data_dir='./md17_ef')
        return test_pred
    else:
        print(f'Dataset {dataset} not supported.')


if __name__ == '__main__':

    dataset, molecule, train_on_forces_only = sys.argv[1], sys.argv[2], bool(
        sys.argv[3].lower() in ['true', '1'])

    pred = train_apply(dataset=dataset,
                       molecule=molecule,
                       train_on_forces_only=train_on_forces_only)
