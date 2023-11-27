# PML-PAINN

##### Examplary cli calls for training

- `python run_training.py baseline aspirin`
  starts training and validating the MLP baseline model on the Aspirin molecule from the MD17 dataset.

- `python run_training.py md_17_ef uracil`
  starts training and validating the PaiNN model on MD17 energies and forces for the Uracil molecule.

- `python run_training.py qm9 energy_U0`
  starts training and validating the PaiNN model on QM9 for predicting the QM9.U0 property.

  
##### Examplary cli calls for testing
- `python run_inference.py PML-PAINN/md17_ef/best_inference_model_aspirin PML-PAINN/md17_ef/ aspirin`
  starts testing the PaiNN model on MD17 energies and forces for the Aspirin molecule.

- `python run_inference.py PML-PAINN/mlp_baseline/mlp PML-PAINN/md17_ef/ aspirin`
  starts testing the MLP baseline model on a custom regression task for MD17 forces for the Aspirin molecule.


##### Data Storage
The latest version of data that we used for training and testing can be found at `/home/space/datasets/qm9/` and `/home/space/datasets/md17_ef/`
