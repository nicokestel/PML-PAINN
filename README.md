# PML-PAINN

##### Examplary cli calls for training
- `python run_training.py md17_ef uracil`
  starts training and validating the PaiNN model on MD17 uracil energies and forces.

- `python run_training.py md17_ef uracil`
  starts training and validating the PaiNN model on MD17 uracil energies and forces.

- `python run_training.py md17_ef_ablation 4`
  starts training and validating the ablated PaiNN model on MD17 aspirin energies and forces.
  `ablation level 0`: no ablation.
  `ablation level 1`: no scalar product of vector features in update block.
  `ablation level 2`: no vector propagation in message block.
  `ablation level 3`: level 1 and 2 combined.
  `ablation level 4`: no vector features.

  
##### Examplary cli calls for testing
- `python run_inference.py PML-PAINN/md17_ef/best_inference_model_aspirin PML-PAINN/md17_ef/ aspirin`
  starts testing the PaiNN model on MD17 aspirin energies and forces.

- `python run_inference.py PML-PAINN/md17_ef/best_inference_model_aspirin PML-PAINN/md17_ef/ all`
  starts testing all PaiNN models on MD17 aspirin, ethanol, naphthalene, salicylic acid, toluene and uracil energies and forces.

- `python run_inference.py PML-PAINN/md17_ef/ablation_lv4_model_aspirin PML-PAINN/md17_ef/ aspirin`
  starts testing the PaiNN model without vector features on MD17 aspirin energies and forces.


##### Data Storage
The latest version of data that we used for training and testing can be found at `/home/space/datasets/md17_ef/`
