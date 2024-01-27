import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
from scipy import stats

MOLECULES = np.array(['aspirin', 'ethanol', 'naphthalene', 'salicylic_acid', 'toluene', 'uracil'])
MAES = np.array([0.359, 0.170, 0.234, 0.192, 0.089, 0.131])
MAES_ORG = np.array([0.371, 0.230, 0.083, 0.209, 0.102, 0.140])
MAES_KERNEL = np.array([0.478, 0.136, 0.151, 0.221, 0.203, 0.105])
NATOMS_FILE = 'md17_ef/n_atoms.json'

if __name__ == '__main__':

    with open(NATOMS_FILE, 'r') as fp:
        n_atoms_dict = json.load(fp)
        n_atoms = np.array(list(n_atoms_dict.values()))

    # sort values
    sort = np.argsort(n_atoms)
    n_atoms = n_atoms[sort]
    MAES = MAES[sort]
    MAES_ORG = MAES_ORG[sort]
    MAES_KERNEL = MAES_KERNEL[sort]
    MOLECULES = MOLECULES[sort]

    for mol, mae, natoms in zip(MOLECULES, MAES, n_atoms):
        print(mol, mae, natoms)

    plt.figure(figsize=(9, 5))
    plt.plot(n_atoms, MAES, label='Our MS2 results', c='#163B4E')
    plt.plot(n_atoms, MAES_ORG, label='Schütt et al. (2021)', c='#163B4E', linestyle='--')
    plt.plot(n_atoms, MAES_KERNEL, label='FCHL19', c='#163B4E', linestyle=':')


    plt.xticks(n_atoms, [mol.replace('_', ' ').title() + f' ({n})' for mol, n in zip(MOLECULES, n_atoms)], rotation=45)

    plt.title('force MAE vs Number of atoms')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Molecule (#atoms)')

    plt.legend()
    plt.grid()
    #plt.show()

    plt.savefig('visuals/mae_natoms.svg', dpi=300, bbox_inches='tight')
    
    print(np.corrcoef(n_atoms, MAES))
    print(np.corrcoef(n_atoms, MAES_ORG))

    print(stats.spearmanr(n_atoms, MAES_ORG))  # -> high p suggests MAEs and N_ATOMS are samples from independent distributions
                                               # -> Alt. exp: simplicity of molecule structures: naphthalene simple structure, aspirin complex structure, ethanol less atoms but more complex than naphthalene
                                               # -> supporting claim that PaiNN scales to larger molecules
    print(stats.spearmanr(n_atoms, MAES_KERNEL))