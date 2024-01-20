import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import json
from statistics import NormalDist
import sys
import scipy.stats as stats

MOLECULES = ['Aspirin', 'Ethanol', 'Naphthalene', 'Salicylic Acid', 'Toluene', 'Uracil']


def read_stats(ref_stat_file):
    with open(ref_stat_file, 'r') as fp:
        ref_stats = json.load(fp)
    return ref_stats


data_dir = './md17_ef/'

ref_stat_files = glob.glob(os.path.join(data_dir, '*val_maes.json'))
print(f'{len(ref_stat_files)} reference statistics found!')
ref_stat_files=sorted(ref_stat_files)
if len(ref_stat_files) == 0:
    sys.exit(1)


plt.figure(figsize=(13, 7))
for i, ref_stat_file in enumerate(ref_stat_files):
    ref_stats = read_stats(ref_stat_file)
    f_maes = ref_stats['forces_maes']
    #f_mes -= np.mean(f_mes)
    #f_mes /= np.std(f_mes) + 1e-8
    #f_mes = (np.array(ref_stats['forces_mes']) - ref_stats['forces_mean']) / (ref_stats['forces_std'] + 1e-8)
    #f_mes = ref_stats['forces_mes']
    npdf = NormalDist.from_samples(ref_stats['forces_maes'])

    plt.subplot(2, 3, i+1)
    plt.title(MOLECULES[i])
    if i % 3 == 0:
        plt.ylabel('Density')
    if i == 4:
        plt.xlabel('forces MAEs on validation sets')

    x = np.linspace(npdf.mean - 4*npdf.stdev, npdf.mean + 4*npdf.stdev, 100)
    plt.hist(f_maes, density=True, label=f'$\mu={np.mean(f_maes):.5f}$\n$\sigma={np.std(f_maes):.5f}$', color='#163B4E', rwidth=.9)
    plt.plot(x, stats.norm.pdf(x, npdf.mean, npdf.stdev), color='#E0833D')
    plt.axvline(-1.96*np.std(f_maes) + np.mean(f_maes), color='#E0833D', linestyle='--', linewidth=1)
    plt.axvline(1.96*np.std(f_maes) + np.mean(f_maes), color='#E0833D', linestyle='--', linewidth=1, label='$\mu \pm 1.96 \sigma$')
    plt.legend(loc='upper right')
plt.show()

#plt.savefig('visuals/MAE_Distribution.svg', dpi=300, bbox_inches='tight')





