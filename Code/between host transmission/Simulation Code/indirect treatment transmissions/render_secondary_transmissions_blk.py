import numpy as np
from scipy.integrate import solve_ivp
from numba import jit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
matplotlib.rcParams['axes.labelsize']=24
matplotlib.rcParams['xtick.labelsize']=16
matplotlib.rcParams['ytick.labelsize']=16
matplotlib.rcParams['font.sans-serif']="Arial"
matplotlib.rcParams['font.family']="sans-serif"

WT_untreated_data = '20210529_new_recalibration-txset00.txt'
WT_treated_data = '20210529_new_recalibration_parameter_sweep_list-WTTIP_WT-txset00-rho1-5_psi0-02.txt'
TIP_data = '20210529_new_recalibration_parameter_sweep_list-WTTIP_TIP-txset00-rho1-5_psi0-02.txt'

df_untreat=pd.read_csv(WT_untreated_data)
df_treat=pd.read_csv(WT_treated_data)
df_TIP=pd.read_csv(TIP_data)

plt.figure(figsize=(7*0.65, 6*0.65))
WT_only_color = 'black'#'#333399'
WT_TIP_color = '#CC3399'

plt.hist(df_untreat['R0'].values, color=WT_only_color,
         density=True, bins=np.arange(0, 10),histtype='step',
         linewidth=3)
plt.hist(df_treat['R0'].values, color=WT_TIP_color,
         density=True, bins=np.arange(0, 10),histtype='step',
         linewidth=3)
plt.hist(df_TIP['R0'].values, color=WT_TIP_color,
         density=True, bins=np.arange(0, 10),histtype='step',
         linewidth=3 , linestyle='dotted')


ax=plt.gca()
plt.xlabel('Secondary transmissions')
plt.ylabel('Fraction')
plt.tight_layout(pad=1.2)
plt.xticks((0, 2, 4, 6, 8, 10))
plt.xlim(0, 10)
plt.yticks((0, 0.5, 1.0))
plt.ylim(0, 1)

figurename = 'secondary_tx_indirect_treatment.png'
figurename_pdf = 'secondary_tx_indirect_treatment.pdf'

plt.savefig(figurename, dpi=300)
plt.savefig(figurename_pdf, dpi=300, transparent=True)
