# within host simplified

We combined the within-host dynamics model (see `within host URT LRT`) with a host-to-host transmission model (Goyal A et al. eLife 2021) wherein simulated individuals probabilistically encountered an index case, then had a transmission probability dependent on the index case's viral load.

The transmission probability is calculated from viral load using two parameters ($alpha$ = 10.18, $\lambda$ = 7.165) where were estimated based on combined measurements of patient viral load and culture probability (Jones TC et al. Science 2021).

We first recalibrated the model in the context of wildtype only (non-treated) transmission, then simulated the effect of either direct TIP treatment (high dosage) or indirect TIP treatment (low dosage). The results for a representative directly-treated individual are also provided.

Relevant to Figure S1G.

We used:
- python3 (3.7.3)
- numpy (1.19.4)
- scipy (1.5.4)
- numba (0.51.2)
- matplotlib (3.3.3)
- pandas (1.1.4)
- seaborn (0.11.0)
