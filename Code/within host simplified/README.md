# within host simplified

Relevant to Figure S1G.

We used a less-complex within host model of SARS-CoV-2 and TIP dynamics, ignoring spatial compartmentalization of the respiratory tract,  assuming monotonically decreasing viral load after the peak (i.e. no spread of the infection to new cells), and assuming simplified first-order rate kinetics for viral clearance.

All code, figures, and results are present under the `Simulation Code` folder. Run `within_host_model.py` to generate the panels for Fig. S1G.

A few files of note:
- `model_validation.pdf/.png` to confirm that our model matches the patient data.
- `Singapore 17 result.pdf/.png` matches one of the panels in Figure S1G
- `TIP_effect_psi0-02_rho1-5.pdf/.png` shows individual responses to TIP treatment across other patient-calibrated simulations
- `TIP_heatmap_effect_heatmap.pdf/.png` provides the zoomed-out heatmap in Figure S1G.
- `TIP_heatmap_effect_zoom_upper_right_heatmap.pdf/.png` provides the zoomed-in heatmap in Figure S1G.
- The `_AUC` versions of the figures provide the integrated area under the curve (rather than peak viral load).

Dependencies are:
- python3 (3.7.3)
- numpy (1.19.4)
- scipy (1.5.4)
- numba (0.51.2)
- matplotlib (3.3.3)
- pandas (1.1.4)
- seaborn (0.11.0)
