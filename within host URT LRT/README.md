# within host URT LRT

Relevant to Figure 1C-D, Figure S1A-F.

We extended a model of SARS-CoV-2 dynamics in the upper and lower respiratory tracts (URT and LRT) (medRxiv, 2020.2009.2025.20201772) to include IP dynamics. Major assumptions in this model include (1) treating the adaptive immune response as an exponential increase in viral clearance at 2 weeks post infection, (2) accounting for spatial spread of infection to new target cells, so called "target cell extension".

All code, figures, and results are present under the `Simulation Code` folder. Files of note:
- `within_host_model_base.py` simulates and plots results for Fig. 1C-D, Fig. S1A-B, S1E-F.
- `within_host_model_base_100virion.py` simulates and plots results for Fig. S1C

The resultant .csv files from running the simulation code are present in the folder -- so it is possible to simply use the `...from_file...` function calls (and comment out the other function calls) within `main()` to render plots.
