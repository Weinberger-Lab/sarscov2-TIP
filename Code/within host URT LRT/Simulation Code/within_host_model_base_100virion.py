# 2021-2021 / Mike Pablo
# Simulate infection with wildtype SARS-CoV-2, with and without treatment by TIP (therapeutic interfering particle).
# This is based on an implementation of the 'Extended Target Cell' model in Ruian Ke et al. medRxiv 10.1101/2020.09.25.20201772v1
# 'Kinetics of SARS-CoV-2 infection in the human upper and lower respiratory tracts and their relationship with
# infectiousness'.
#
# Note that the TIP parameters xi and delta are set to 1 to have no effect, and they are not described in the manuscript.
# They were used to explore other theoretical mechanisms.
# ==============================================================================================================
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

def get_data():
    columnames = ['URT_time', 'URT_log10VL', 'LRT_time', 'LRT_log10VL']
    offset = 0.31  # careful inspection suggests a 0.31 shift in the log10VL.

    sourcedir = '/Users/mikepablo/Documents/GitHub/covid19-TIP/population-models/reevaluating within host params/Ke et al data extraction/'
    indA = pd.read_csv(sourcedir+'panelA_rounded_padded.csv', skiprows=[0], header=0, names=columnames)
    indB = pd.read_csv(sourcedir+'panelB_rounded_padded.csv', skiprows=[0], header=0, names=columnames)
    indC = pd.read_csv(sourcedir+'panelC_rounded_padded.csv', skiprows=[0], header=0, names=columnames)
    indD = pd.read_csv(sourcedir+'panelD_rounded_padded.csv', skiprows=[0], header=0, names=columnames)
    indE = pd.read_csv(sourcedir+'panelE_rounded_padded.csv', skiprows=[0], header=0, names=columnames)
    indF = pd.read_csv(sourcedir+'panelF_rounded_padded.csv', skiprows=[0], header=0, names=columnames)
    indG = pd.read_csv(sourcedir+'panelG_rounded_padded.csv', skiprows=[0], header=0, names=columnames)
    indH = pd.read_csv(sourcedir+'panelH_rounded_padded.csv', skiprows=[0], header=0, names=columnames)
    indI = pd.read_csv(sourcedir+'panelI_rounded_padded.csv', skiprows=[0], header=0, names=columnames)

    inds = [indA, indB, indC, indD, indE, indF, indG, indH, indI]

    for ind in inds:
        ind['URT_log10VL'] += offset
        ind['LRT_log10VL'] += offset

    return inds

def simulate_single_inf(params, t0, tf):
    """
    Solves the extended target cell model from Ke et al. over the specified time period, then
    returns simulated log10 throat swabs and sputum sample viral loads
    :param params: Parameters necessary for simulation, see below. Has more parameters than the input to single_inf_ODE
    :return: time and timeseries for log10VT and log10VS, the log10 virions/swab determined by the model.
    """

    beta_T, delta1, pi_T, beta_S, delta2, pi_S, t_tau, log10TN, w, T10, T20, VT0, c, k, gamma = params
    k1 = k
    k2 = k
    odeparams = (beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma)

    # Solve from t0 to tau
    first_Y0 = [T10, 0, 0, VT0, T20, 0, 0, 0]
    first_time = np.append(np.arange(t0, t_tau), t_tau)
    first_time = np.unique(first_time) # in case t_tau was an integer; will drop duplicated final point
    sol = solve_ivp(lambda t, y: single_inf_ODE(t, y, odeparams),
                    [first_time[0], first_time[-1]], first_Y0, t_eval=first_time)
    if not sol.success:
        print('Integration failed on first step, pre-target cell extension. params:', params)
        raise

    # Set up initial conditions for second step.
    second_Y0 = sol.y[:, -1]
    second_Y0[4] += 10**log10TN # target cell extension in LRT
    second_time = np.append(t_tau, np.arange(np.ceil(t_tau), tf))
    second_time = np.unique(second_time) # in case t_tau was an integer; will drop duplicated initial point
    sol2 = solve_ivp(lambda t, y: single_inf_ODE(t, y, odeparams),
                    [second_time[0], second_time[-1]], second_Y0, t_eval=second_time)

    # Stitch results together, keeping only the integer timepoints.
    all_times = np.append(sol.t, sol2.t)
    index_to_keep = np.equal(np.mod(all_times, 1), 0)
    all_sol = np.concatenate((sol.y, sol2.y), axis=1) # Append along time axis

    # Final results
    times = all_times[index_to_keep]
    solution = all_sol[:, index_to_keep]

    VT = solution[3]
    VS = solution[7]

    cutlog10 = lambda x: np.log10(x) if x > 0 else 0

    log10VT = [cutlog10(x) for x in VT]
    log10VS = [cutlog10(x) for x in VS]

    return times, log10VT, log10VS

def simulate_dual_inf(params, t0, tf):
    """

    Solves an adapted version of the extended target cell model from Ke et al. over the specified time period, then
    returns simulated log10 throat swabs and sputum sample viral loads.
    The adapted version expands the equations from 8 microbiological states to 16, where the eight new states are
    needed to describe TIP entry, coinfection, and TIP production.
    :param params: Parameters necessary for simulation, see below. Has more parameters than the input to single_inf_ODE
    :return: time and timeseries for log10VTWT, log10VSWT, log10VTTIP, and log10VSTIP, which are the log10 virions
    per sample detected.
    """

    beta_T, delta1, pi_T, beta_S, delta2, pi_S, t_tau, log10TN, w, T10, T20, VT0, c, k, gamma, rho, psi, xi, delta_TIP, fractionTIP = params
    k1 = k
    k2 = k
    odeparams = (beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, rho, psi, xi, delta_TIP)

    # Solve from t0 to tau
    # Initial condition: assume 1 cell is productively infected w/ WT virion, and pre-defined proportion of TIP-treated cells
    TIP_converted_naive_cells_URT = T10 * fractionTIP
    T10 = T10 - TIP_converted_naive_cells_URT
    T1TIP0 = TIP_converted_naive_cells_URT

    TIP_converted_naive_cells_LRT = T20 * fractionTIP
    T20 = T20 - TIP_converted_naive_cells_LRT
    T2TIP0 = TIP_converted_naive_cells_LRT
    first_Y0 = [T10, 0, 0, VT0, T1TIP0, 0, 0, 0, T20, 0, 0, 0, T2TIP0, 0, 0, 0]

    first_time = np.append(np.arange(t0, t_tau), t_tau)
    first_time = np.unique(first_time) # in case t_tau was an integer; will drop duplicated final point
    sol = solve_ivp(lambda t, y: dual_inf_ODE(t, y, odeparams),
                    [first_time[0], first_time[-1]], first_Y0, t_eval=first_time)
    if not sol.success:
        print('Integration failed on first step, pre-target cell extension. params:', params)
        raise

    # Set up initial conditions for second step.
    second_Y0 = sol.y[:, -1]
    TCE_LRT = 10**log10TN
    TCE_LRT_preconverted = TCE_LRT * fractionTIP
    TCE_LRT -= TCE_LRT_preconverted
    second_Y0[8] += TCE_LRT # target cell extension in LRT
    second_Y0[12] += TCE_LRT_preconverted

    second_time = np.append(t_tau, np.arange(np.ceil(t_tau), tf))
    second_time = np.unique(second_time) # in case t_tau was an integer; will drop duplicated initial point
    sol2 = solve_ivp(lambda t, y: dual_inf_ODE(t, y, odeparams),
                    [second_time[0], second_time[-1]], second_Y0, t_eval=second_time)

    # Stitch results together, keeping only the integer timepoints.
    all_times = np.append(sol.t, sol2.t)
    index_to_keep = np.equal(np.mod(all_times, 1), 0)
    all_sol = np.concatenate((sol.y, sol2.y), axis=1) # Append along time axis

    # Final results
    times = all_times[index_to_keep]
    solution = all_sol[:, index_to_keep]

    VTWT = solution[3]
    VSWT = solution[11]
    VTTIP = solution[7]
    VSTIP = solution[15]

    cutlog10 = lambda x: np.log10(x) if x > 0 else 0

    log10VTWT = [cutlog10(x) for x in VTWT]
    log10VSWT = [cutlog10(x) for x in VSWT]
    log10VTTIP = [cutlog10(x) for x in VTTIP]
    log10VSTIP = [cutlog10(x) for x in VSTIP]

    return times, log10VTWT, log10VSWT, log10VTTIP, log10VSTIP

@jit
def dual_inf_ODE(t, y, params):
    """
    Model ODEs for co-infection by wildtype SARS-CoV-2 and TIP, written w/ scipy.integrate.solve_ivp() in mind.
    :param t: Time, days
    :param y: Within-host microbiological states (cells & virions by physiological location)
    :param params: Model parameters, see below for descriptive names.
    :return: Rate change for each microbiological state.
    """
    # Unpack states ------------------------------------------------------------------------------------
    # The states are as follows. All quantities are absolute numbers, not concentrations, so no volume scaling is made
    # during transport between the upper respiratory tract (URT) and the lower respiratory tract (LRT)
    # 0. T1 - target cells in URT
    # 1. E1WT - eclipse-stage cells in URT w/ wildtype virion (infected but not producing virions)
    # 2. I1WT - wildtype infected cells producing virions (URT)
    # 3. VTWT - wildtype virions SAMPLED by a pharyngeal throat swab in URT
    # 4. T1TIP - target cells in URT infected w/ TIP
    # 5. E1D - eclipse-stage cells in URT w/ coinfection by wildtype virus and TIP (infected but not producing virion)
    # 6. I1D - coinfected cells producing both wildtype virion and TIP (URT)
    # 7. VTTIP - TIP virions SAMPLED by a pharyngeal throat swab in URT {assume a hypothetical equivalent test to WT}
    # 8. T2 - target cells in LRT
    # 9. E2WT - eclipse-stage cells in LRT w/ wildtype virion (infected but not producing virions)
    # 10. I2WT - wildtype infected cells producing virions (LRT)
    # 11. VSWT - wildtype virions SAMPLED by a sputum sample from LRT
    # 12. T2TIP - target cells in LRT infected w/ TIP
    # 13. E2D - eclipse-stage cells in LRT w/ coinfection by wildtype virus and TIP (infected but not producing virion)
    # 14. I2D - coinfected cells producing both wildtype virion and TIP (LRT)
    # 15. VSTIP - TIP virions SAMPLEd by a sputum sample from LRT
    T1, E1WT, I1WT, VTWT, T1TIP, E1D, I1D, VTTIP, T2, E2WT, I2WT, VSWT, T2TIP, E2D, I2D, VSTIP = y

    # Unpack parameters ------------------------------------------------------------------------------------
    # beta_T = target cell infection rate constant in URT, divided by sampling fraction f1 (beta_T = beta1/f1)
    # k1 = eclipse phase rate constant, for transition from nonproductive infected cell to productive.
    # pi_T = virion production rate by infected cells in URT, multiplied by sampling fraction f1 (pi_T = f1*p1)
    # delta1 = clearance rate of virion-producing infected cells
    # beta_S, analogous to beta_T but for LRT (beta_S = beta2/f2)
    # k2, analogous to k1 but for LRT
    # pi_S, analogous to pi_T but for LRT (pi_S = f2*p2)
    # delta2, analogous to delta1 but for LRT
    # c = clearance rate constant for virions in both URT and LRT
    # w = exponential growth rate for the clearance of virion-producing infected cells (i.e. adaptive immunity)
    # gamma = scaled rate of virion transfer from URT to LRT, calculates production of VS from VT (gamma = f2/f1*g12)
    # rho = relative TIP virion production rate from coinfected cells (vs. wildtype virion production from coinf. cells)
    # psi = relative wildtype virion production rate from coinfected cells (vs. wildtype virion production from singly infected cells)
    # xi = relative target cell infection rate for TIP versus virion.
    # See Ruian Ke et al. medRxiv 10.1101/2020.09.25.20201772v1 for further definitions, parameter names match.
    beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, rho, psi, xi, delta_TIP = params

    n_entries = 16
    dydt = np.zeros(n_entries)

    # Calculate effect of adaptive immune response
    if t >= 14:
        delta1 = delta1*np.exp(w*(t-14))
        delta2 = delta2*np.exp(w*(t-14))
    else:
        delta1 = delta1
        delta2 = delta2

    # dT1/dt
    dydt[0] = -beta_T*VTWT*T1 - xi*beta_T*VTTIP*T1

    # dE1WT/dt
    dydt[1] = +beta_T*VTWT*T1 - k1*E1WT

    # dI1WT/dt
    dydt[2] = k1*E1WT - delta1*I1WT

    # dVTWT/dt
    dydt[3] = pi_T*I1WT - c*VTWT + psi*pi_T*I1D

    # dT1TIP/dt
    dydt[4] = xi*beta_T*VTTIP*T1 - beta_T*VTWT*T1TIP

    # dE1D/dt
    dydt[5] = beta_T*VTWT*T1TIP - k1*E1D

    # dI1D/dt
    #dydt[6] = k1*E1D - delta1*I1D
    # Note that delta_TIP=1.
    # Non-unity values explore differential effect on immune clearance of TIP-carrying infected cells.
    dydt[6] = k1*E1D - delta1*delta_TIP*I1D

    # dVTTIP/dt
    dydt[7] = rho*pi_T*I1D - c*VTTIP

    # dT2/dt
    dydt[8] = -beta_S*VSWT*T2 - xi*beta_S*VSTIP*T2

    # dE2WT/dt
    dydt[9] = beta_S*VSWT*T2 - k2*E2WT

    # dI2WT/dt
    dydt[10] = k2*E2WT - delta2*I2WT

    # dVSWT/dt
    dydt[11] = pi_S*I2WT - c*VSWT + psi*pi_S*I2D + gamma*VTWT

    # dT2TIP/dt
    dydt[12] = xi*beta_S*VSTIP*T2 - beta_S*VSWT*T2TIP

    # dE2D/dt
    dydt[13] = beta_S*VSWT*T2TIP - k2*E2D

    # dI2D/dt
    dydt[14] = k2*E2D - delta2*delta_TIP*I2D

    # dVSTIP/dt
    dydt[15] = rho*pi_S*I2D - c*VSTIP + gamma*VTTIP

    return dydt

@jit
def single_inf_ODE(t, y, params):
    """
    Model ODES, written with scipy.integrate.solve_ivp() in mind. Single infection by  wildtype SARS-CoV-2.
    :param t: Time, days
    :param y: Within-host microbiological states (cells & virions by physiological location)
    :param params: Model parameters, see below for descriptive names.
    :return dydt: Rate change for each microbiological state.
    """
    # Unpack states ------------------------------------------------------------------------------------
    # The states are as follows. All quantities are absolute numbers, not concentrations, so no volume scaling is made
    # during transport between the upper respiratory tract (URT) and the lower respiratory tract (LRT)
    # 0. T1 - target cells in URT
    # 1. E1 - eclipse-stage cells in URT (infected but not producing virions)
    # 2. I1 - infected cells producing virions (URT)
    # 3. VT - virions SAMPLED by a pharyngeal throat swab in URT
    # 4. T2 - target cells in LRT
    # 5. E2 - eclipse-age cells in LRT (infected but not producing virions)
    # 6. I2 - infected cells producing virions in LRT
    # 7. VS - virions SAMPLED by a sputum sample (LRT)
    T1, E1, I1, VT, T2, E2, I2, VS = y

    # Unpack parameters ------------------------------------------------------------------------------------
    # beta_T = target cell infection rate constant in URT, divided by sampling fraction f1 (beta_T = beta1/f1)
    # k1 = eclipse phase rate constant, for transition from nonproductive infected cell to productive.
    # pi_T = virion production rate by infected cells in URT, multiplied by sampling fraction f1 (pi_T = f1*p1)
    # delta1 = clearance rate of virion-producing infected cells
    # beta_S, analogous to beta_T but for LRT (beta_S = beta2/f2)
    # k2, analogous to k1 but for LRT
    # pi_S, analogous to pi_T but for LRT (pi_S = f2*p2)
    # delta2, analogous to delta1 but for LRT
    # c = clearance rate constant for virions in both URT and LRT
    # w = exponential growth rate for the clearance of virion-producing infected cells (i.e. adaptive immunity)
    # gamma = scaled rate of virion transfer from URT to LRT, calculates production of VS from VT (gamma = f2/f1*g12)
    # See Ruian Ke et al. medRxiv 10.1101/2020.09.25.20201772v1 for further definitions, parameter names match.
    beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma = params

    n_entries = 8
    dydt = np.zeros(n_entries)

    # Calculate effect of adaptive immune response
    if t >= 14:
        delta1 = delta1*np.exp(w*(t-14))
        delta2 = delta2*np.exp(w*(t-14))
    else:
        delta1 = delta1
        delta2 = delta2

    # dT1/dt
    dydt[0] = -beta_T*VT*T1

    # dE1/dt
    dydt[1] = +beta_T*VT*T1 - k1*E1

    # dI1/dt
    dydt[2] = +k1*E1 - delta1*I1

    # dVT/dt
    dydt[3] = pi_T*I1 - c*VT

    # dT2/dt
    dydt[4] = -beta_S*VS*T2

    # dE2/dt
    dydt[5] = +beta_S*VS*T2 - k2*E2

    # dI2/dt
    dydt[6] = +k2*E2 - delta2*I2

    # dVS/dt
    dydt[7] = pi_S*I2 + gamma*VT - c*VS

    return dydt

def source_individual_params(targ_ind):
    """
    Returns an individual parameter set from Tables S6 and [fixed params in] Table 3 in the Ke et al paper.
    :param targ_ind: Specifies which individual to return
    :return: params - (beta_T, delta1, pi_T, beta_S, delta2, pi_S, t_tau, log10TN, w, T10, T20, I10, c, k, gamma)
    """
    # # Note - Table S6 lists beta1, beta2, pi1, pi2 rather than beta_T, beta_S, pi_T, pi_S.
    # # This seems to be a typo since I can reproduce their curves.
    # all_params = [[21.45e-6, 0.86,  3.68, 0.17e-7,  2.2, 10.89,  14.7, 8.21, 0.06, 4e6, 4.8e8, 1, 10, 4, 0.001],
    #               [1.31e-6,  1.82, 15.53,  0.8e-7, 2.18,  2.46,    15, 8.44, 0.18, 4e6, 4.8e8, 1, 10, 4, 0.001],
    #               [13.35e-6, 1.16, 11.61, 2.63e-7, 4.17,  1.67,   6.5, 7.92, 0, 4e6, 4.8e8, 1, 10, 4, 0.001],
    #               [2.4e-6,   3.55, 11.53, 1.35e-7,  1.6,   1.7,  15.7, 10.99, 2.4, 4e6, 4.8e8, 1, 10, 4, 0.001],
    #               [1.41e-6,  1.42, 12.47, 1.06e-7, 2.17,  1.08,    22, 8.21, 0, 4e6, 4.8e8, 1, 10, 4, 0.001],
    #               [6.94e-6,  0.76,  5.89, 0.17e-7, 3.33, 10.34,  17.3, 8.79, 0.15, 4e6, 4.8e8, 1, 10, 4, 0.001],
    #               [18.21e-6, 0.38,  8.74, 9.19e-7, 0.41,  0.15, 17.85, 9, 0.22, 4e6, 4.8e8, 1, 10, 4, 0.001],
    #               [5.12e-6,  3.53,   4.5,  4.9e-7, 2.04,  1.64,   8.3, 6.89, 1.89, 4e6, 4.8e8, 1, 10, 4, 0.001],
    #               [1.53e-6,  4.06,  9.65, 0.29e-7, 3.96,  8.15, 17.11, 9.47, 0.66, 4e6, 4.8e8, 1, 10, 4, 0.001],
    #              ]

    # I've modified it so that I10 is instead VT0, the initial (swabbable) virions from the URT.
    #                                                                                          v--- This is V1(t=0)
    all_params = [[21.45e-6, 0.86,  3.68, 0.17e-7,  2.2, 10.89,  14.7, 8.21, 0.06, 4e6, 4.8e8, 100, 10, 4, 0.001],
                  [1.31e-6,  1.82, 15.53,  0.8e-7, 2.18,  2.46,    15, 8.44, 0.18, 4e6, 4.8e8, 100, 10, 4, 0.001],
                  [13.35e-6, 1.16, 11.61, 2.63e-7, 4.17,  1.67,   6.5, 7.92, 0, 4e6, 4.8e8,    100, 10, 4, 0.001],
                  [2.4e-6,   3.55, 11.53, 1.35e-7,  1.6,   1.7,  15.7, 10.99, 2.4, 4e6, 4.8e8, 100, 10, 4, 0.001],
                  [1.41e-6,  1.42, 12.47, 1.06e-7, 2.17,  1.08,    22, 8.21, 0, 4e6, 4.8e8,    100, 10, 4, 0.001],
                  [6.94e-6,  0.76,  5.89, 0.17e-7, 3.33, 10.34,  17.3, 8.79, 0.15, 4e6, 4.8e8, 100, 10, 4, 0.001],
                  [18.21e-6, 0.38,  8.74, 9.19e-7, 0.41,  0.15, 17.85, 9, 0.22, 4e6, 4.8e8,    100, 10, 4, 0.001],
                  [5.12e-6,  3.53,   4.5,  4.9e-7, 2.04,  1.64,   8.3, 6.89, 1.89, 4e6, 4.8e8, 100, 10, 4, 0.001],
                  [1.53e-6,  4.06,  9.65, 0.29e-7, 3.96,  8.15, 17.11, 9.47, 0.66, 4e6, 4.8e8, 100, 10, 4, 0.001],
                 ]
    currparam = all_params[targ_ind]
    return currparam

def calculate_log10VLs(rho, psi, xi, delta_TIP, fractionTIP):
    log10VL_WT_URT_singleInf = np.nan*np.ones((31, 9))
    log10VL_WT_LRT_singleInf = np.nan*np.ones((31, 9))
    log10VL_WT_URT_dualInf = np.nan*np.ones((31, 9))
    log10VL_WT_LRT_dualInf = np.nan*np.ones((31, 9))
    log10VL_TIP_URT_dualInf = np.nan*np.ones((31, 9))
    log10VL_TIP_LRT_dualInf = np.nan*np.ones((31, 9))
    for targ_ind in range(9):
        single_params = source_individual_params(targ_ind)
        dual_params = single_params.copy()
        dual_params.append(rho)
        dual_params.append(psi)
        dual_params.append(xi)
        dual_params.append(delta_TIP)
        dual_params.append(fractionTIP)
        # Integrate over smaller window for certain individuals to avoid bad numerics
        if targ_ind == 3:
            t_single, log10VT, log10VS = simulate_single_inf(single_params, 0, 20)
            t_dual, log10VTWT, log10VSWT, log10VTTIP, log10VSTIP = simulate_dual_inf(dual_params, 0, 20)
        elif targ_ind == 7:
            t_single, log10VT, log10VS = simulate_single_inf(single_params, 0, 16)
            t_dual, log10VTWT, log10VSWT, log10VTTIP, log10VSTIP = simulate_dual_inf(dual_params, 0, 16)
        else:
            t_single, log10VT, log10VS = simulate_single_inf(single_params, 0, 30)
            t_dual, log10VTWT, log10VSWT, log10VTTIP, log10VSTIP = simulate_dual_inf(dual_params, 0, 30)

        npts = len(log10VT)
        log10VL_WT_URT_singleInf[0:npts, targ_ind] = log10VT
        log10VL_WT_LRT_singleInf[0:npts, targ_ind] = log10VS
        log10VL_WT_URT_dualInf[0:npts, targ_ind] = log10VTWT
        log10VL_WT_LRT_dualInf[0:npts, targ_ind] = log10VSWT
        log10VL_TIP_URT_dualInf[0:npts, targ_ind] = log10VTTIP
        log10VL_TIP_LRT_dualInf[0:npts, targ_ind] = log10VSTIP


    return log10VL_WT_URT_singleInf, log10VL_WT_LRT_singleInf,\
           log10VL_WT_URT_dualInf, log10VL_WT_LRT_dualInf, \
           log10VL_TIP_URT_dualInf, log10VL_TIP_LRT_dualInf


def visualize_impact_of_TIP_prophylaxis_timecourses_v2(figurebasename,
                                                       rho, psi, xi, delta_TIP, fractionTIP):
    patient_data = get_data()
    log10VL_WT_URT_singleInf, log10VL_WT_LRT_singleInf, \
    log10VL_WT_URT_dualInf, log10VL_WT_LRT_dualInf, \
    log10VL_TIP_URT_dualInf, log10VL_TIP_LRT_dualInf = calculate_log10VLs(rho=rho,
                                                                          psi=psi,
                                                                          xi=xi,
                                                                          delta_TIP=delta_TIP,
                                                                          fractionTIP=fractionTIP)

    fig, axs = plt.subplots(3, 3, figsize=(7, 6))

    URT_WT_only_color = '#000000'
    URT_WT_TIP_color = '#336699'
    LRT_WT_only_color = '#999999'
    LRT_WT_TIP_color = '#66CCFF'

    t = np.arange(0, 31, 1)
    for targ_ind in range(9):
        pltcol = int(np.mod(targ_ind, 3))
        pltrow = int(np.floor(targ_ind / 3))

        # URT
        axs[pltrow, pltcol].plot(t, log10VL_WT_URT_singleInf[:, targ_ind], label='URT', color=URT_WT_only_color, linewidth=2.5)
        axs[pltrow, pltcol].plot(t, log10VL_WT_URT_dualInf[:, targ_ind], label='URT (+TIP)', color=URT_WT_TIP_color, linewidth=2.5)
        axs[pltrow, pltcol].set_ylim(0, 10)

        # LRT
        axs[pltrow, pltcol].plot(t, log10VL_WT_LRT_singleInf[:, targ_ind], label='LRT', color=LRT_WT_only_color, linewidth=2.5)
        axs[pltrow, pltcol].plot(t, log10VL_WT_LRT_dualInf[:, targ_ind], label='LRT (+TIP)', color=LRT_WT_TIP_color, linewidth=2.5)

        axs[pltrow, pltcol].plot([-2, 32], [2, 2], 'k', label='LOD', linewidth=1, linestyle='dotted')

        axs[pltrow, pltcol].set_xlim([0, 30])
        axs[pltrow, pltcol].set_ylim([0, 10])
        axs[pltrow, pltcol].set_xticks([0, 10, 20, 30])
        axs[pltrow, pltcol].set_yticks([0, 2, 4, 6, 8, 10])
        axs[pltrow, pltcol].set_xticklabels(labels=['0','10','20','30'],fontsize=14)
        axs[pltrow, pltcol].set_yticklabels(labels=['0','2','4','6','8','10'], fontsize=14)
        axs[pltrow, pltcol].text(25, 8.5, targ_ind+1)

    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=.15, hspace=.15)

    # Turn off ticks for specific subplots
    axs[0, 1].set_yticklabels([])
    axs[0, 2].set_yticklabels([])
    axs[1, 1].set_yticklabels([])
    axs[1, 2].set_yticklabels([])
    axs[2, 1].set_yticklabels([])
    axs[2, 2].set_yticklabels([])

    axs[0, 0].set_xticklabels([])
    axs[0, 1].set_xticklabels([])
    axs[0, 2].set_xticklabels([])
    axs[1, 0].set_xticklabels([])
    axs[1, 1].set_xticklabels([])
    axs[1, 2].set_xticklabels([])

    axs[2, 1].set_xlabel('time past exposure (days)')
    axs[1, 0].set_ylabel('log10 SARS-CoV-2')

    #plt.tight_layout(pad=1.2)

    figurename = figurebasename + 'WTchange_URTLRT_nicer.png'
    figurename_pdf = figurebasename + 'WTchange_URTLRT_nicer.pdf'

    fig.savefig(figurename, dpi=300)
    fig.savefig(figurename_pdf, dpi=300, transparent=True)


def get_change_peak_VL(filebasename, rho, psi, xi, delta_TIP):
    fractionTIPs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    log10FC_matrix_URT = np.zeros((len(fractionTIPs), 9))
    log10FC_matrix_LRT = np.zeros((len(fractionTIPs), 9))

    log10AUCDifference_matrix_URT = np.zeros((len(fractionTIPs), 9))
    log10AUCDifference_matrix_LRT = np.zeros((len(fractionTIPs), 9))
    LOD = 2

    for fracIndex, fractionTIP in enumerate(fractionTIPs):
        print('working on fractionTIP {:.2f}'.format(fractionTIP))
        log10VL_WT_URT_singleInf, log10VL_WT_LRT_singleInf, \
        log10VL_WT_URT_dualInf, log10VL_WT_LRT_dualInf, \
        log10VL_TIP_URT_dualInf, log10VL_TIP_LRT_dualInf = calculate_log10VLs(rho=rho,
                                                                              psi=psi,
                                                                              xi=xi,
                                                                              delta_TIP=delta_TIP,
                                                                              fractionTIP=fractionTIP)

        for targ_ind in range(9):
            WT_URT_singleInf_peak_index = np.nanargmax(log10VL_WT_URT_singleInf[:, targ_ind])
            WT_URT_dualInf_peak_index = np.nanargmax(log10VL_WT_URT_dualInf[:, targ_ind])

            WT_LRT_singleInf_peak_index = np.nanargmax(log10VL_WT_LRT_singleInf[:, targ_ind])
            WT_LRT_dualInf_peak_index = np.nanargmax(log10VL_WT_LRT_dualInf[:, targ_ind])

            # Calcualte difference between single and dual infection at their peaks, limtied by LOD
            dual_peakval_URT = np.maximum(log10VL_WT_URT_dualInf[WT_URT_dualInf_peak_index, targ_ind], LOD)
            single_peakval_URT = np.maximum(log10VL_WT_URT_singleInf[WT_URT_singleInf_peak_index, targ_ind], LOD)

            dual_peakval_LRT = np.maximum(log10VL_WT_LRT_dualInf[WT_LRT_dualInf_peak_index, targ_ind], LOD)
            single_peakval_LRT = np.maximum(log10VL_WT_LRT_singleInf[WT_LRT_singleInf_peak_index, targ_ind], LOD)

            WT_URT_log10Difference = single_peakval_URT - dual_peakval_URT
            WT_LRT_log10Difference = single_peakval_LRT - dual_peakval_LRT

            log10FC_matrix_URT[fracIndex, targ_ind] = WT_URT_log10Difference
            log10FC_matrix_LRT[fracIndex, targ_ind] = WT_LRT_log10Difference

            # Calculate difference in AUC.
            # First, find AUC [limited by LOD] in 10**(log10VL)space [removing nans]. Then take log10 of that.
            single_AUC_URT = np.log10(np.nansum(np.power(10, log10VL_WT_URT_singleInf[:, targ_ind])))
            dual_AUC_URT = np.log10(np.nansum(np.power(10, log10VL_WT_URT_dualInf[:, targ_ind])))
            single_AUC_LRT = np.log10(np.nansum(np.power(10, log10VL_WT_LRT_singleInf[:, targ_ind])))
            dual_AUC_LRT = np.log10(np.nansum(np.power(10, log10VL_WT_LRT_dualInf[:, targ_ind])))

            # Difference between single and dual infection, limited by LOD [100 virions].
            log10AUCDifference_matrix_URT[fracIndex, targ_ind] = single_AUC_URT - dual_AUC_URT
            log10AUCDifference_matrix_LRT[fracIndex, targ_ind] = single_AUC_LRT - dual_AUC_LRT

    # Construct a pandas dataframe
    df = pd.DataFrame(columns=('log10Difference', 'log10AUCDifference', 'fractionTIP', 'patientID', 'VLcompartment'))
    for i in range(len(fractionTIPs)):
        for j in range(9):
            df = df.append({'log10Difference': log10FC_matrix_URT[i, j],
                            'log10AUCDifference': log10AUCDifference_matrix_URT[i, j],
                            'fractionTIP': fractionTIPs[i],
                            'patientID': j,
                            'VLcompartment': 'URT'},
                           ignore_index=True)
            df = df.append({'log10Difference': log10FC_matrix_LRT[i, j],
                            'log10AUCDifference': log10AUCDifference_matrix_LRT[i, j],
                            'fractionTIP': fractionTIPs[i],
                            'patientID': j,
                            'VLcompartment': 'LRT'},
                           ignore_index=True)
    # URT figure
    plt.figure()
    sns.stripplot(data=df[df['VLcompartment'] == 'URT'], x='fractionTIP', y='log10Difference', color='k')
    plt.plot(np.arange(0, len(fractionTIPs)), np.nanmedian(log10FC_matrix_URT, axis=1),
             'k', marker='_', markersize=20, linestyle='None')
    plt.xlabel('Fraction URT cells pre-converted to TIP carriers')
    plt.ylabel('Change in peak log10VL due to TIPs')
    plt.ylim((-0.21526150309991682, 4.565798949695638))
    plt.savefig(filebasename+'_URT.png', dpi=300)

    # LRT figure
    plt.figure()
    sns.stripplot(data=df[df['VLcompartment'] == 'LRT'], x='fractionTIP', y='log10Difference', color='k')
    plt.plot(np.arange(0, len(fractionTIPs)), np.nanmedian(log10FC_matrix_LRT, axis=1),
             'k', marker='_', markersize=20, linestyle='None')
    plt.xlabel('Fraction URT cells pre-converted to TIP carriers')
    plt.ylabel('Change in peak log10VL due to TIPs')
    #plt.ylim((-0.21526150309991682, 4.565798949695638))
    plt.savefig(filebasename+'_LRT.png', dpi=300)

    df.to_csv(filebasename+'.csv')

def get_change_peak_VL_fromfile(filebasename, rho, psi, xi, delta_TIP, csvfile):
    fractionTIPs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    log10FC_matrix_URT = np.zeros((len(fractionTIPs), 9))
    log10FC_matrix_LRT = np.zeros((len(fractionTIPs), 9))
    LOD = 2

    # Construct a pandas dataframe
    df = pd.read_csv(csvfile)

    for i, fracTIP in enumerate(fractionTIPs):
        for patientID in range(9):
            log10FC_matrix_LRT[i, patientID] = df[(df['patientID']==patientID) &
                                                  (df['fractionTIP']==fracTIP) &
                                                  (df['VLcompartment']=='LRT')]['log10Difference'].values[0]
            log10FC_matrix_URT[i, patientID] = df[(df['patientID']==patientID) &
                                                  (df['fractionTIP']==fracTIP) &
                                                  (df['VLcompartment']=='URT')]['log10Difference'].values[0]

    # URT figure
    plt.figure(figsize=(8, 6.4))
    sns.stripplot(data=df[df['VLcompartment'] == 'URT'], x='fractionTIP', y='log10Difference', color='k')
    plt.plot(np.arange(0, len(fractionTIPs)), np.nanmedian(log10FC_matrix_URT, axis=1),
             'r', marker='_', markersize=40, linestyle='None')
    plt.xlabel('TIP-carrying cells (fraction)')
    plt.ylabel('Reduction in peak log10 SARS-CoV-2')
    plt.ylim((-0.21526150309991682, 4.565798949695638))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax=plt.gca()
    ax.invert_yaxis()
    plt.savefig(filebasename+'_URT.png', dpi=300)
    plt.savefig(filebasename+'_URT.pdf', dpi=300, transparent=True)

    # LRT figure
    plt.figure(figsize=(8, 6.4))
    sns.stripplot(data=df[df['VLcompartment'] == 'LRT'], x='fractionTIP', y='log10Difference', color='k')
    plt.plot(np.arange(0, len(fractionTIPs)), np.nanmedian(log10FC_matrix_LRT, axis=1),
             'r', marker='_', markersize=40, linestyle='None')
    plt.xlabel('TIP-carrying cells (fraction)')
    plt.ylabel('Reduction in peak log10 SARS-CoV-2')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax=plt.gca()
    ax.invert_yaxis()
    #plt.ylim((-0.21526150309991682, 4.565798949695638))
    plt.savefig(filebasename+'_LRT.png', dpi=300)
    plt.savefig(filebasename+'_LRT.pdf', dpi=300, transparent=True)

    df.to_csv(filebasename+'.csv')

def __main__():
    # Run simulations, save .csv of results
    get_change_peak_VL(filebasename='100virion_result',
                       rho=1.5, psi=0.02, xi=1, delta_TIP=1)

    # Use .csv of results to render dot-plot of peak patient responses.
    get_change_peak_VL_fromfile(filebasename='100virion_result',
                                rho=1.5, psi=0.02, xi=1, delta_TIP=1,
                                csvfile='100virion_result.csv')

if __name__ == '__main__':
    __main__()

