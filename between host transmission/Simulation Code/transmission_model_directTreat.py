# 2021 May 29 - Mike Pablo
# Re-implementing the transmission model code.
# A major clean-up point: I'm going to choose a small set of possible
# transmission distributions based on the 'culture probability vs. log10 viral load'
# plot in Figure 2C from Jones TC and Biele G et al. 'Estimating infectiousness throughout
# SARS-CoV-2 infection course'. Science 2021.
# We can approximate their curve with a Hill function:
# P = (log10VL ^ r) / (K^r + log10VL ^ r)
# for culture probability (transmission probability) P
# r = 10.18, K = 7.165
#
# I will still need to explore a range of contact parameters (dispersion)

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numba import jit
import multiprocessing
import traceback
from scipy.stats import ks_2samp

def simulate_transmissions_wrapper(paramset):
    dispersion = paramset['dispersion']
    tzero = paramset['tzero']
    alpha = paramset['alpha']
    lambda_tx = paramset['lambda_tx']
    theta = paramset['theta']
    n_transmitters = paramset['n_transmitters']
    dT = paramset['dT']
    Tx_max = paramset['Tx_max']
    hostParameterFile = paramset['hostParameterFile']
    rho = paramset['rho']
    psi = paramset['psi']
    xi = paramset['xi']
    eta = paramset['eta']
    delta_TIP = paramset['delta_TIP']
    try:
        printvec = [dispersion, tzero, alpha, lambda_tx, theta, Tx_max]
        print('Attempting WTTIP-WT simulation with parameters: ', printvec)
        results_WT = simulate_transmissions_WTTIP_WT_tx(dispersion=dispersion, tzero=tzero, alpha=alpha, lambda_tx=lambda_tx,
                                                        theta=theta, n_transmitters=n_transmitters, dT=dT, Tx_sat=Tx_max,
                                                        hostParameterFile=hostParameterFile,
                                                        rho=rho, psi=psi, xi=xi, eta=eta, delta_TIP=delta_TIP)
        # Get relevant statistics
        GT_infections = results_WT['GT'].values
        R0_infections = results_WT['R0'].values
        SI_infections = results_WT['SI'].values

        # Weird bug in numpy 1.19.2 vs pandas 1.1.3
        # Indexing .values from GT_infection, R0_infections, and SI_infections
        # gives a ndarray with dtype='object'. np.nanmean sometimes fails with dtype='object'
        # Since the float type can handle our data just fine, we cast to that first.
        meanGT = np.nanmean(GT_infections.astype('float'))
        meanR0 = np.nanmean(R0_infections.astype('float'))
        meanSI = np.nanmean(SI_infections.astype('float'))

        # We'll compute RSS based on R0_heights_ref
        R0_array_ref, R0_heights_ref = get_reference_individual_R0_data()
        R0_heights_sim = extract_simulated_R0_bar_heights(R0_infections)
        R0_RSS = np.sum((R0_heights_sim - R0_heights_ref) ** 2)

        fit_results_WT = {'dispersion': dispersion,
                       'tzero': tzero,
                       'alpha': alpha,
                       'lambda_tx': lambda_tx,
                       'theta': theta,
                       'Tx_max': Tx_max,
                       'rho': rho,
                       'psi': psi,
                       'xi': xi,
                       'eta': eta,
                       'delta_TIP': delta_TIP,
                       'meanR0': meanR0,
                       'meanGT': meanGT,
                       'meanSI': meanSI,
                       'R0_RSS': R0_RSS}

    except Exception as ex:
        template = "Skipping parameter set; an exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        print(traceback.format_exc())
        fit_results_WT = {'dispersion': dispersion,
                       'tzero': tzero,
                       'alpha': alpha,
                       'lambda_tx': lambda_tx,
                       'theta': theta,
                       'Tx_max': Tx_max,
                       'rho': rho,
                       'psi': psi,
                       'xi': xi,
                       'eta': eta,
                       'delta_TIP': delta_TIP,
                       'meanR0': np.nan,
                       'meanGT': np.nan,
                       'meanSI': np.nan,
                       'R0_RSS': np.nan}

    try:
        printvec = [dispersion, tzero, alpha, lambda_tx, theta, Tx_max]
        print('Attempting WTTIP-WTTIP simulation with parameters: ', printvec)
        results_WTTIP = simulate_transmissions_WTTIP_WTTIP_tx(dispersion=dispersion, tzero=tzero, alpha=alpha,
                                                             lambda_tx=lambda_tx,
                                                             theta=theta, n_transmitters=n_transmitters, dT=dT,
                                                             Tx_sat=Tx_max,
                                                             hostParameterFile=hostParameterFile,
                                                             rho=rho, psi=psi, xi=xi, eta=eta, delta_TIP=delta_TIP)
        # Get relevant statistics
        GT_infections = results_WTTIP['GT'].values
        R0_infections = results_WTTIP['R0'].values
        SI_infections = results_WTTIP['SI'].values

        # Weird bug in numpy 1.19.2 vs pandas 1.1.3
        # Indexing .values from GT_infection, R0_infections, and SI_infections
        # gives a ndarray with dtype='object'. np.nanmean sometimes fails with dtype='object'
        # Since the float type can handle our data just fine, we cast to that first.
        meanGT = np.nanmean(GT_infections.astype('float'))
        meanR0 = np.nanmean(R0_infections.astype('float'))
        meanSI = np.nanmean(SI_infections.astype('float'))

        # We'll compute RSS based on R0_heights_ref
        R0_array_ref, R0_heights_ref = get_reference_individual_R0_data()
        R0_heights_sim = extract_simulated_R0_bar_heights(R0_infections)
        R0_RSS = np.sum((R0_heights_sim - R0_heights_ref) ** 2)

        fit_results_WTTIP = {'dispersion': dispersion,
                             'tzero': tzero,
                             'alpha': alpha,
                             'lambda_tx': lambda_tx,
                             'theta': theta,
                             'Tx_max': Tx_max,
                             'rho': rho,
                             'psi': psi,
                             'xi': xi,
                             'eta': eta,
                             'delta_TIP': delta_TIP,
                             'meanR0': meanR0,
                             'meanGT': meanGT,
                             'meanSI': meanSI,
                             'R0_RSS': R0_RSS}

    except Exception as ex:
        template = "Skipping parameter set; an exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        print(traceback.format_exc())
        fit_results_WTTIP = {'dispersion': dispersion,
                             'tzero': tzero,
                             'alpha': alpha,
                             'lambda_tx': lambda_tx,
                             'theta': theta,
                             'Tx_max': Tx_max,
                             'rho': rho,
                             'psi': psi,
                             'xi': xi,
                             'eta': eta,
                             'delta_TIP': delta_TIP,
                             'meanR0': np.nan,
                             'meanGT': np.nan,
                             'meanSI': np.nan,
                             'R0_RSS': np.nan}

    try:
        printvec = [dispersion, tzero, alpha, lambda_tx, theta, Tx_max]
        print('Attempting WTTIP-TIP simulation with parameters: ', printvec)
        results_TIP = simulate_transmissions_WTTIP_TIP_tx(dispersion=dispersion, tzero=tzero, alpha=alpha,
                                                            lambda_tx=lambda_tx,
                                                            theta=theta, n_transmitters=n_transmitters, dT=dT,
                                                            Tx_sat=Tx_max,
                                                            hostParameterFile=hostParameterFile,
                                                            rho=rho, psi=psi, xi=xi, eta=eta, delta_TIP=delta_TIP)
        # Get relevant statistics
        GT_infections = results_TIP['GT'].values
        R0_infections = results_TIP['R0'].values
        SI_infections = results_TIP['SI'].values

        # Weird bug in numpy 1.19.2 vs pandas 1.1.3
        # Indexing .values from GT_infection, R0_infections, and SI_infections
        # gives a ndarray with dtype='object'. np.nanmean sometimes fails with dtype='object'
        # Since the float type can handle our data just fine, we cast to that first.
        meanGT = np.nanmean(GT_infections.astype('float'))
        meanR0 = np.nanmean(R0_infections.astype('float'))
        meanSI = np.nanmean(SI_infections.astype('float'))

        # We'll compute RSS based on R0_heights_ref
        R0_array_ref, R0_heights_ref = get_reference_individual_R0_data()
        R0_heights_sim = extract_simulated_R0_bar_heights(R0_infections)
        R0_RSS = np.sum((R0_heights_sim - R0_heights_ref) ** 2)

        fit_results_TIP = {'dispersion': dispersion,
                           'tzero': tzero,
                           'alpha': alpha,
                           'lambda_tx': lambda_tx,
                           'theta': theta,
                           'Tx_max': Tx_max,
                           'rho': rho,
                           'psi': psi,
                           'xi': xi,
                           'eta': eta,
                           'delta_TIP': delta_TIP,
                           'meanR0': meanR0,
                           'meanGT': meanGT,
                           'meanSI': meanSI,
                           'R0_RSS': R0_RSS}

    except Exception as ex:
        template = "Skipping parameter set; an exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        print(traceback.format_exc())
        fit_results_TIP = {'dispersion': dispersion,
                           'tzero': tzero,
                           'alpha': alpha,
                           'lambda_tx': lambda_tx,
                           'theta': theta,
                           'Tx_max': Tx_max,
                           'rho': rho,
                           'psi': psi,
                           'xi': xi,
                           'eta': eta,
                           'delta_TIP': delta_TIP,
                           'meanR0': np.nan,
                           'meanGT': np.nan,
                           'meanSI': np.nan,
                           'R0_RSS': np.nan}

    return fit_results_WT, fit_results_WTTIP, fit_results_TIP


def write_host_parameter_file_Goyal(filename, n_entries):
    # This introduces quite arbitrary heterogeneity around the parameter estimates (from Goyal et al.,
    # "When we introduce simulated heterogeneity in cases of SARS-CoV-2 (by increasing the
    # standard deviation of the random effects of parameters β by 20, δ by 2, k by 2 and π by 5 in the
    # original distribution from [ref 24])")
    # Fixed effects = mean, random effects = std.
    # The values listed in their paper don't always match up to the ones I find in their GitHub:
    # ========================================================================
    # PARAMETER         |   REPORTED          | APPROXIMATE, FROM .TXT FILE
    # beta              | (-7.23, 0.2)        | (-7.23, 0.2)
    # delta             | (3.13, 0.02)        | (3.13, 0.06)
    # k                 | (0.08, 0.02)        | (0.078, 0.002)
    # p                 | 10^(2.59, 0.05)     | 10^(2.59, 0.05)
    # m                 | (3.21, 0.33)        | (3.21, 0.33*1.155)
    # w                 | 10^(-4.55, 0.01)    | 10^(-4.55, 0.01)
    # ========================================================================
    # See also "validate_Goyal_parameter_sampling()". I use the 'approximate, from txt file' entries.

    mean_log10beta = -7.23  # log10(virions)/day
    std_log10beta = 0.2
    mean_delta = 3.13  # 1/day cells^{-k}
    std_delta = 0.02*3
    mean_k = 0.078
    std_k = 0.002
    mean_log10pi = 2.59  #log10(?)/day
    std_log10pi = 0.05
    mean_m = 3.21  # 1/day 1/cells
    std_m = 0.33*3.5
    mean_log10w = 4.55
    std_log10w = 0.01

    r = 10
    de = 1
    q = 2.4 * 10**-5
    c = 15
    E50 = 100  # this value wasn't listed in their manuscript, but their SimulatedParameters.txt file available from
               # GitHub had E50 fixed to 100 across all parameters.

    # Generate a list of entries, and write them to the filename.
    # Header names: beta, delta, k, m, w, de, E50, r, q, p, c
    # The E50 is 'phi' in their within-host ODEs.

    # Create empty dataframe to store parameters
    results = pd.DataFrame(columns=['beta', 'delta', 'k', 'm', 'w', 'de', 'E50', 'r', 'q', 'p', 'c'])

    for i in range(n_entries):
        beta = 10**np.random.normal(loc=mean_log10beta, scale=std_log10beta)
        delta = np.random.normal(loc=mean_delta, scale=std_delta)
        k = np.random.normal(loc=mean_k, scale=std_k)
        p = 10**np.random.normal(loc=mean_log10pi, scale=std_log10pi)
        m = np.random.normal(loc=mean_m, scale=std_m)
        w = 10**np.random.normal(loc=mean_log10w, scale=std_log10w)

        results = results.append({'beta': beta,
                                  'delta': delta,
                                  'k': k,
                                  'm': m,
                                  'w': w,
                                  'de': de,
                                  'E50': E50,
                                  'r': r,
                                  'q': q,
                                  'p': p,
                                  'c': c}, ignore_index=True)
    results.to_csv(filename, header=True, index=False)

def validate_Goyal_parameter_sampling():
    # The parameter sampling that showed up in SimulatedParameters.txt (from their GitHub)
    # doesn't quite match up with the parameter sampling described in their medRxiv paper.
    # Values are written as the (MEAN, STD) of a normal distribution.
    # ========================================================================
    # PARAMETER         |   REPORTED          | APPROXIMATE, FROM .TXT FILE
    # beta              | (-7.23, 0.2)        | (-7.23, 0.2)
    # delta             | (3.13, 0.02)        | (3.13, 0.02*3)
    # k                 | (0.08, 0.02)        | (0.078, 0.002)
    # p                 | 10^(2.59, 0.05)     | 10^(2.59, 0.05)
    # m                 | (3.21, 0.33)        | (3.21, 0.33*1.155)
    # w                 | 10^(-4.55, 0.01)    | 10^(-4.55, 0.01)
    # ========================================================================
    # This is shown below; first histogram (default, blue) is from their datafile,
    # second histogram (default, orange) is from the "approximate, from txt file"

    df = pd.read_csv('Other-Initial-Results/SimulatedParameters.txt')
    fig, axes = plt.subplots(nrows=2, ncols=3)
    stdscale_delta = 3  # stated 2 in paper
    stdscale_m = 3.5 # not listed in paper
    # expected to need scaling on std. of beta, k, pi but not observed based on the data.

    # beta
    axes[0,0].hist(df['beta'], density=True, bins=np.arange(0,2.5e-7,.05e-7), alpha=0.5)
    test=10**np.random.normal(-7.23,0.2,size=1000)
    axes[0,0].hist(test, density=True, bins=np.arange(0,2.5e-7,.05e-7), alpha=0.5)
    axes[0,0].set_title('beta')

    #delta
    axes[0,1].hist(df['delta'], density=True, bins=np.arange(2.75,3.5,0.05), alpha=0.5)
    test=np.random.normal(3.13,0.02*stdscale_delta,size=1000)
    axes[0,1].hist(test, density=True, bins=np.arange(2.75,3.5,0.05), alpha=0.5)
    axes[0,1].set_title('delta')

    #k
    axes[0,2].hist(df['k'], density=True, bins=np.arange(0.065,0.085,0.001), alpha=0.5)
    test=np.random.normal(0.078,0.002,size=1000)  # Table S1 says mean=0.8, std=0.02; replace w/ mean=0.078
    axes[0,2].hist(test, density=True, bins=np.arange(0.065,0.085,0.001), alpha=0.5)
    axes[0,2].set_title('k')

    #pi
    axes[1,0].hist(df['p'], density=True, bins=np.arange(200,600,10), alpha=0.5)
    test=10**(np.random.normal(2.59,0.05,size=1000))
    axes[1,0].hist(test, density=True, bins=np.arange(200,600,10), alpha=0.5)
    axes[1,0].set_title('pi')

    #m
    axes[1,1].hist(df['m'], density=True, bins=np.arange(0,15,0.5), alpha=0.5)
    test=np.random.normal(3.21,0.33*stdscale_m,size=1000) # Table S1 says mean=3.21, std=0.33; replace w/ std=0.33*3.5
    axes[1,1].hist(test, density=True, bins=np.arange(0,15,0.5), alpha=0.5)
    axes[1,1].set_title('m')

    #w
    axes[1,2].hist(df['w'], density=True, bins=np.arange(2.5e-5,3e-5,.025e-5), alpha=0.5)
    test=10**(np.random.normal(-4.55,0.01,size=1000)) # table S1 says -4.55
    axes[1,2].hist(test, density=True, bins=np.arange(2.5e-5,3e-5,.025e-5), alpha=0.5)
    axes[1,2].set_title('w')
    plt.show()

def write_host_parameter_file_Ke(filename, n_entriesPerInd):
    # 20201105 - After visualizing dynamics, I'm omitting the 9th individual
    # Ke et al. report a 95% confidence interval for the best-fit parameters.
    # They're almost certainly non-Gaussian, but the distributions are not published,
    # so I'll assume they are for the sake of resampling.
    # Several of their individuals have parameters that cannot be identified.
    # To have representation of those individuals, I'll just assume the window is +/- mean.
    # I'm also going to clip all sampling to reasonable numbers (no going below zero, e.g.)
    # and will generate overlay plots of the predictions on the data.

    # We'll assume a normal distribution with mean given by **_means, and standard deviation given by
    # (UB-LB)/(2*1.96). We will draw n_entries from this normal distribution, but discarding and re-drawing
    # if a sample falls outside of the UB/LB.

    ## HARDCODED MEAN AND 95% CI FOR NINE INDIVIDUALS
    # These values are obtained from Table S6 and Table S7.
    # np.nan in lower/upper bounds (LB/UB) means it was not listed in Table S7.
    betaT_means = [21.45e-6, 1.31e-6, 13.35e-6, 2.4e-6, 1.41e-6, 6.94e-6, 18.21e-6, 5.12e-6, 1.53e-6]
    betaT_LB = [5.3e-6, 0.03e-6, 2.3e-6, 0.003e-6, 0.2e-6, 0.9e-6, np.nan, 0.9e-6, np.nan]
    betaT_UB = [206.5e-6, 4.5e-6, np.nan, 29e-6, 9.8e-6, np.nan, np.nan, 25.9e-6, np.nan]

    delta1_means = [0.86, 1.82, 1.16, 3.55, 1.42, 0.76, 0.38, 3.53, 4.06]
    delta1_LB = [0.5, 1.1, 0.8, 0.7, 0.6, 0.5, 0.3, 1.1, np.nan]
    delta1_UB = [1.2, 10, 1.5, 10, 1.3, 1, 0.5, 36.7, np.nan]

    piT_means = [3.63, 15.53, 11.61, 11.53, 12.47, 5.89, 8.74, 4.5, 9.65]
    piT_LB = [0.4, 4.1, 0.2, 1.5, 1.6, 0.1, np.nan, 0.6, np.nan]
    piT_UB = [5.7, 1021, 227.8, np.nan, 113.1, 111.3, np.nan, 56.2, np.nan]

    betaS_means = [0.17e-7, 0.8e-7, 2.63e-7, 1.35e-7, 1.06e-7, 0.17e-7, 9.19e-7, 4.9e-7, 0.29e-7]
    betaS_LB = [0.02e-7, 0.1e-7, 1.6e-7, 0.03e-7, 0.7e-7, 0.06e-7, np.nan, np.nan, np.nan]
    betaS_UB = [1.2e-7, 2.8e-7, 78.6e-7, 4.1e-7, 2e-7, 1.1e-7, np.nan, np.nan, np.nan]

    delta2_means = [2.2, 2.18, 4.17, 1.6, 2.17, 3.33, 0.41, 2.04, 3.96]
    delta2_LB = [1.2, 1, 1, 0.8, 1.1, 1.2, 0.3, 0.9, np.nan]
    delta2_UB = [4.6, 7.4, 10.0, 1.8, 6.5, 10, 0.7, 10, np.nan]

    piS_means = [10.89, 2.46, 1.67, 1.7, 1.08, 10.34, 0.15, 1.64, 8.15]
    piS_LB = [2, 0.5, 0.2, np.nan, 0.8, np.nan, 0.1, 0.3, np.nan]
    piS_UB = [145.9, 15.1, 7.5, np.nan, 1.4, np.nan, 75, 10.6, np.nan]

    ttau_means = [14.7, 15, 6.5, 15.7, 22, 17.3, 17.85, 8.3, 17.11]
    ttau_LB = [12.2, 13, 4.2, 14.9, 15.9, 15.5, 18.5, 3.6, np.nan]
    ttau_UB = [16.6, 16.4, 7.6, 15.9, 29.1, 18.9, 19.7, 30, np.nan]

    log10TN_means = [8.21, 8.44, 7.92, 10.99, 8.21, 8.79, 9, 6.89, 9.47]
    log10TN_LB = [7.6, 7.7, 6.1, 9.8, 7.8, 8.1, 5.4, 0, np.nan]
    log10TN_UB = [8.7, 8.9, 8.3, 13.9, 8.3, 9, 10.1, 8, np.nan]

    w_means = [0.06, 0.18, 0, 2.4, 0, 0.15, 0.22, 1.89, 0.66]
    w_LB = [0, 0, 0, 0.5, 0, 0, 0.1, 0.2, np.nan]
    w_UB = [0.3, 0.3, 0.1, 10, 0.1, 0.3, 0.25, 10, np.nan]

    # Fixed parameters
    k1 = 4
    k2 = 4
    gamma = 0.001
    T10 = 4e6
    T20 = 4.8e8
    I10 = 1
    c = 10

    n_inds = len(w_means)

    results = pd.DataFrame(columns=['betaT', 'delta1', 'piT', 'betaS', 'delta2',
                                    'piS', 'ttau', 'log10TN', 'w', 'k1', 'k2',
                                    'gamma', 'T10', 'T20', 'I10', 'c'])
    for i in range(n_inds):
        # Do not include the last individual.
        if i == 8:
            print('Excluding last individual from Ke et al. within-host parameter set.')
            continue
        betaT_mean_i = betaT_means[i]
        delta1_mean_i = delta1_means[i]
        piT_mean_i = piT_means[i]
        betaS_mean_i = betaS_means[i]
        delta2_mean_i = delta2_means[i]
        piS_mean_i = piS_means[i]
        ttau_mean_i = ttau_means[i]
        log10TN_mean_i = log10TN_means[i]
        w_mean_i = w_means[i]

        betaT_std_i = (betaT_UB[i] - betaT_LB[i])/(2*1.96)
        delta1_std_i = (delta1_UB[i] - delta1_LB[i])/(2*1.96)
        piT_std_i = (piT_UB[i] - piT_LB[i])/(2*1.96)
        betaS_std_i = (betaS_UB[i] - betaS_LB[i])/(2*1.96)
        delta2_std_i = (delta2_UB[i] - delta2_LB[i])/(2*1.96)
        piS_std_i = (piS_UB[i] - piS_LB[i])/(2*1.96)
        ttau_std_i = (ttau_UB[i] - ttau_LB[i])/(2*1.96)
        log10TN_std_i = (log10TN_UB[i] - log10TN_LB[i])/(2*1.96)
        w_std_i = (w_UB[i] - w_LB[i])/(2*1.96)

        # Set these standard deviations equal to the mean, if it was not well-defined
        betaT_std_i = betaT_mean_i if np.isnan(betaT_std_i) else betaT_std_i
        delta1_std_i = delta1_mean_i if np.isnan(delta1_std_i) else delta1_std_i
        piT_std_i = piT_mean_i if np.isnan(piT_std_i) else piT_std_i
        betaS_std_i = betaS_mean_i if np.isnan(betaS_std_i) else betaS_std_i
        delta2_std_i = delta2_mean_i if np.isnan(delta2_std_i) else delta2_std_i
        piS_std_i = piS_mean_i if np.isnan(piS_std_i) else piS_std_i
        ttau_std_i = ttau_mean_i if np.isnan(ttau_std_i) else ttau_std_i
        log10TN_std_i = log10TN_mean_i if np.isnan(log10TN_std_i) else log10TN_std_i
        w_std_i = w_mean_i if np.isnan(w_std_i) else w_std_i

        # Draw random vars for each. Redraw if outside LB/UB.
        # If LB/UB are nan, we'll take it anyway. However, we'll force all parameters to be positive.
        for j in range(n_entriesPerInd):
            withinBounds = False
            while not withinBounds:
                betaT = np.random.normal(loc=betaT_mean_i, scale=betaT_std_i)
                withinBounds = (betaT_LB[i] <= betaT <= betaT_UB[i]) or\
                               ((np.isnan(betaT_LB[i]) or np.isnan(betaT_UB[i])) and betaT>0)

            withinBounds = False
            while not withinBounds:
                delta1 = np.random.normal(loc=delta1_mean_i, scale=delta1_std_i)
                withinBounds = (delta1_LB[i] <= delta1 <= delta1_UB[i]) or\
                               ((np.isnan(delta1_LB[i]) or np.isnan(delta1_UB[i])) and delta1>0)

            withinBounds = False
            while not withinBounds:
                piT = np.random.normal(loc=piT_mean_i, scale=piT_std_i)
                withinBounds = (piT_LB[i] <= piT <= piT_UB[i]) or\
                               ((np.isnan(piT_LB[i]) or np.isnan(piT_UB[i])) and piT>0)

            withinBounds = False
            while not withinBounds:
                betaS = np.random.normal(loc=betaS_mean_i, scale=betaS_std_i)
                withinBounds = (betaS_LB[i] <= betaS <= betaS_UB[i]) or\
                               ((np.isnan(betaS_LB[i]) or np.isnan(betaS_UB[i])) and betaS>0)

            withinBounds = False
            while not withinBounds:
                delta2 = np.random.normal(loc=delta2_mean_i, scale=delta2_std_i)
                withinBounds = (delta2_LB[i] <= delta2 <= delta2_UB[i]) or\
                               ((np.isnan(delta2_LB[i]) or np.isnan(delta2_UB[i])) and delta2>0)

            withinBounds = False
            while not withinBounds:
                piS = np.random.normal(loc=piS_mean_i, scale=piS_std_i)
                withinBounds = (piS_LB[i] <= piS <= piS_UB[i]) or\
                               ((np.isnan(piS_LB[i]) or np.isnan(piS_UB[i])) and piS>0)

            withinBounds = False
            while not withinBounds:
                ttau = np.random.normal(loc=ttau_mean_i, scale=ttau_std_i)
                withinBounds = (ttau_LB[i] <= ttau <= ttau_UB[i]) or\
                               ((np.isnan(ttau_LB[i]) or np.isnan(ttau_UB[i])) and ttau>0)

            withinBounds = False
            while not withinBounds:
                log10TN = np.random.normal(loc=log10TN_mean_i, scale=log10TN_std_i)
                withinBounds = (log10TN_LB[i] <= log10TN <= log10TN_UB[i]) or\
                               ((np.isnan(log10TN_LB[i]) or np.isnan(log10TN_UB[i])) and log10TN>0)

            withinBounds = False
            while not withinBounds:
                w = np.random.normal(loc=w_mean_i, scale=w_std_i)
                withinBounds = (w_LB[i] <= w <= w_UB[i]) or\
                               ((np.isnan(w_LB[i]) or np.isnan(w_UB[i])) and w>0)

            # Now finally have parameters we can use.
            results = results.append({'betaT': betaT,
                                      'delta1': delta1,
                                      'piT': piT,
                                      'betaS': betaS,
                                      'delta2': delta2,
                                      'piS': piS,
                                      'ttau': ttau,
                                      'log10TN': log10TN,
                                      'w': w,
                                      'k1': k1,
                                      'k2': k2,
                                      'gamma': gamma,
                                      'T10': T10,
                                      'T20': T20,
                                      'I10': I10,
                                      'c': c}, ignore_index=True)
    results.to_csv(filename, header=True, index=False)

# Probably cleaner to merge w/ other simulate_transmissions fns, but separating the logic for now.
def simulate_transmissions_WTTIP_WT_tx(dispersion, tzero, alpha, lambda_tx, theta,
                                       n_transmitters, dT, Tx_sat, hostParameterFile,
                                       rho, psi, xi, eta, delta_TIP):
    """
    Estimate number of secondary wildtype virus infections if the index patient has both wildtype virus and TIP.


    :param dispersion:  Scale parameter of gamma-distributed number of daily contacts (related to variance)
    :param tzero:  'Viral non-replication phase' duration (days); time needed to have a productively infected cell
    :param alpha: Slope in Hill function linking viral load to transmission (VL^alpha / (VL^alpha + lambda^alpha))
    :param lambda_tx: Half-max VL in Hill fn linking viral load to tx (VL^alpha / (VL^alpha + lambda^alpha))
    :param theta:  Mean number of daily contacts (the shape parameter of the gamma distribution is theta/dispersion
    :param n_transmitters:  Total number of primary transmitters to simulate
    :param dT:  Temporal resolution of viral load timecourse (days)
    :param Tx_sat:  Maximal probability of transmission (Tx_max = 1 should recover Goyal et al, if using their ODEs)
    :param hostParameterFile:  Filepath of a .txt file containing a set of individual parameters to sample from.
    :param rho: Relative TIP virion production rate from coinfected cells (vs. wildtype virion production from coinf. cells)
    :param psi: Relative wildtype virion production rate from coinfected cells (vs. wildtype virion production from singly infected cells)
    :param xi: Relative target cell infection rate for TIP versus virion.
    :param eta: Relative number of TIPs initially present versus number of wildtype virion
    :param delta_TIP: Relative death rate of infected cells that also have TIP (delta_TIP > 1, longer-lived due to TIP)
    :return results:  Pandas DataFrame with columns R0, GT, and ST corresponding to individual reproductive number,
    generation time, and serial interval. Has np.nan values in GT and ST if no secondaary infections occurred from an
    individual.
    """

    use_Goyal_ODEs = False
    use_Ke_ODEs = True

    assert(use_Goyal_ODEs != use_Ke_ODEs)

    # These are for defining the within-host viral load dynamics
    parameters = pd.read_csv(hostParameterFile)

    # Create empty dataframe to store individual R0, generation time, and serial interval
    results = pd.DataFrame(columns=['R0', 'GT', 'SI'])

    for i in range(n_transmitters):
        print(i)
        if use_Goyal_ODEs:
            print('not implemented')
            pass
            # # ODE_Y0, ODE_params = get_Goyal_ODE_setup(parameters.iloc[i])  # Draw in order
            # targind = np.random.randint(low=0, high=len(parameters))
            # ODE_Y0, ODE_params = get_Goyal_ODE_setup(parameters.iloc[targind])  # Draw randomly
            # ODE_func = Goyal_ODEs  # We'll pass the Goyal_ODEs function to simulate_replication_phase()

        elif use_Ke_ODEs:
            # ODE_Y0, ODE_params = get_Ke_ODE_setup(parameters.iloc[i])  # Draw in order
            targind = np.random.randint(low=0, high=len(parameters))
            ODE_Y0, ODE_params = get_Ke_ODE_setup(parameters.iloc[targind])  # Draw randomly
            # Unpack and then repack with additional TIP params
            beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, ttau, log10TN = ODE_params
            ODE_params = (beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, ttau, log10TN,
                          rho, psi, xi, eta, delta_TIP)
            ODE_func = Ke_ODEs_TIP  # We'll pass the Ke_ODEs function to simulate_replication_phase()

        incubation_period_infector = max((0, np.random.gamma(shape=3.45, scale=1 / 0.66)))
        incubation_period_infectee = max((0, np.random.gamma(shape=3.45, scale=1 / 0.66)))

        # How many secondary infections did this individual cause?
        # When was the first transmission event by this person?
        # Separate out by the viral nonreplicative and replication phase.
        try:
            NRP_infection_total, NRP_infection_times = simulate_nonreplication_phase(tzero, dT, ODE_Y0[2], alpha, lambda_tx,
                                                                                     theta, dispersion, Tx_sat)
            RP_infection_total, RP_infection_times = simulate_replication_phase_WTTIP_WT_tx(tzero, dT, alpha, lambda_tx,
                                                                                            theta, dispersion, ODE_Y0,
                                                                                            ODE_params, ODE_func, Tx_sat)
        except Exception as ex:
            template = "Skipping parameter set; an exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print('Transmission statistics set to np.nan')
            print(message)
            print(traceback.format_exc())

            results = results.append({'R0': np.nan, 'GT': np.nan, 'SI': np.nan}, ignore_index=True)
            continue

        R0 = NRP_infection_total + RP_infection_total

        if NRP_infection_total > 0:  # First infection occurred during non replication phase
            GT = round(NRP_infection_times[0])
            SI = round(NRP_infection_times[0] + (incubation_period_infectee - incubation_period_infector))
        elif NRP_infection_total == 0 and RP_infection_total > 0:
            GT = round(RP_infection_times[0] + tzero)
            SI = round(RP_infection_times[0] + tzero + (incubation_period_infectee - incubation_period_infector))
        elif NRP_infection_total == 0 and RP_infection_total == 0:
            GT = np.nan
            SI = np.nan
        else:
            print('error, debug me')

        results = results.append({'R0': R0, 'GT': GT, 'SI': SI}, ignore_index=True)
    return results


def simulate_transmissions_WTTIP_WTTIP_tx(dispersion, tzero, alpha, lambda_tx, theta,
                                       n_transmitters, dT, Tx_sat, hostParameterFile,
                                       rho, psi, xi, eta, delta_TIP):
    """
    Estimate number of secondary wildtype virus+TIP infections if the index patient has both wildtype virus and TIP.

    :param dispersion:  Scale parameter of gamma-distributed number of daily contacts (related to variance)
    :param tzero:  'Viral non-replication phase' duration (days); time needed to have a productively infected cell
    :param alpha: Slope in Hill function linking viral load to transmission (VL^alpha / (VL^alpha + lambda^alpha))
    :param lambda_tx: Half-max VL in Hill fn linking viral load to tx (VL^alpha / (VL^alpha + lambda^alpha))
    :param theta:  Mean number of daily contacts (the shape parameter of the gamma distribution is theta/dispersion
    :param n_transmitters:  Total number of primary transmitters to simulate
    :param dT:  Temporal resolution of viral load timecourse (days)
    :param Tx_sat:  Maximal probability of transmission (Tx_max = 1 should recover Goyal et al, if using their ODEs)
    :param hostParameterFile:  Filepath of a .txt file containing a set of individual parameters to sample from.
    :param rho: Relative TIP virion production rate from coinfected cells (vs. wildtype virion production from coinf. cells)
    :param psi: Relative wildtype virion production rate from coinfected cells (vs. wildtype virion production from singly infected cells)
    :param xi: Relative target cell infection rate for TIP versus virion.
    :param eta: Relative number of TIPs initially present versus number of wildtype virion
    :param delta_TIP: Relative death rate of infected cells that also have TIP (delta_TIP > 1, longer-lived due to TIP)
    :return results:  Pandas DataFrame with columns R0, GT, and ST corresponding to individual reproductive number,
    generation time, and serial interval. Has np.nan values in GT and ST if no secondaary infections occurred from an
    individual.
    """

    use_Goyal_ODEs = False
    use_Ke_ODEs = True

    assert (use_Goyal_ODEs != use_Ke_ODEs)

    # These are for defining the within-host viral load dynamics
    parameters = pd.read_csv(hostParameterFile)

    # Create empty dataframe to store individual R0, generation time, and serial interval
    results = pd.DataFrame(columns=['R0', 'GT', 'SI'])

    for i in range(n_transmitters):
        # print(i)
        if use_Goyal_ODEs:
            print('not implemented')
            pass
            # # ODE_Y0, ODE_params = get_Goyal_ODE_setup(parameters.iloc[i])  # Draw in order
            # targind = np.random.randint(low=0, high=len(parameters))
            # ODE_Y0, ODE_params = get_Goyal_ODE_setup(parameters.iloc[targind])  # Draw randomly
            # ODE_func = Goyal_ODEs  # We'll pass the Goyal_ODEs function to simulate_replication_phase()

        elif use_Ke_ODEs:
            # ODE_Y0, ODE_params = get_Ke_ODE_setup(parameters.iloc[i])  # Draw in order
            targind = np.random.randint(low=0, high=len(parameters))
            ODE_Y0, ODE_params = get_Ke_ODE_setup(parameters.iloc[targind])  # Draw randomly
            # Unpack and then repack with additional TIP params
            beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, ttau, log10TN = ODE_params
            ODE_params = (beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, ttau, log10TN,
                          rho, psi, xi, eta, delta_TIP)
            ODE_func = Ke_ODEs_TIP  # We'll pass the Ke_ODEs function to simulate_replication_phase()

        incubation_period_infector = max((0, np.random.gamma(shape=3.45, scale=1 / 0.66)))
        incubation_period_infectee = max((0, np.random.gamma(shape=3.45, scale=1 / 0.66)))

        # How many secondary infections did this individual cause?
        # When was the first transmission event by this person?
        # Separate out by the viral nonreplicative and replication phase.
        try:
            NRP_infection_total, NRP_infection_times = simulate_nonreplication_phase(tzero, dT, ODE_Y0[2], alpha, lambda_tx,
                                                                                     theta, dispersion, Tx_sat)
            RP_infection_total, RP_infection_times = simulate_replication_phase_WTTIP_WTTIP_tx(tzero, dT, alpha, lambda_tx,
                                                                                               theta, dispersion, ODE_Y0,
                                                                                               ODE_params, ODE_func, Tx_sat)
        except Exception as ex:
            template = "Skipping parameter set; an exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print('Transmission statistics set to np.nan')
            print(message)
            print(traceback.format_exc())

            continue

        R0 = NRP_infection_total + RP_infection_total

        if NRP_infection_total > 0:  # First infection occurred during non replication phase
            GT = round(NRP_infection_times[0])
            SI = round(NRP_infection_times[0] + (incubation_period_infectee - incubation_period_infector))
        elif NRP_infection_total == 0 and RP_infection_total > 0:
            GT = round(RP_infection_times[0] + tzero)
            SI = round(RP_infection_times[0] + tzero + (incubation_period_infectee - incubation_period_infector))
        elif NRP_infection_total == 0 and RP_infection_total == 0:
            GT = np.nan
            SI = np.nan
        else:
            print('error, debug me')

        results = results.append({'R0': R0, 'GT': GT, 'SI': SI}, ignore_index=True)
    return results

def simulate_transmissions_WTTIP_TIP_tx(dispersion, tzero, alpha, lambda_tx, theta,
                                        n_transmitters, dT, Tx_sat, hostParameterFile,
                                        rho, psi, xi, eta, delta_TIP):
    """
    Estimate number of secondary TIP infections if the index patient has both wildtype virus and TIP.

    :param dispersion:  Scale parameter of gamma-distributed number of daily contacts (related to variance)
    :param tzero:  'Viral non-replication phase' duration (days); time needed to have a productively infected cell
    :param alpha: Slope in Hill function linking viral load to transmission (VL^alpha / (VL^alpha + lambda^alpha))
    :param lambda_tx: Half-max VL in Hill fn linking viral load to tx (VL^alpha / (VL^alpha + lambda^alpha))
    :param theta:  Mean number of daily contacts (the shape parameter of the gamma distribution is theta/dispersion
    :param n_transmitters:  Total number of primary transmitters to simulate
    :param dT:  Temporal resolution of viral load timecourse (days)
    :param Tx_sat:  Maximal probability of transmission (Tx_max = 1 should recover Goyal et al, if using their ODEs)
    :param hostParameterFile:  Filepath of a .txt file containing a set of individual parameters to sample from.
    :param rho: Relative TIP virion production rate from coinfected cells (vs. wildtype virion production from coinf. cells)
    :param psi: Relative wildtype virion production rate from coinfected cells (vs. wildtype virion production from singly infected cells)
    :param xi: Relative target cell infection rate for TIP versus virion.
    :param eta: Relative number of TIPs initially present versus number of wildtype virion
    :param delta_TIP: Relative death rate of infected cells that also have TIP (delta_TIP > 1, longer-lived due to TIP)
    :return results:  Pandas DataFrame with columns R0, GT, and ST corresponding to individual reproductive number,
    generation time, and serial interval. Has np.nan values in GT and ST if no secondaary infections occurred from an
    individual.
    """

    use_Goyal_ODEs = False
    use_Ke_ODEs = True

    assert (use_Goyal_ODEs != use_Ke_ODEs)

    # These are for defining the within-host viral load dynamics
    parameters = pd.read_csv(hostParameterFile)

    # Create empty dataframe to store individual R0, generation time, and serial interval
    results = pd.DataFrame(columns=['R0', 'GT', 'SI'])

    for i in range(n_transmitters):
        # print(i)
        if use_Goyal_ODEs:
            print('not implemented')
            pass
            # # ODE_Y0, ODE_params = get_Goyal_ODE_setup(parameters.iloc[i])  # Draw in order
            # targind = np.random.randint(low=0, high=len(parameters))
            # ODE_Y0, ODE_params = get_Goyal_ODE_setup(parameters.iloc[targind])  # Draw randomly
            # ODE_func = Goyal_ODEs  # We'll pass the Goyal_ODEs function to simulate_replication_phase()

        elif use_Ke_ODEs:
            # ODE_Y0, ODE_params = get_Ke_ODE_setup(parameters.iloc[i])  # Draw in order
            targind = np.random.randint(low=0, high=len(parameters))
            ODE_Y0, ODE_params = get_Ke_ODE_setup(parameters.iloc[targind])  # Draw randomly
            # Unpack and then repack with additional TIP params
            beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, ttau, log10TN = ODE_params
            ODE_params = (beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, ttau, log10TN,
                          rho, psi, xi, eta, delta_TIP)
            ODE_func = Ke_ODEs_TIP  # We'll pass the Ke_ODEs function to simulate_replication_phase()

        incubation_period_infector = max((0, np.random.gamma(shape=3.45, scale=1 / 0.66)))
        incubation_period_infectee = max((0, np.random.gamma(shape=3.45, scale=1 / 0.66)))

        # How many secondary infections did this individual cause?
        # When was the first transmission event by this person?
        # Separate out by the viral nonreplicative and replication phase.
        try:
            NRP_infection_total, NRP_infection_times = simulate_nonreplication_phase(tzero, dT, ODE_Y0[2], alpha, lambda_tx,
                                                                                     theta, dispersion, Tx_sat)
            RP_infection_total, RP_infection_times = simulate_replication_phase_WTTIP_TIP_tx(tzero, dT, alpha, lambda_tx,
                                                                                             theta, dispersion, ODE_Y0,
                                                                                             ODE_params, ODE_func, Tx_sat)
        except Exception as ex:
            template = "Skipping parameter set; an exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print('Transmission statistics set to np.nan')
            print(message)
            print(traceback.format_exc())

            results = results.append({'R0': np.nan, 'GT': np.nan, 'SI': np.nan}, ignore_index=True)
            continue

        R0 = NRP_infection_total + RP_infection_total

        if NRP_infection_total > 0:  # First infection occurred during non replication phase
            GT = round(NRP_infection_times[0])
            SI = round(NRP_infection_times[0] + (incubation_period_infectee - incubation_period_infector))
        elif NRP_infection_total == 0 and RP_infection_total > 0:
            GT = round(RP_infection_times[0] + tzero)
            SI = round(RP_infection_times[0] + tzero + (incubation_period_infectee - incubation_period_infector))
        elif NRP_infection_total == 0 and RP_infection_total == 0:
            GT = np.nan
            SI = np.nan
        else:
            print('error, debug me')

        results = results.append({'R0': R0, 'GT': GT, 'SI': SI}, ignore_index=True)
    return results

def simulate_nonreplication_phase(tzero, dT, V_0, alpha, lambda_tx, theta, dispersion, Tx_sat):
    """
    Simulates infections that might occur (anomalously) during the nonreplication phase.
    :param tzero: Non-replication phase duration
    :param dT: Time resolution of viral dynamics
    :param alpha: Slope for calculating VL to infectiousness curve
    :param lambda_tx: Viral load at half-maximal infectiousness
    :param theta: Parameter for number of contacts
    :param dispersion: Parameters for variance of number of contacts
    :param Tx_sat: Saturating value for the infectiousness curve.
    :return: total_new_infections, time_new_infections
    """
    # Set up discretized timepoints for NRPDI.
    NRPDI_discrete = np.arange(0, tzero, dT)

    # Viral loads before the first infected cell are assumed to be V_0
    VL = V_0 * np.ones(len(NRPDI_discrete))

    # Calculate infectiousness based on viral load.
    VL = np.maximum(0, VL)  # in case V_0 wound up negative from weird numerics issues
    Prob_V = Tx_sat * VL ** alpha / (lambda_tx ** alpha + VL ** alpha)

    # Calculate expected number of contacts during the nonreplication phase
    n_daily_contacts = np.random.gamma(shape=theta / dispersion, scale=dispersion,
                                       size=(len(NRPDI_discrete)))

    # check for successful transmissions - does a tx event happen, and how many people are infected?
    RV = np.random.random(len(Prob_V))

    # maximally, new infections =  dT * n_daily_contacts * Prob_V_NRPDI
    n_new_infections = dT * n_daily_contacts[Prob_V > RV] * Prob_V[Prob_V > RV]
    time_new_infections = NRPDI_discrete[Prob_V > RV]

    # Track number of new infections that occur during NRPDI period.
    total_new_infections = round(sum(n_new_infections))

    return total_new_infections, time_new_infections

# Would probably be more efficient to do in one code block, but first
# compute separately as a sanity check
def simulate_replication_phase_WTTIP_WT_tx(tzero, dT, alpha, lambda_tx, theta, dispersion, ODE_Y0, ODE_params, ODE_func, Tx_sat):
    """
    Model transmissions of WT virus alone by WT+TIP coinfected individuals
    Simulates population transmission during viral phase replication. Uses either the Goyal et al (Schiffer)
    ODEs for calculating within-host viral dynamics, or the Ke et al ODEs.
    :param tzero: Non-replication phase duration
    :param dT: Time resolution of viral dynamics
    :param alpha: Slope for calculating VL to infectiousness curve
    :param lambda_tx: Viral load at half-maximal infectiousness
    :param theta: Parameter for number of contacts
    :param dispersion: Parameters for variance of number of contacts
    :param ODE_Y0: Initial conditions for within-host model
    :param ODE_params: Parameters for within-host model
    :param ODE_func: Function handle for within-host model
    :param Tx_sat: Saturating value for the infectiousness curve.
    :return: total_new_infections, time_new_infections
    """
    tf = tzero + 20  # Max at 16 for now -- Ke ODEs seem unstable?
    if ODE_func.__name__ == 'Goyal_ODEs':
        print('not implemented')
        pass
        # ODE_t = np.arange(tzero, tf, dT)
        # sol = solve_ivp(lambda t, y: ODE_func(t, y, ODE_params),
        #                 [ODE_t[0], ODE_t[-1]], ODE_Y0, t_eval=ODE_t)
        #
        # if not sol.success:
        #     print('Integration failed, debug me!')
        #
        # # Assume that the viral load should be (on average) zero after 20 days.
        # # This is a pretty strong assumption to make, given that they calculate tx events going out 30 days..
        # sol.y[2, sol.t > (tzero + 20)] = 0
        # V_tmp = np.maximum(sol.y[2, :], 0)
        # times = sol.t

    elif ODE_func.__name__ == 'Ke_ODEs_TIP':
        # I started using the Radau solver for the Ke ODEs because I've had some numerics problems.
        beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, t_tau, log10TN,\
        rho, psi, xi, eta, delta_TIP = ODE_params
        T10, E10, I10, VT0, T20, E20, I20, VS0 = ODE_Y0
        Ke_ODE_params = (beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, rho, psi, xi, delta_TIP)

        # Assume pretreatment reaching 50% of patient cells. Also assume that we're setting
        # VT0, rather than I10
        fractionTIP = 0.5
        TIP_converted_naive_cells_URT = T10 * fractionTIP
        T10 = T10 - TIP_converted_naive_cells_URT
        T1TIP0 = TIP_converted_naive_cells_URT
        TIP_converted_naive_cells_LRT = T20 * fractionTIP
        T20 = T20 - TIP_converted_naive_cells_LRT
        T2TIP0 = TIP_converted_naive_cells_LRT
        VT0 = I10 # note that this is not effected by psi since we assume a pre-treated ind. gets infected by wt
        first_Y0 = [T10, 0, 0, VT0, T1TIP0, 0, 0, 0, T20, 0, 0, 0, T2TIP0, 0, 0, 0]

        # first_Y0 = [T10, 0, I10, 0, T20, 0, 0, 0]
        # This assumed TIPs create 'target cells in URT infected w/ TIP'.
        # first_Y0 = [T10, 0, I10*psi, 0, I10*eta, 0, 0, 0, T20, 0, 0, 0, 0, 0, 0, 0]
        # Try having 1D cells, (coinfected cells producing wildtype virus and TIP)
        # first_Y0 = [T10, 0, I10 * psi, 0, I10 * rho, 0, I10 * rho, 0, T20, 0, 0, 0, 0, 0, 0, 0]

        first_time = np.append(np.arange(tzero, t_tau), t_tau)
        first_time = np.unique(first_time)  # in case t_tau was an integer; will drop duplicated final point

        # If first_time only has one element, we should skip to target cell extension
        if len(first_time)>1:
            sol = solve_ivp(lambda t, y: ODE_func(t, y, Ke_ODE_params),
                            [first_time[0], first_time[-1]], first_Y0, t_eval=first_time, method='Radau')

            if not sol.success:
                print('Integration failed on first step, pre-target cell extension. params:', Ke_ODE_params)
                raise Exception('Integration failure during pre-target cell extension in Ke ODEs')

            # Set up initial conditions for second step.
            if tf > np.ceil(t_tau+1):
                second_Y0 = sol.y[:, -1]
                #second_Y0[4] += 10 ** log10TN  # target cell extension in LRT

                # # In the TIP ODEs, the LRT target cells are index 8.
                # second_Y0[8] += 10 ** log10TN  # target cell extension in LRT

                # Extension with proportion of preconverted cells
                TCE_LRT = 10 ** log10TN
                TCE_LRT_preconverted = TCE_LRT * fractionTIP
                TCE_LRT -= TCE_LRT_preconverted
                second_Y0[8] += TCE_LRT  # target cell extension in LRT
                second_Y0[12] += TCE_LRT_preconverted

                second_time = np.append(t_tau, np.arange(np.ceil(t_tau), tf))
                second_time = np.unique(second_time)  # in case t_tau was an integer; will drop duplicated initial point
                sol2 = solve_ivp(lambda t, y: ODE_func(t, y, Ke_ODE_params),
                                 [second_time[0], second_time[-1]], second_Y0, t_eval=second_time, method='Radau')

                if not sol2.success:
                    print('Integration failed on second step, post-target cell extension. params:', Ke_ODE_params)
                    raise Exception('Integration failure during pre-target cell extension in Ke ODEs')
                try:
                    # Stitch results together, keeping only the integer timepoints.
                    all_times = np.append(sol.t, sol2.t)
                    index_to_keep = np.equal(np.mod(all_times, 1), 0)
                    all_sol = np.concatenate((sol.y, sol2.y), axis=1)  # Append along time axis

                    # Final results
                    times = all_times[index_to_keep]
                    solution = all_sol[:, index_to_keep]

                    # We'll focus on throat swab viral load
                    VT = solution[3]
                    VT_TIP = solution[7]
                    V_tmp = np.maximum(VT, 0)
                    VTIP_tmp = np.maximum(VT_TIP, 0)
                except:
                    print('test')
            else:
                index_to_keep = np.equal(np.mod(sol.t, 1), 0)
                times = sol.t[index_to_keep]
                solution = sol.y[:, index_to_keep]
                VT = solution[3]
                VT_TIP = solution[7]
                V_tmp = np.maximum(VT, 0)
                VTIP_tmp = np.maximum(VT_TIP, 0)
        else:  # first_time had 1 or fewer elements. we start from post-extension.
            if tf > np.ceil(t_tau + 1):
                #second_Y0 = [T10, 0, I10, 0, T20 + 10 ** log10TN, 0, 0, 0]
                # This assumed TIPs create 'target cells in URT infected w/ TIP'.
                # second_Y0 = [T10, 0, I10*psi, 0, I10*eta, 0, 0, 0, T20, 0, 0, 0, 0, 0, 0, 0]
                # Try having 1D cells, (coinfected cells producing wildtype virus and TIP)
                # second_Y0 = [T10, 0, I10 * psi, 0, I10 * rho, 0, I10 * rho, 0, T20, 0, 0, 0, 0, 0, 0, 0]

                # TIP pretreatment
                second_Y0 = [T10, 0, 0, VT0, T1TIP0, 0, 0, 0, T20, 0, 0, 0, T2TIP0, 0, 0, 0]
                TCE_LRT = 10 ** log10TN
                TCE_LRT_preconverted = TCE_LRT * fractionTIP
                TCE_LRT -= TCE_LRT_preconverted
                second_Y0[8] += TCE_LRT  # target cell extension in LRT
                second_Y0[12] += TCE_LRT_preconverted


                second_time = np.append(tzero, np.arange(np.ceil(tzero), tf))
                second_time = np.unique(second_time)  # in case t_tau was an integer; will drop duplicated initial point
                sol2 = solve_ivp(lambda t, y: ODE_func(t, y, Ke_ODE_params),
                                 [second_time[0], second_time[-1]], second_Y0, t_eval=second_time, method='Radau')

                if not sol2.success:
                    print('Integration failed on second step, post-target cell extension. params:', Ke_ODE_params)
                    raise Exception('Integration failure during pre-target cell extension in Ke ODEs')

                try:
                    # Stitch results together, keeping only the integer timepoints.
                    all_times = sol2.t
                    index_to_keep = np.equal(np.mod(all_times, 1), 0)
                    all_sol = sol2.y  # Append along time axis

                    # Final results
                    times = all_times[index_to_keep]
                    solution = all_sol[:, index_to_keep]

                    # We'll focus on throat swab viral load
                    VT = solution[3]
                    VT_TIP = solution[7]
                    V_tmp = np.maximum(VT, 0)
                    VTIP_tmp = np.maximum(VT_TIP, 0)

                except:
                    print('test')

            else:  # the simulation time is less than t_tau
                raise Exception('Attempted a too-short simulation: pre-target cell expansion was only one step, but expansion point is after final desired timepoint.')

    # time_V_final = sol.t  # No need to know <when> new infs happened here.
    # Calculate the rest of the infectiousness dynamics by viral load.
    # Prob_V = Tx_sat * V_tmp ** alpha / (lambda_tx ** alpha + V_tmp ** alpha)
    cutlog10 = lambda x: np.log10(x) if x > 0 else 0
    # Prob_V = Tx_sat * cutlog10(VL) ** alpha / (lambda_tx ** alpha + cutlog10(VL) ** alpha)
    Prob_V = [Tx_sat * cutlog10(vl) ** alpha / (lambda_tx ** alpha + cutlog10(vl) ** alpha) for vl in V_tmp]
    Prob_V = np.array(Prob_V)

    # Draw daily contacts for time points. Be careful w/ integration failure --> wrong-length vectors...
    n_daily_contacts = np.random.gamma(shape=theta / dispersion, scale=dispersion, size=(len(times)))

    # check for successful transmissions - does a tx event happen, and how many people are infected?
    RV = np.random.random(len(Prob_V))
    n_new_infections = dT * n_daily_contacts * Prob_V  # n_new_infections = n_new_infections_fullvec[Prob_V>RV]

    # Get timings of new infections
    time_new_infections = times[(Prob_V > RV) & (n_new_infections > 0)]

    # Subset only the relevant days of the new infections
    n_new_infections = n_new_infections[Prob_V > RV]

    # Track number of new infections that occur during NRPDI period.
    total_new_infections = round(sum(n_new_infections))

    return total_new_infections, time_new_infections

def simulate_replication_phase_WTTIP_WTTIP_tx(tzero, dT, alpha, lambda_tx, theta, dispersion, ODE_Y0, ODE_params, ODE_func, Tx_sat):
    """
    Model transmissions of WT virus +TIP  by WT+TIP coinfected individuals
    Simulates population transmission during viral phase replication. Uses either the Goyal et al (Schiffer)
    ODEs for calculating within-host viral dynamics, or the Ke et al ODEs.
    :param tzero: Non-replication phase duration
    :param dT: Time resolution of viral dynamics
    :param alpha: Slope for calculating VL to infectiousness curve
    :param lambda_tx: Viral load at half-maximal infectiousness
    :param theta: Parameter for number of contacts
    :param dispersion: Parameters for variance of number of contacts
    :param ODE_Y0: Initial conditions for within-host model
    :param ODE_params: Parameters for within-host model
    :param ODE_func: Function handle for within-host model
    :param Tx_sat: Saturating value for the infectiousness curve.
    :return: total_new_infections, time_new_infections
    """
    tf = tzero + 20  # Max at 16 for now -- Ke ODEs seem unstable?
    if ODE_func.__name__ == 'Goyal_ODEs':
        print('not implemented')
        pass
        # ODE_t = np.arange(tzero, tf, dT)
        # sol = solve_ivp(lambda t, y: ODE_func(t, y, ODE_params),
        #                 [ODE_t[0], ODE_t[-1]], ODE_Y0, t_eval=ODE_t)
        #
        # if not sol.success:
        #     print('Integration failed, debug me!')
        #
        # # Assume that the viral load should be (on average) zero after 20 days.
        # # This is a pretty strong assumption to make, given that they calculate tx events going out 30 days..
        # sol.y[2, sol.t > (tzero + 20)] = 0
        # V_tmp = np.maximum(sol.y[2, :], 0)
        # times = sol.t

    elif ODE_func.__name__ == 'Ke_ODEs_TIP':
        # I started using the Radau solver for the Ke ODEs because I've had some numerics problems.
        beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, t_tau, log10TN,\
        rho, psi, xi, eta, delta_TIP = ODE_params
        T10, E10, I10, VT0, T20, E20, I20, VS0 = ODE_Y0
        Ke_ODE_params = (beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, rho, psi, xi, delta_TIP)
        # Assume pretreatment reaching 50% of patient cells. Also assume that we're setting
        # VT0, rather than I10
        fractionTIP = 0.5
        TIP_converted_naive_cells_URT = T10 * fractionTIP
        T10 = T10 - TIP_converted_naive_cells_URT
        T1TIP0 = TIP_converted_naive_cells_URT
        TIP_converted_naive_cells_LRT = T20 * fractionTIP
        T20 = T20 - TIP_converted_naive_cells_LRT
        T2TIP0 = TIP_converted_naive_cells_LRT
        VT0 = I10 # note that this is not effected by psi since we assume a pre-treated ind. gets infected by wt
        first_Y0 = [T10, 0, 0, VT0, T1TIP0, 0, 0, 0, T20, 0, 0, 0, T2TIP0, 0, 0, 0]

        #first_Y0 = [T10, 0, I10, 0, T20, 0, 0, 0]
        # This assumed TIPs create 'target cells in URT infected w/ TIP'.
        # first_Y0 = [T10, 0, I10*psi, 0, I10*eta, 0, 0, 0, T20, 0, 0, 0, 0, 0, 0, 0]
        # Try having 1D cells, (coinfected cells producing wildtype virus and TIP)
        # first_Y0 = [T10, 0, I10*psi, 0, I10*rho, 0, I10*rho, 0, T20, 0, 0, 0, 0, 0, 0, 0]

        first_time = np.append(np.arange(tzero, t_tau), t_tau)
        first_time = np.unique(first_time)  # in case t_tau was an integer; will drop duplicated final point

        # If first_time only has one element, we should skip to target cell extension
        if len(first_time)>1:
            sol = solve_ivp(lambda t, y: ODE_func(t, y, Ke_ODE_params),
                            [first_time[0], first_time[-1]], first_Y0, t_eval=first_time, method='Radau')

            if not sol.success:
                print('Integration failed on first step, pre-target cell extension. params:', Ke_ODE_params)
                raise Exception('Integration failure during pre-target cell extension in Ke ODEs')

            # Set up initial conditions for second step.
            if tf > np.ceil(t_tau+1):
                second_Y0 = sol.y[:, -1]
                # second_Y0[8] += 10 ** log10TN  # target cell extension in LRT

                # Extension with proportion of preconverted cells
                TCE_LRT = 10 ** log10TN
                TCE_LRT_preconverted = TCE_LRT * fractionTIP
                TCE_LRT -= TCE_LRT_preconverted
                second_Y0[8] += TCE_LRT  # target cell extension in LRT
                second_Y0[12] += TCE_LRT_preconverted


                second_time = np.append(t_tau, np.arange(np.ceil(t_tau), tf))
                second_time = np.unique(second_time)  # in case t_tau was an integer; will drop duplicated initial point
                sol2 = solve_ivp(lambda t, y: ODE_func(t, y, Ke_ODE_params),
                                 [second_time[0], second_time[-1]], second_Y0, t_eval=second_time, method='Radau')

                if not sol2.success:
                    print('Integration failed on second step, post-target cell extension. params:', Ke_ODE_params)
                    raise Exception('Integration failure during pre-target cell extension in Ke ODEs')
                try:
                    # Stitch results together, keeping only the integer timepoints.
                    all_times = np.append(sol.t, sol2.t)
                    index_to_keep = np.equal(np.mod(all_times, 1), 0)
                    all_sol = np.concatenate((sol.y, sol2.y), axis=1)  # Append along time axis

                    # Final results
                    times = all_times[index_to_keep]
                    solution = all_sol[:, index_to_keep]

                    # We'll focus on throat swab viral load
                    VT = solution[3]
                    VT_TIP = solution[7]
                    V_tmp = np.maximum(VT, 0)
                    VTIP_tmp = np.maximum(VT_TIP, 0)
                except:
                    print('test')
            else:
                index_to_keep = np.equal(np.mod(sol.t, 1), 0)
                times = sol.t[index_to_keep]
                solution = sol.y[:, index_to_keep]
                VT = solution[3]
                VT_TIP = solution[7]
                V_tmp = np.maximum(VT, 0)
                VTIP_tmp = np.maximum(VT_TIP, 0)
        else:  # first_time had 1 or fewer elements. we start from post-extension.
            if tf > np.ceil(t_tau + 1):
                #second_Y0 = [T10, 0, I10, 0, T20 + 10 ** log10TN, 0, 0, 0]
                # This assumed TIPs create 'target cells in URT infected w/ TIP'.
                # second_Y0 = [T10, 0, I10*psi, 0, I10*eta, 0, 0, 0, T20, 0, 0, 0, 0, 0, 0, 0]
                # Try having 1D cells, (coinfected cells producing wildtype virus and TIP)
                # second_Y0 = [T10, 0, I10 * psi, 0, I10*rho, 0, I10 * rho, 0, T20, 0, 0, 0, 0, 0, 0, 0]

                # TIP pretreatment
                second_Y0 = [T10, 0, 0, VT0, T1TIP0, 0, 0, 0, T20, 0, 0, 0, T2TIP0, 0, 0, 0]
                TCE_LRT = 10 ** log10TN
                TCE_LRT_preconverted = TCE_LRT * fractionTIP
                TCE_LRT -= TCE_LRT_preconverted
                second_Y0[8] += TCE_LRT  # target cell extension in LRT
                second_Y0[12] += TCE_LRT_preconverted


                second_time = np.append(tzero, np.arange(np.ceil(tzero), tf))
                second_time = np.unique(second_time)  # in case t_tau was an integer; will drop duplicated initial point
                sol2 = solve_ivp(lambda t, y: ODE_func(t, y, Ke_ODE_params),
                                 [second_time[0], second_time[-1]], second_Y0, t_eval=second_time, method='Radau')

                if not sol2.success:
                    print('Integration failed on second step, post-target cell extension. params:', Ke_ODE_params)
                    raise Exception('Integration failure during pre-target cell extension in Ke ODEs')

                try:
                    # Stitch results together, keeping only the integer timepoints.
                    all_times = sol2.t
                    index_to_keep = np.equal(np.mod(all_times, 1), 0)
                    all_sol = sol2.y  # Append along time axis

                    # Final results
                    times = all_times[index_to_keep]
                    solution = all_sol[:, index_to_keep]

                    # We'll focus on throat swab viral load
                    VT = solution[3]
                    VT_TIP = solution[7]
                    V_tmp = np.maximum(VT, 0)
                    VTIP_tmp = np.maximum(VT_TIP, 0)

                except:
                    print('test')

            else:  # the simulation time is less than t_tau
                raise Exception('Attempted a too-short simulation: pre-target cell expansion was only one step, but expansion point is after final desired timepoint.')

    # time_V_final = sol.t  # No need to know <when> new infs happened here.
    # Calculate the rest of the infectiousness dynamics by viral load.
    # Prob_V = Tx_sat * V_tmp ** alpha / (lambda_tx ** alpha + V_tmp ** alpha)
    # Prob_VTIP = Tx_sat * VTIP_tmp ** alpha / (lambda_tx ** alpha + VTIP_tmp ** alpha)

    cutlog10 = lambda x: np.log10(x) if x > 0 else 0
    # Prob_V = Tx_sat * cutlog10(VL) ** alpha / (lambda_tx ** alpha + cutlog10(VL) ** alpha)
    Prob_V = [Tx_sat * cutlog10(vl) ** alpha / (lambda_tx ** alpha + cutlog10(vl) ** alpha) for vl in V_tmp]
    Prob_V = np.array(Prob_V)

    Prob_VTIP = [Tx_sat * cutlog10(vl) ** alpha / (lambda_tx ** alpha + cutlog10(vl) ** alpha) for vl in VTIP_tmp]
    Prob_VTIP = np.array(Prob_VTIP)

    Prob_DualTx = Prob_V * Prob_VTIP

    # Draw daily contacts for time points. Be careful w/ integration failure --> wrong-length vectors...
    n_daily_contacts = np.random.gamma(shape=theta / dispersion, scale=dispersion, size=(len(times)))

    # check for successful transmissions - does a tx event happen, and how many people are infected?
    RV = np.random.random(len(Prob_DualTx))
    n_new_infections = dT * n_daily_contacts * Prob_DualTx  # n_new_infections = n_new_infections_fullvec[Prob_V>RV]

    # Get timings of new infections
    time_new_infections = times[(Prob_DualTx > RV) & (n_new_infections > 0)]

    # Subset only the relevant days of the new infections
    n_new_infections = n_new_infections[Prob_DualTx > RV]

    # Track number of new infections that occur during NRPDI period.
    total_new_infections = round(sum(n_new_infections))

    return total_new_infections, time_new_infections

def simulate_replication_phase_WTTIP_TIP_tx(tzero, dT, alpha, lambda_tx, theta, dispersion, ODE_Y0, ODE_params, ODE_func, Tx_sat):
    """
    Model transmissions of TIP  by WT+TIP coinfected individuals
    Simulates population transmission during viral phase replication. Uses either the Goyal et al (Schiffer)
    ODEs for calculating within-host viral dynamics, or the Ke et al ODEs.
    :param tzero: Non-replication phase duration
    :param dT: Time resolution of viral dynamics
    :param alpha: Slope for calculating VL to infectiousness curve
    :param lambda_tx: Viral load at half-maximal infectiousness
    :param theta: Parameter for number of contacts
    :param dispersion: Parameters for variance of number of contacts
    :param ODE_Y0: Initial conditions for within-host model
    :param ODE_params: Parameters for within-host model
    :param ODE_func: Function handle for within-host model
    :param Tx_sat: Saturating value for the infectiousness curve.
    :return: total_new_infections, time_new_infections
    """
    tf = tzero + 20  # Max at 16 for now -- Ke ODEs seem unstable?
    if ODE_func.__name__ == 'Goyal_ODEs':
        print('not implemented')
        pass
        # ODE_t = np.arange(tzero, tf, dT)
        # sol = solve_ivp(lambda t, y: ODE_func(t, y, ODE_params),
        #                 [ODE_t[0], ODE_t[-1]], ODE_Y0, t_eval=ODE_t)
        #
        # if not sol.success:
        #     print('Integration failed, debug me!')
        #
        # # Assume that the viral load should be (on average) zero after 20 days.
        # # This is a pretty strong assumption to make, given that they calculate tx events going out 30 days..
        # sol.y[2, sol.t > (tzero + 20)] = 0
        # V_tmp = np.maximum(sol.y[2, :], 0)
        # times = sol.t

    elif ODE_func.__name__ == 'Ke_ODEs_TIP':
        # I started using the Radau solver for the Ke ODEs because I've had some numerics problems.
        beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, t_tau, log10TN,\
        rho, psi, xi, eta, delta_TIP = ODE_params
        T10, E10, I10, VT0, T20, E20, I20, VS0 = ODE_Y0
        Ke_ODE_params = (beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, rho, psi, xi, delta_TIP)
        #first_Y0 = [T10, 0, I10, 0, T20, 0, 0, 0]
        # This assumed TIPs create 'target cells in URT infected w/ TIP'.
        # first_Y0 = [T10, 0, I10*psi, 0, I10*eta, 0, 0, 0, T20, 0, 0, 0, 0, 0, 0, 0]
        # Try having 1D cells, (coinfected cells producing wildtype virus and TIP)
        # first_Y0 = [T10, 0, I10*psi, 0, I10 * rho, 0, I10*rho, 0, T20, 0, 0, 0, 0, 0, 0, 0]

        # Assume pretreatment reaching 50% of patient cells. Also assume that we're setting
        # VT0, rather than I10
        fractionTIP = 0.5
        TIP_converted_naive_cells_URT = T10 * fractionTIP
        T10 = T10 - TIP_converted_naive_cells_URT
        T1TIP0 = TIP_converted_naive_cells_URT
        TIP_converted_naive_cells_LRT = T20 * fractionTIP
        T20 = T20 - TIP_converted_naive_cells_LRT
        T2TIP0 = TIP_converted_naive_cells_LRT
        VT0 = I10  # note that this is not effected by psi since we assume a pre-treated ind. gets infected by wt
        first_Y0 = [T10, 0, 0, VT0, T1TIP0, 0, 0, 0, T20, 0, 0, 0, T2TIP0, 0, 0, 0]

        first_time = np.append(np.arange(tzero, t_tau), t_tau)
        first_time = np.unique(first_time)  # in case t_tau was an integer; will drop duplicated final point

        # If first_time only has one element, we should skip to target cell extension
        if len(first_time)>1:
            sol = solve_ivp(lambda t, y: ODE_func(t, y, Ke_ODE_params),
                            [first_time[0], first_time[-1]], first_Y0, t_eval=first_time, method='Radau')

            if not sol.success:
                print('Integration failed on first step, pre-target cell extension. params:', Ke_ODE_params)
                raise Exception('Integration failure during pre-target cell extension in Ke ODEs')

            # Set up initial conditions for second step.
            if tf > np.ceil(t_tau+1):
                second_Y0 = sol.y[:, -1]
                # second_Y0[8] += 10 ** log10TN  # target cell extension in LRT

                # Extension with proportion of preconverted cells
                TCE_LRT = 10 ** log10TN
                TCE_LRT_preconverted = TCE_LRT * fractionTIP
                TCE_LRT -= TCE_LRT_preconverted
                second_Y0[8] += TCE_LRT  # target cell extension in LRT
                second_Y0[12] += TCE_LRT_preconverted


                second_time = np.append(t_tau, np.arange(np.ceil(t_tau), tf))
                second_time = np.unique(second_time)  # in case t_tau was an integer; will drop duplicated initial point
                sol2 = solve_ivp(lambda t, y: ODE_func(t, y, Ke_ODE_params),
                                 [second_time[0], second_time[-1]], second_Y0, t_eval=second_time, method='Radau')

                if not sol2.success:
                    print('Integration failed on second step, post-target cell extension. params:', Ke_ODE_params)
                    raise Exception('Integration failure during pre-target cell extension in Ke ODEs')
                try:
                    # Stitch results together, keeping only the integer timepoints.
                    all_times = np.append(sol.t, sol2.t)
                    index_to_keep = np.equal(np.mod(all_times, 1), 0)
                    all_sol = np.concatenate((sol.y, sol2.y), axis=1)  # Append along time axis

                    # Final results
                    times = all_times[index_to_keep]
                    solution = all_sol[:, index_to_keep]

                    # We'll focus on throat swab viral load
                    VT = solution[3]
                    VT_TIP = solution[7]
                    V_tmp = np.maximum(VT, 0)
                    VTIP_tmp = np.maximum(VT_TIP, 0)
                except:
                    print('test')
            else:
                index_to_keep = np.equal(np.mod(sol.t, 1), 0)
                times = sol.t[index_to_keep]
                solution = sol.y[:, index_to_keep]
                VT = solution[3]
                VT_TIP = solution[7]
                V_tmp = np.maximum(VT, 0)
                VTIP_tmp = np.maximum(VT_TIP, 0)
        else:  # first_time had 1 or fewer elements. we start from post-extension.
            if tf > np.ceil(t_tau + 1):
                #second_Y0 = [T10, 0, I10, 0, T20 + 10 ** log10TN, 0, 0, 0]
                # This assumed TIPs create 'target cells in URT infected w/ TIP'.
                # second_Y0 = [T10, 0, I10*psi, 0, I10*eta, 0, 0, 0, T20, 0, 0, 0, 0, 0, 0, 0]
                # Try having 1D cells, (coinfected cells producing wildtype virus and TIP)
                # second_Y0 = [T10, 0, I10 * psi, 0, I10 * rho, 0, I10 * rho, 0, T20, 0, 0, 0, 0, 0, 0, 0]

                # TIP pretreatment
                second_Y0 = [T10, 0, 0, VT0, T1TIP0, 0, 0, 0, T20, 0, 0, 0, T2TIP0, 0, 0, 0]
                TCE_LRT = 10 ** log10TN
                TCE_LRT_preconverted = TCE_LRT * fractionTIP
                TCE_LRT -= TCE_LRT_preconverted
                second_Y0[8] += TCE_LRT  # target cell extension in LRT
                second_Y0[12] += TCE_LRT_preconverted

                second_time = np.append(tzero, np.arange(np.ceil(tzero), tf))
                second_time = np.unique(second_time)  # in case t_tau was an integer; will drop duplicated initial point
                sol2 = solve_ivp(lambda t, y: ODE_func(t, y, Ke_ODE_params),
                                 [second_time[0], second_time[-1]], second_Y0, t_eval=second_time, method='Radau')

                if not sol2.success:
                    print('Integration failed on second step, post-target cell extension. params:', Ke_ODE_params)
                    raise Exception('Integration failure during pre-target cell extension in Ke ODEs')

                try:
                    # Stitch results together, keeping only the integer timepoints.
                    all_times = sol2.t
                    index_to_keep = np.equal(np.mod(all_times, 1), 0)
                    all_sol = sol2.y  # Append along time axis

                    # Final results
                    times = all_times[index_to_keep]
                    solution = all_sol[:, index_to_keep]

                    # We'll focus on throat swab viral load
                    VT = solution[3]
                    VT_TIP = solution[7]
                    V_tmp = np.maximum(VT, 0)
                    VTIP_tmp = np.maximum(VT_TIP, 0)

                except:
                    print('test')

            else:  # the simulation time is less than t_tau
                raise Exception('Attempted a too-short simulation: pre-target cell expansion was only one step, but expansion point is after final desired timepoint.')

    # time_V_final = sol.t  # No need to know <when> new infs happened here.
    # Calculate the rest of the infectiousness dynamics by viral load.
    #Prob_V = Tx_sat * V_tmp ** alpha / (lambda_tx ** alpha + V_tmp ** alpha)
    # Prob_VTIP = Tx_sat * VTIP_tmp ** alpha / (lambda_tx ** alpha + VTIP_tmp ** alpha)

    cutlog10 = lambda x: np.log10(x) if x > 0 else 0
    # Prob_V = Tx_sat * cutlog10(VL) ** alpha / (lambda_tx ** alpha + cutlog10(VL) ** alpha)
    Prob_VTIP = [Tx_sat * cutlog10(vl) ** alpha / (lambda_tx ** alpha + cutlog10(vl) ** alpha) for vl in VTIP_tmp]
    Prob_VTIP = np.array(Prob_VTIP)

    # Draw daily contacts for time points. Be careful w/ integration failure --> wrong-length vectors...
    n_daily_contacts = np.random.gamma(shape=theta / dispersion, scale=dispersion, size=(len(times)))

    # check for successful transmissions - does a tx event happen, and how many people are infected?
    RV = np.random.random(len(Prob_VTIP))
    n_new_infections = dT * n_daily_contacts * Prob_VTIP  # n_new_infections = n_new_infections_fullvec[Prob_V>RV]

    # Get timings of new infections
    time_new_infections = times[(Prob_VTIP > RV) & (n_new_infections > 0)]

    # Subset only the relevant days of the new infections
    n_new_infections = n_new_infections[Prob_VTIP > RV]

    # Track number of new infections that occur during NRPDI period.
    total_new_infections = round(sum(n_new_infections))

    return total_new_infections, time_new_infections

def get_Goyal_ODE_setup(parameter_df_entry):
    # Take in a row of a dataframe, and return parameter and Y0 used for ODE simulation
    beta = parameter_df_entry['beta']
    delta = parameter_df_entry['delta']
    k = parameter_df_entry['k']
    m = parameter_df_entry['m']
    w = parameter_df_entry['w']
    de = parameter_df_entry['de']
    E50 = parameter_df_entry['E50']
    r = parameter_df_entry['r']
    q = parameter_df_entry['q']
    p = parameter_df_entry['p']
    c = parameter_df_entry['c']

    # Initial conditions for ODE; one of the ICs depends on parameters:
    S_0 = 1e7
    I_0 = 1
    V_0 = p * I_0 / c
    E_0 = 0
    M1_0 = 1
    M2_0 = 0

    ODE_Y0 = (S_0, I_0, V_0, M1_0, M2_0, E_0)
    ODE_params = (beta, delta, k, m, r, E50, p, c, w, q, de)
    return ODE_Y0, ODE_params

def get_Ke_ODE_setup(parameter_df_entry):
    # Take in a row of a dataframe, and return parameter and Y0 used for ODE simulation
    beta_T = parameter_df_entry['betaT']
    k1 = parameter_df_entry['k1']
    pi_T = parameter_df_entry['piT']
    delta1 = parameter_df_entry['delta1']
    beta_S = parameter_df_entry['betaS']
    k2 = parameter_df_entry['k2']
    pi_S = parameter_df_entry['piS']
    delta2 = parameter_df_entry['delta2']
    c = parameter_df_entry['c']
    w = parameter_df_entry['w']
    gamma = parameter_df_entry['gamma']
    ttau = parameter_df_entry['ttau']
    log10TN = parameter_df_entry['log10TN']

    # Initial conditions for ODE
    T10 = parameter_df_entry['T10']
    T20 = parameter_df_entry['T20']
    I10 = parameter_df_entry['I10']# * 100
    # print('Warning - Set I10 to 100.')
    E10 = 0
    VT0 = 0
    E20 = 0
    I20 = 0
    VS0 = 0
    ODE_Y0 = [T10, E10, I10, VT0, T20, E20, I20, VS0]

    ODE_params = (beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, ttau, log10TN)
    return ODE_Y0, ODE_params


@jit
def Goyal_ODEs(t, y, params):
    """
    Model ODEs, written with scipy.integrate.solve_ivp() in mind. Follows within-host model from Goyal et al.
    :param t: Time, days
    :param y: Microbiological states (cells and virions)
    :param params: Model parameters, see below for descriptive names
    :return dydt: Rate change for each microbiological state.
    """
    # Unpack states
    # Susceptible (S), infected (I), and virion-producing cells (V)
    # Precursor cell states (M1, M2) to immune effector cells (E)
    S, I, V, M1, M2, E = y

    # Unpack parameters
    # Transmission rate constant (beta)
    # Innate immune clearance parameters (delta, k)
    # Adaptive immune clearance by effector cell parameters (m, r, E50)
    # Virion production and clearance rates (p, c)
    # Effector cell precursor stimulation by infected cells and pre-existing precursor (w)
    # Precursor differentiation rates (q)
    # Effector cell death (de)
    beta, delta, k, m, r, E50, p, c, w, q, de = params

    dydt = np.zeros(6)

    # dS/dt
    dydt[0] = -beta*V*S

    # dI/dt
    dydt[1] = beta*V*S - delta*I**k*I - m*E**r*I/(E**r+E50**r)

    # dV/dt
    dydt[2] = p*I - c*V

    # dM1/dt
    dydt[3] = w*I*M1 - q*M1

    # dM2/dt
    dydt[4] = q*(M1-M2)

    # dE/dt
    dydt[5] = q*M2 - de*E

    return dydt

@jit
def Ke_ODEs_TIP(t, y, params):
    """
    Model ODEs for co-infection by wildtype SARS-CoV-2 and TIP, written w/ scipy.integrate.solve_ivp() in mind.
    :param t: Time, days
    :param y: Within-host microbiological states (cells & virions by physiological location)
    :param params: Model parameters, see below for descriptive names.
    :return dydt: Rate change for each microbiological state.
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
    # delta_TIP = relative clearance rate for an infected cell also containing TIP (
    # See Ruian Ke et al. medRxiv 10.1101/2020.09.25.20201772v1 for further definitions, parameter names match.
    beta_T, k1, pi_T, delta1, beta_S, k2, pi_S, delta2, c, w, gamma, rho, psi, xi, delta_TIP = params

    n_entries = 16
    dydt = np.zeros(n_entries)

    # Calculate effect of adaptive immune response
    if t >= 14:
        delta1 = delta1 * np.exp(w * (t - 14))
        delta2 = delta2 * np.exp(w * (t - 14))
    else:
        delta1 = delta1
        delta2 = delta2

    # dT1/dt
    dydt[0] = -beta_T * VTWT * T1 - xi * beta_T * VTTIP * T1

    # dE1WT/dt
    dydt[1] = +beta_T * VTWT * T1 - k1 * E1WT

    # dI1WT/dt
    dydt[2] = k1 * E1WT - delta1 * I1WT

    # dVTWT/dt
    dydt[3] = pi_T * I1WT - c * VTWT + psi * pi_T * I1D

    # dT1TIP/dt
    dydt[4] = xi * beta_T * VTTIP * T1 - beta_T * VTWT * T1TIP

    # dE1D/dt
    dydt[5] = beta_T * VTWT * T1TIP - k1 * E1D

    # dI1D/dt
    dydt[6] = k1 * E1D - delta1 * delta_TIP *  I1D

    # dVTTIP/dt
    dydt[7] = rho * pi_T * I1D - c * VTTIP

    # dT2/dt
    dydt[8] = -beta_S * VSWT * T2 - xi * beta_S * VSTIP * T2

    # dE2WT/dt
    dydt[9] = beta_S * VSWT * T2 - k2 * E2WT

    # dI2WT/dt
    dydt[10] = k2 * E2WT - delta2 * I2WT

    # dVSWT/dt
    dydt[11] = pi_S * I2WT - c * VSWT + psi * pi_S * I2D + gamma * VTWT

    # dT2TIP/dt
    dydt[12] = xi * beta_S * VSTIP * T2 - beta_S * VSWT * T2TIP

    # dE2D/dt
    dydt[13] = beta_S * VSWT * T2TIP - k2 * E2D

    # dI2D/dt
    dydt[14] = k2 * E2D - delta2 * delta_TIP * I2D

    # dVSTIP/dt
    dydt[15] = rho * pi_S * I2D - c * VSTIP + gamma * VTTIP

    return dydt

def get_reference_serial_interval_array():
    """
    Returns reference array of infector-infectee serial intervals, which can be plotted as a histogram or CDF.
    True source is PMID 32191173: Du Z et al. 'Serial Interval of COVID-19 among Publicly Reported Confirmed Cases'
    Emerging Infectious Diseases 2020, June, 26(6) 1341-1343.
        Based on "...468 COVID-19 transmission events reported in mainland China outside of Hubei Province during
        January 21–February 8, 2020. Each report consists of a probable date of symptom onset for both the infector
        and infectee, as well as the probable locations of infection for both case-patients. The data include only
        confirmed cases compiled from online reports from 18 provincial centers for disease control and prevention
        (https://github.com/MeyersLabUTexas/COVID-19)."
    :return: SI_array, 109 individual serial interval values
    """
    SI_array = np.concatenate((-8*np.ones(1),
                               -5*np.ones(1),
                               -4*np.ones(3),
                               -3*np.ones(1),
                               -2*np.ones(1),
                               0*np.ones(8),
                               1*np.ones(7),
                               2*np.ones(10),
                               3*np.ones(14),
                               4*np.ones(8),
                               5*np.ones(12),
                               6*np.ones(8),
                               7*np.ones(6),
                               8*np.ones(7),
                               9*np.ones(5),
                               10*np.ones(4),
                               11*np.ones(3),
                               12*np.ones(3),
                               13*np.ones(3),
                               14*np.ones(1),
                               15*np.ones(1),
                               16*np.ones(2)))

    return SI_array

def extract_simulated_R0_bar_heights(R0_data):
    ni = np.zeros(11)
    for i in range(10):
        ni[i] = sum(R0_data==i)
    ni[10] = sum(R0_data>=10)
    ni = ni/np.sum(ni)
    return ni

def get_reference_individual_R0_data():
    """
    Returns reference array of individual-level R0, which can be plotted as a histogram or CDF.
    True source is https://cmmid.github.io/topics/covid19/overdispersion-from-outbreaksize.html
    This return format is a 1000-element list of (sorted) individual R0 values.
    I should verify the actual parameterization they use to generate this vector...
    :return R0_array, R0_heights: R0_array: 1000 individual R0 values based on overdispersion estimates (e.g. superspreading)
    R0_heights: 11 'bar heights' (from a probability-normalized histogram) corresponding to values of:
        R0=={1,2,3,4,5,6,7,8,9} and R0>=10.
    """
    R0_array = np.concatenate( (np.zeros(720),
                                np.ones(800-720),
                                2*np.ones(835-800),
                                3*np.ones(860-835),
                                4*np.ones(880-860),
                                5*np.ones(895-880),
                                6*np.ones(905-895),
                                7*np.ones(910-905),
                                8*np.ones(913-910),
                                9*np.ones(916-913),
                                10*np.ones(919-916),
                                15*np.ones(940-919),
                                25*np.ones(1000-940)))

    R0_heights = [0.72, 0.08, 0.035, 0.025, 0.020, 0.015, 0.01, 0.005, 0.004, 0.003, 1-0.917]

    return R0_array, R0_heights

def visualize_results_WTTIP_TIP(df_result_entry, hostParameterFile, resultName, figureName):
    # Plot histogram and cdf of individual R0 and SI, as well as KS statistic
    # Plot histogram of generation time
    dispersion = df_result_entry['dispersion']
    tzero = df_result_entry['tzero']
    alpha = df_result_entry['alpha']
    lambda_tx = df_result_entry['lambda_tx']
    theta = df_result_entry['theta']
    Tx_max = df_result_entry['Tx_max']
    rho = df_result_entry['rho']
    psi = df_result_entry['psi']
    xi = df_result_entry['xi']
    eta = df_result_entry['eta']
    delta_TIP = df_result_entry['delta_TIP']

    # Re-run the transmission simulation get values specific to that entry.
    n_transmitters = 10000
    print('Using {} transmitters'.format(n_transmitters))
    results = simulate_transmissions_WTTIP_TIP_tx(dispersion=dispersion, tzero=tzero, alpha=alpha, lambda_tx=lambda_tx,
                                                  theta=theta, n_transmitters=n_transmitters, dT=1, Tx_sat=Tx_max,
                                                  hostParameterFile=hostParameterFile,
                                                  rho=rho, psi=psi, xi=xi, eta=eta, delta_TIP=delta_TIP)

    results.to_csv(resultName, header=True, index=False)

    fig, axs = plt.subplots(3, 2)

    # Clean up
    R0vals = results['R0'].values
    SIvals = results['SI'].values
    GTvals = results['GT'].values

    R0_sim = R0vals.astype('float')  # Cast to float to avoid weird bugs w/ np.isnan vs types from pd.DataFrame
    SI_sim = SIvals.astype('float')
    GT_sim = GTvals.astype('float')

    SI_sim = SI_sim[~np.isnan(SI_sim)]
    GT_sim = GT_sim[~np.isnan(GT_sim)]

    if len(SI_sim) == 0 or len(GT_sim) == 0:
        print('Result from', hostParameterFile, 'has no valid serial interval or generation time. Skipping.')

    # R0
    R0_array_ref, R0_heights_ref = get_reference_individual_R0_data()
    R0_axmin = 0
    R0_axmax = np.max((np.max(R0_sim), np.max(R0_array_ref)))
    axs[0, 0].hist(R0_sim,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   label='Simulation',
                   histtype='step')
    axs[0, 0].hist(R0_array_ref,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   label='Data',
                   histtype='step')
    axs[0, 0].set_xlim(0, 10)
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].text(4, 0.15, 'Sim. Mean = {:.2f}'.format(np.nanmean(R0_sim)), fontsize=10)
    axs[0, 0].text(4, 0.3, 'Exp. Mean = {:.2f}'.format(np.nanmean(R0_array_ref)), fontsize=10)
    axs[0, 0].set_xlabel('Individual R0')
    axs[0, 0].set_ylabel('Fraction')
    axs[0, 0].legend()

    # R0, CDF
    # Calculate KS statistic
    ks_stat, ks_p = ks_2samp(R0_array_ref, R0_sim)
    axs[0, 1].hist(R0_sim,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   cumulative=True,
                   label='Simulation',
                   histtype='step')
    axs[0, 1].hist(R0_array_ref,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   cumulative=True,
                   label='Data',
                   histtype='step')
    axs[0, 1].set_xlim(0, 10)
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].set_xlabel('Individual R0')
    axs[0, 1].set_ylabel('Fraction')
    #axs[0, 1].legend(loc='lower right')
    axs[0, 1].text(6.5, 0.7, 'p = {:.2f}'.format(ks_p), fontsize=10)

    # SI
    SI_array_ref = get_reference_serial_interval_array()
    SI_axmin = np.min((np.min(SI_sim), np.min(SI_array_ref)))
    SI_axmax = np.max((np.max((np.max(SI_sim), np.max(SI_array_ref))), 21))

    axs[1, 0].hist(SI_sim,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   label='Simulation',
                   histtype='step')
    axs[1, 0].hist(SI_array_ref,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   label='Data',
                   histtype='step')
    axs[1, 0].set_xlim(-10, 20)
    axs[1, 0].set_ylim(0, 0.25)
    axs[1, 0].text(-8, 0.21, 'Sim. Mean = {:.2f}'.format(np.nanmean(SI_sim)), fontsize=10)
    axs[1, 0].text(-8, 0.15, 'Exp. Mean = {:.2f}'.format(np.nanmean(SI_array_ref)), fontsize=10)
    axs[1, 0].set_xlabel('Serial Interval (d)')
    axs[1, 0].set_ylabel('Fraction')
    #axs[1, 0].legend()

    # SI, CDF
    # Calculate KS statistic
    ks_stat, ks_p = ks_2samp(SI_array_ref, SI_sim)
    axs[1, 1].hist(SI_sim,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   cumulative=True,
                   label='Simulation',
                   histtype='step')
    axs[1, 1].hist(SI_array_ref,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   cumulative=True,
                   label='Data',
                   histtype='step')
    axs[1, 1].set_xlim(-10, 20)
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].set_xlabel('Serial Interval (d)')
    axs[1, 1].set_ylabel('Fraction')
    #axs[1, 1].legend(loc='lower right')
    axs[1, 1].text(11, 0.7, 'p = {:.2f}'.format(ks_p), fontsize=10)

    # GT
    GT_axmin = np.min(GT_sim)
    GT_axmax = np.max(GT_sim)

    axs[2, 0].hist(GT_sim,
                   bins=np.arange(GT_axmin, GT_axmax),
                   density=True,
                   label='Simulation',
                   histtype='step')
    axs[2, 0].set_xlim(0, 15)
    axs[2, 0].set_ylim(0, 1)
    axs[2, 0].text(2, 0.7, 'Mean = {:.2f}'.format(np.nanmean(GT_sim)), fontsize=10)
    axs[2, 0].set_xlabel('Generation Time (d)')
    axs[2, 0].set_ylabel('Fraction')
    #axs[2, 0].legend()

    axs[2, 1].set_visible(False)
    plt.tight_layout()

    plt.savefig(figureName, dpi=300)
    plt.close(fig='all')

def visualize_results_WTTIP_WTTIP(df_result_entry, hostParameterFile, resultName, figureName):
    # Plot histogram and cdf of individual R0 and SI, as well as KS statistic
    # Plot histogram of generation time
    dispersion = df_result_entry['dispersion']
    tzero = df_result_entry['tzero']
    alpha = df_result_entry['alpha']
    lambda_tx = df_result_entry['lambda_tx']
    theta = df_result_entry['theta']
    Tx_max = df_result_entry['Tx_max']
    rho = df_result_entry['rho']
    psi = df_result_entry['psi']
    xi = df_result_entry['xi']
    eta = df_result_entry['eta']
    delta_TIP = df_result_entry['delta_TIP']

    # Re-run the transmission simulation get values specific to that entry.

    n_transmitters = 10000
    print('Using {} transmitters'.format(n_transmitters))
    results = simulate_transmissions_WTTIP_WTTIP_tx(dispersion=dispersion, tzero=tzero, alpha=alpha, lambda_tx=lambda_tx,
                                                   theta=theta, n_transmitters=n_transmitters, dT=1, Tx_sat=Tx_max,
                                                   hostParameterFile=hostParameterFile,
                                                   rho=rho, psi=psi, xi=xi, eta=eta, delta_TIP=delta_TIP)

    results.to_csv(resultName, header=True, index=False)

    fig, axs = plt.subplots(3, 2)

    # Clean up
    R0vals = results['R0'].values
    SIvals = results['SI'].values
    GTvals = results['GT'].values

    R0_sim = R0vals.astype('float')  # Cast to float to avoid weird bugs w/ np.isnan vs types from pd.DataFrame
    SI_sim = SIvals.astype('float')
    GT_sim = GTvals.astype('float')

    SI_sim = SI_sim[~np.isnan(SI_sim)]
    GT_sim = GT_sim[~np.isnan(GT_sim)]

    if len(SI_sim) == 0 or len(GT_sim) == 0:
        print('Result from', hostParameterFile, 'has no valid serial interval or generation time. Skipping.')

    # R0
    R0_array_ref, R0_heights_ref = get_reference_individual_R0_data()
    R0_axmin = 0
    R0_axmax = np.max((np.max(R0_sim), np.max(R0_array_ref)))
    axs[0, 0].hist(R0_sim,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   label='Simulation',
                   histtype='step')
    axs[0, 0].hist(R0_array_ref,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   label='Data',
                   histtype='step')
    axs[0, 0].set_xlim(0, 10)
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].text(4, 0.15, 'Sim. Mean = {:.2f}'.format(np.nanmean(R0_sim)), fontsize=10)
    axs[0, 0].text(4, 0.3, 'Exp. Mean = {:.2f}'.format(np.nanmean(R0_array_ref)), fontsize=10)
    axs[0, 0].set_xlabel('Individual R0')
    axs[0, 0].set_ylabel('Fraction')
    axs[0, 0].legend()

    # R0, CDF
    # Calculate KS statistic
    ks_stat, ks_p = ks_2samp(R0_array_ref, R0_sim)
    axs[0, 1].hist(R0_sim,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   cumulative=True,
                   label='Simulation',
                   histtype='step')
    axs[0, 1].hist(R0_array_ref,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   cumulative=True,
                   label='Data',
                   histtype='step')
    axs[0, 1].set_xlim(0, 10)
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].set_xlabel('Individual R0')
    axs[0, 1].set_ylabel('Fraction')
    #axs[0, 1].legend(loc='lower right')
    axs[0, 1].text(6.5, 0.7, 'p = {:.2f}'.format(ks_p), fontsize=10)

    # SI
    SI_array_ref = get_reference_serial_interval_array()
    SI_axmin = np.min((np.min(SI_sim), np.min(SI_array_ref)))
    SI_axmax = np.max((np.max((np.max(SI_sim), np.max(SI_array_ref))), 21))

    axs[1, 0].hist(SI_sim,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   label='Simulation',
                   histtype='step')
    axs[1, 0].hist(SI_array_ref,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   label='Data',
                   histtype='step')
    axs[1, 0].set_xlim(-10, 20)
    axs[1, 0].set_ylim(0, 0.25)
    axs[1, 0].text(-8, 0.21, 'Sim. Mean = {:.2f}'.format(np.nanmean(SI_sim)), fontsize=10)
    axs[1, 0].text(-8, 0.15, 'Exp. Mean = {:.2f}'.format(np.nanmean(SI_array_ref)), fontsize=10)
    axs[1, 0].set_xlabel('Serial Interval (d)')
    axs[1, 0].set_ylabel('Fraction')
    #axs[1, 0].legend()

    # SI, CDF
    # Calculate KS statistic
    ks_stat, ks_p = ks_2samp(SI_array_ref, SI_sim)
    axs[1, 1].hist(SI_sim,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   cumulative=True,
                   label='Simulation',
                   histtype='step')
    axs[1, 1].hist(SI_array_ref,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   cumulative=True,
                   label='Data',
                   histtype='step')
    axs[1, 1].set_xlim(-10, 20)
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].set_xlabel('Serial Interval (d)')
    axs[1, 1].set_ylabel('Fraction')
    #axs[1, 1].legend(loc='lower right')
    axs[1, 1].text(11, 0.7, 'p = {:.2f}'.format(ks_p), fontsize=10)

    # GT
    GT_axmin = np.min(GT_sim)
    GT_axmax = np.max(GT_sim)

    axs[2, 0].hist(GT_sim,
                   bins=np.arange(GT_axmin, GT_axmax),
                   density=True,
                   label='Simulation',
                   histtype='step')
    axs[2, 0].set_xlim(0, 15)
    axs[2, 0].set_ylim(0, 1)
    axs[2, 0].text(2, 0.7, 'Mean = {:.2f}'.format(np.nanmean(GT_sim)), fontsize=10)
    axs[2, 0].set_xlabel('Generation Time (d)')
    axs[2, 0].set_ylabel('Fraction')
    #axs[2, 0].legend()

    axs[2, 1].set_visible(False)
    plt.tight_layout()

    plt.savefig(figureName, dpi=300)

def visualize_results_WTTIP_WT(df_result_entry, hostParameterFile, resultName, figureName):
    # Plot histogram and cdf of individual R0 and SI, as well as KS statistic
    # Plot histogram of generation time
    dispersion = df_result_entry['dispersion']
    tzero = df_result_entry['tzero']
    alpha = df_result_entry['alpha']
    lambda_tx = df_result_entry['lambda_tx']
    theta = df_result_entry['theta']
    Tx_max = df_result_entry['Tx_max']
    rho = df_result_entry['rho']
    psi = df_result_entry['psi']
    xi = df_result_entry['xi']
    eta = df_result_entry['eta']
    delta_TIP = df_result_entry['delta_TIP']

    # Re-run the transmission simulation get values specific to that entry.

    n_transmitters = 10000
    print('Using {} transmitters'.format(n_transmitters))
    results = simulate_transmissions_WTTIP_WT_tx(dispersion=dispersion, tzero=tzero, alpha=alpha, lambda_tx=lambda_tx,
                                                 theta=theta, n_transmitters=n_transmitters, dT=1, Tx_sat=Tx_max,
                                                 hostParameterFile=hostParameterFile,
                                                 rho=rho, psi=psi, xi=xi, eta=eta, delta_TIP=delta_TIP)

    results.to_csv(resultName, header=True, index=False)

    fig, axs = plt.subplots(3, 2)

    # Clean up
    R0vals = results['R0'].values
    SIvals = results['SI'].values
    GTvals = results['GT'].values

    R0_sim = R0vals.astype('float')  # Cast to float to avoid weird bugs w/ np.isnan vs types from pd.DataFrame
    SI_sim = SIvals.astype('float')
    GT_sim = GTvals.astype('float')

    # Drop nan-valued entries
    print('nan-valued entries:')
    print('\tR0: {}\n\tSI: {}\n\tGT: {}\n'.format(np.sum(np.isnan(R0_sim)),
                                                  np.sum(np.isnan(SI_sim)),
                                                  np.sum(np.isnan(GT_sim))))
    R0_sim = R0_sim[~np.isnan(R0_sim)]
    SI_sim = SI_sim[~np.isnan(SI_sim)]
    GT_sim = GT_sim[~np.isnan(GT_sim)]

    if len(SI_sim) == 0 or len(GT_sim) == 0:
        print('Result from', hostParameterFile, 'has no valid serial interval or generation time. Skipping.')

    # R0
    R0_array_ref, R0_heights_ref = get_reference_individual_R0_data()
    R0_axmin = 0
    R0_axmax = np.max((np.max(R0_sim), np.max(R0_array_ref)))
    axs[0, 0].hist(R0_sim,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   label='Simulation',
                   histtype='step')
    axs[0, 0].hist(R0_array_ref,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   label='Data',
                   histtype='step')
    axs[0, 0].set_xlim(0, 10)
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].text(4, 0.15, 'Sim. Mean = {:.2f}'.format(np.nanmean(R0_sim)), fontsize=10)
    axs[0, 0].text(4, 0.3, 'Exp. Mean = {:.2f}'.format(np.nanmean(R0_array_ref)), fontsize=10)
    axs[0, 0].set_xlabel('Individual R0')
    axs[0, 0].set_ylabel('Fraction')
    axs[0, 0].legend()

    # R0, CDF
    # Calculate KS statistic
    ks_stat, ks_p = ks_2samp(R0_array_ref, R0_sim)
    axs[0, 1].hist(R0_sim,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   cumulative=True,
                   label='Simulation',
                   histtype='step')
    axs[0, 1].hist(R0_array_ref,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   cumulative=True,
                   label='Data',
                   histtype='step')
    axs[0, 1].set_xlim(0, 10)
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].set_xlabel('Individual R0')
    axs[0, 1].set_ylabel('Fraction')
    #axs[0, 1].legend(loc='lower right')
    axs[0, 1].text(6.5, 0.7, 'p = {:.2f}'.format(ks_p), fontsize=10)

    # SI
    SI_array_ref = get_reference_serial_interval_array()
    SI_axmin = np.min((np.min(SI_sim), np.min(SI_array_ref)))
    SI_axmax = np.max((np.max((np.max(SI_sim), np.max(SI_array_ref))), 21))

    axs[1, 0].hist(SI_sim,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   label='Simulation',
                   histtype='step')
    axs[1, 0].hist(SI_array_ref,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   label='Data',
                   histtype='step')
    axs[1, 0].set_xlim(-10, 20)
    axs[1, 0].set_ylim(0, 0.25)
    axs[1, 0].text(-8, 0.21, 'Sim. Mean = {:.2f}'.format(np.nanmean(SI_sim)), fontsize=10)
    axs[1, 0].text(-8, 0.15, 'Exp. Mean = {:.2f}'.format(np.nanmean(SI_array_ref)), fontsize=10)
    axs[1, 0].set_xlabel('Serial Interval (d)')
    axs[1, 0].set_ylabel('Fraction')
    #axs[1, 0].legend()

    # SI, CDF
    # Calculate KS statistic
    ks_stat, ks_p = ks_2samp(SI_array_ref, SI_sim)
    axs[1, 1].hist(SI_sim,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   cumulative=True,
                   label='Simulation',
                   histtype='step')
    axs[1, 1].hist(SI_array_ref,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   cumulative=True,
                   label='Data',
                   histtype='step')
    axs[1, 1].set_xlim(-10, 20)
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].set_xlabel('Serial Interval (d)')
    axs[1, 1].set_ylabel('Fraction')
    #axs[1, 1].legend(loc='lower right')
    axs[1, 1].text(11, 0.7, 'p = {:.2f}'.format(ks_p), fontsize=10)

    # GT
    GT_axmin = np.min(GT_sim)
    GT_axmax = np.max(GT_sim)

    axs[2, 0].hist(GT_sim,
                   bins=np.arange(GT_axmin, GT_axmax),
                   density=True,
                   label='Simulation',
                   histtype='step')
    axs[2, 0].set_xlim(0, 15)
    axs[2, 0].set_ylim(0, 1)
    axs[2, 0].text(2, 0.7, 'Mean = {:.2f}'.format(np.nanmean(GT_sim)), fontsize=10)
    axs[2, 0].set_xlabel('Generation Time (d)')
    axs[2, 0].set_ylabel('Fraction')
    #axs[2, 0].legend()

    axs[2, 1].set_visible(False)
    plt.tight_layout()

    plt.savefig(figureName, dpi=300)

def visualize_TIP_impact(wildtypeResultFile, TIPResultFile, figureName):
    df_wt = pd.read_csv(wildtypeResultFile)
    df_TIP = pd.read_csv(TIPResultFile)

    # Left: overall CDF; right: zoom-in of top 75%ile?
    fig, axs = plt.subplots(3, 2)
    cdf_y_zoom = [0.75, 1]

    # Clean up
    R0_wt = df_wt['R0'].values
    SI_wt = df_wt['SI'].values
    GT_wt = df_wt['GT'].values

    R0_TIP = df_TIP['R0'].values
    SI_TIP = df_TIP['SI'].values
    GT_TIP = df_TIP['GT'].values

    R0_wt = R0_wt.astype('float')
    SI_wt = SI_wt.astype('float')
    GT_wt = GT_wt.astype('float')
    R0_TIP = R0_TIP.astype('float')
    SI_TIP = SI_TIP.astype('float')
    GT_TIP = GT_TIP.astype('float')

    SI_wt = SI_wt[~np.isnan(SI_wt)]
    GT_wt = GT_wt[~np.isnan(GT_wt)]
    SI_TIP = SI_TIP[~np.isnan(SI_TIP)]
    GT_TIP = GT_TIP[~np.isnan(GT_TIP)]

    # R0
    R0_axmin = 0
    R0_axmax = np.max((np.max(R0_wt), np.max(R0_TIP)))

    axs[0, 0].hist(R0_wt,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   cumulative=True,
                   label='No TIP',
                   histtype='step')
    axs[0, 0].hist(R0_TIP,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   cumulative=True,
                   label='TIP',
                   histtype='step')
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_xlim(R0_axmin, R0_axmax-1)
    axs[0, 0].set_xlabel('Individual R0')
    axs[0, 0].set_ylabel('CDF')
    axs[0, 0].legend(loc='lower right')

    # R0 zoom
    axs[0, 1].hist(R0_wt,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   cumulative=True,
                   label='No TIP',
                   histtype='step')
    axs[0, 1].hist(R0_TIP,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   cumulative=True,
                   label='TIP',
                   histtype='step')
    axs[0, 1].set_ylim(cdf_y_zoom)
    axs[0, 1].set_xlim(R0_axmin, R0_axmax-1)
    axs[0, 1].set_xlabel('Individual R0')
    axs[0, 1].set_ylabel('CDF')

    # SI
    SI_axmin = np.min((np.min(SI_wt), np.min(SI_TIP)))
    SI_axmax = np.max((np.max((np.max(SI_wt), np.max(SI_TIP))), 21))
    axs[1, 0].hist(SI_wt,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   cumulative=True,
                   label='Simulation',
                   histtype='step')
    axs[1, 0].hist(SI_TIP,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   cumulative=True,
                   label='Data',
                   histtype='step')
    axs[1, 0].set_xlim(SI_axmin, SI_axmax-1)
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].set_xlabel('Serial Interval (d)')
    axs[1, 0].set_ylabel('CDF')

    # SI zoom
    axs[1, 1].hist(SI_wt,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   cumulative=True,
                   label='Simulation',
                   histtype='step')
    axs[1, 1].hist(SI_TIP,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   cumulative=True,
                   label='Data',
                   histtype='step')
    axs[1, 1].set_xlim(SI_axmin, SI_axmax-1)
    axs[1, 1].set_ylim(cdf_y_zoom)
    axs[1, 1].set_xlabel('Serial Interval (d)')
    axs[1, 1].set_ylabel('CDF')

    # GT
    GT_axmin = np.min((np.min(GT_wt), np.min(GT_TIP)))
    GT_axmax = np.max((np.max(GT_wt), np.max(GT_TIP)))

    axs[2, 0].hist(GT_wt,
                   bins=np.arange(GT_axmin, GT_axmax),
                   density=True,
                   cumulative=True,
                   label='No TIP',
                   histtype='step')
    axs[2, 0].hist(GT_TIP,
                   bins=np.arange(GT_axmin, GT_axmax),
                   density=True,
                   cumulative=True,
                   label='TIP',
                   histtype='step')
    axs[2, 0].set_xlim(GT_axmin, GT_axmax)
    axs[2, 0].set_ylim(0, 1)
    axs[2, 0].set_xlabel('Generation Time (d)')
    axs[2, 0].set_ylabel('CDF')

    # GT Zoom
    axs[2, 1].hist(GT_wt,
                   bins=np.arange(GT_axmin, GT_axmax),
                   density=True,
                   cumulative=True,
                   label='No TIP',
                   histtype='step')
    axs[2, 1].hist(GT_TIP,
                   bins=np.arange(GT_axmin, GT_axmax),
                   density=True,
                   cumulative=True,
                   label='TIP',
                   histtype='step')
    axs[2, 1].set_xlim(GT_axmin, GT_axmax)
    axs[2, 1].set_ylim(cdf_y_zoom)
    axs[2, 1].set_xlabel('Generation Time (d)')
    axs[2, 1].set_ylabel('CDF')

    plt.tight_layout()

    plt.savefig(figureName, dpi=300)

def get_transmission_curve(Tx_sat, alpha, lambda_tx, V):
    Prob_V = Tx_sat * V ** alpha / (lambda_tx ** alpha + V ** alpha)
    return Prob_V

def plot_saved_comparison_results(WTWTfile, DualWTfile, figureName):
    WTdf = pd.read_csv(WTWTfile)
    DualWTdf = pd.read_csv(DualWTfile)

    fig, axs = plt.subplots(3, 2)

    R0_WTWTvals = WTdf['R0'].values
    SI_WTWTvals = WTdf['SI'].values
    GT_WTWTvals = WTdf['GT'].values

    R0_WTWT = R0_WTWTvals.astype('float')  # Cast to float to avoid weird bugs w/ np.isnan vs types from pd.DataFrame
    SI_WTWT = SI_WTWTvals.astype('float')
    GT_WTWT = GT_WTWTvals.astype('float')

    SI_WTWT = SI_WTWT[~np.isnan(SI_WTWT)]
    GT_WTWT = GT_WTWT[~np.isnan(GT_WTWT)]

    R0_DualWTvals = DualWTdf['R0'].values
    SI_DualWTvals = DualWTdf['SI'].values
    GT_DualWTvals = DualWTdf['GT'].values

    R0_DualWT = R0_DualWTvals.astype('float')  # Cast to float to avoid weird bugs w/ np.isnan vs types from pd.DataFrame
    SI_DualWT = SI_DualWTvals.astype('float')
    GT_DualWT = GT_DualWTvals.astype('float')

    SI_DualWT = SI_DualWT[~np.isnan(SI_DualWT)]
    GT_DualWT = GT_DualWT[~np.isnan(GT_DualWT)]


    # R0
    R0_axmin = 0
    R0_axmax = np.max((np.max(R0_WTWT), np.max(R0_DualWT)))
    axs[0, 0].hist(R0_WTWT,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   label='WT->WT',
                   histtype='step')
    axs[0, 0].hist(R0_DualWT,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   label='Dual->WT',
                   histtype='step')

    axs[0, 0].set_xlim(0, 50)
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_xlabel('Individual R0')
    axs[0, 0].set_ylabel('Fraction')
    axs[0, 0].legend()

    # R0, CDF
    # Calculate KS statistic
    ks_stat, ks_p = ks_2samp(R0_DualWT, R0_WTWT)
    axs[0, 1].hist(R0_WTWT,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   cumulative=True,
                   label='WT->WT',
                   histtype='step')
    axs[0, 1].hist(R0_DualWT,
                   bins=np.arange(R0_axmin, R0_axmax),
                   density=True,
                   cumulative=True,
                   label='Dual->WT',
                   histtype='step')
    axs[0, 1].set_xlim(0, 50)
    axs[0, 1].set_ylim(0.75, 1)
    axs[0, 1].set_xlabel('Individual R0')
    axs[0, 1].set_ylabel('Fraction')
    #axs[0, 1].legend(loc='lower right')
    #axs[0, 1].text(6.5, 0.7, 'p = {:.2f}'.format(ks_p))

    # SI
    SI_array_ref = get_reference_serial_interval_array()
    SI_axmin = np.min((np.min(SI_WTWT), np.min(SI_DualWT)))
    SI_axmax = np.max((np.max((np.max(SI_WTWT), np.max(SI_DualWT))), 21))

    axs[1, 0].hist(SI_WTWT,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   label='WT->WT',
                   histtype='step')
    axs[1, 0].hist(SI_DualWT,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   label='Dual->WT',
                   histtype='step')
    axs[1, 0].set_xlim(-10, 20)
    axs[1, 0].set_ylim(0, 0.25)
    axs[1, 0].text(-8, 0.21, 'Sim. Mean = {:.2f}'.format(np.nanmean(SI_WTWT)))
    axs[1, 0].text(-8, 0.15, 'Exp. Mean = {:.2f}'.format(np.nanmean(SI_DualWT)))
    axs[1, 0].set_xlabel('Serial Interval (d)')
    axs[1, 0].set_ylabel('Fraction')
    #axs[1, 0].legend()

    # SI, CDF
    # Calculate KS statistic
    ks_stat, ks_p = ks_2samp(SI_DualWT, SI_WTWT)
    axs[1, 1].hist(SI_WTWT,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   cumulative=True,
                   label='WT->WT',
                   histtype='step')
    axs[1, 1].hist(SI_DualWT,
                   bins=np.arange(SI_axmin, SI_axmax),
                   density=True,
                   cumulative=True,
                   label='Dual->WT',
                   histtype='step')
    axs[1, 1].set_xlim(-10, 20)
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].set_xlabel('Serial Interval (d)')
    axs[1, 1].set_ylabel('Fraction')
    #axs[1, 1].legend(loc='lower right')
    axs[1, 1].text(11, 0.7, 'p = {:.2f}'.format(ks_p))

    # GT
    GT_axmin = np.min((np.min(GT_WTWT), np.min(GT_DualWT)))
    GT_axmax = np.max((np.max(GT_WTWT), np.max(GT_DualWT)))

    axs[2, 0].hist(GT_WTWT,
                   bins=np.arange(GT_axmin, GT_axmax),
                   density=True,
                   label='WT->WT',
                   histtype='step')
    axs[2, 0].hist(GT_DualWT,
                   bins=np.arange(GT_axmin, GT_axmax),
                   density=True,
                   label='Dual->WT',
                   histtype='step')
    axs[2, 0].set_xlim(0, 15)
    axs[2, 0].set_ylim(0, 1)
    axs[2, 0].text(2, 0.7, 'Mean = {:.2f}'.format(np.nanmean(GT_WTWT)))
    axs[2, 0].set_xlabel('Generation Time (d)')
    axs[2, 0].set_ylabel('Fraction')
    #axs[2, 0].legend()

    axs[2, 1].set_visible(False)
    plt.tight_layout()

    plt.savefig(figureName, dpi=300)

def __main__():
    doInitialVisualizations = True
    doComparisonToNoTIP = True
    plot_WTWT_vs_DualWT_result = True
    np.random.seed(0)  # For reproducibility

    sourcefolder = 'search1_DirectTreated/'
    WT_calibrated_param_set = '20210529_new_recalibration_parameter_sweep_list'
    TIP_metadata_suffix = 'rho1-5_psi0-02_TIP_pretreat'
    df = pd.read_csv(sourcefolder + WT_calibrated_param_set + '.txt')

    to_test = [0, 1, 2]
    for i in to_test:
        if doInitialVisualizations:
            if True and i>0:
                curr_entry = df.iloc[i].copy(deep=True)
                curr_entry['rho'] = 1.5
                curr_entry['psi'] = 1/50
                curr_entry['xi'] = 1
                curr_entry['eta'] = 1
                curr_entry['delta_TIP'] = 1

                try:
                    visualize_results_WTTIP_WT(df_result_entry=curr_entry,
                                               hostParameterFile='SimulatedHostParameters-Ke.txt',
                                               resultName=sourcefolder + WT_calibrated_param_set + '-WTTIP_WT-txset{:02d}-'.format(i) + TIP_metadata_suffix + '.txt',
                                               figureName=sourcefolder + WT_calibrated_param_set + '-WTTIP_WT-txset{:02d}-'.format(i) + TIP_metadata_suffix + '.png')
                except Exception as ex:
                    template = "Skipping parameter set; an exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)
                    print(traceback.format_exc())

                try:
                    visualize_results_WTTIP_WTTIP(df_result_entry=curr_entry,
                                               hostParameterFile='SimulatedHostParameters-Ke.txt',
                                               resultName=sourcefolder + WT_calibrated_param_set + '-WTTIP_WTTIP-txset{:02d}-'.format(i) + TIP_metadata_suffix + '.txt',
                                               figureName=sourcefolder + WT_calibrated_param_set + '-WTTIP_WTTIP-txset{:02d}-'.format(i) + TIP_metadata_suffix + '.png')
                except Exception as ex:
                    template = "Skipping parameter set; an exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)
                    print(traceback.format_exc())

                try:
                    visualize_results_WTTIP_TIP(df_result_entry=curr_entry,
                                               hostParameterFile='SimulatedHostParameters-Ke.txt',
                                               resultName=sourcefolder + WT_calibrated_param_set + '-WTTIP_TIP-txset{:02d}-'.format(i) + TIP_metadata_suffix + '.txt',
                                               figureName=sourcefolder + WT_calibrated_param_set + '-WTTIP_TIP-txset{:02d}-'.format(i) + TIP_metadata_suffix + '.png')
                except Exception as ex:
                    template = "Skipping parameter set; an exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)
                    print(traceback.format_exc())

        if doComparisonToNoTIP:
            currfile_WT = sourcefolder + '/20210529_new_recalibration-txset{:02d}.txt'.format(i)
            currfile_TIP = sourcefolder + '/20210529_new_recalibration_parameter_sweep_list-WTTIP_TIP-txset{:02d}-{}.txt'.format(i, TIP_metadata_suffix)
            currfigname = sourcefolder + '/20210529_new_recalibration_parameter_sweep_list-WTTIP_TIP-txset{:02d}-{}_WTonlytx_vs_TIPtransmission.png'.format(i, TIP_metadata_suffix)

            visualize_TIP_impact(currfile_WT, currfile_TIP, currfigname)

        # TO DO - Write a function to just load and plot .txt file results from visualize_result()
        if plot_WTWT_vs_DualWT_result:
            WTWTfile = sourcefolder + '/20210529_new_recalibration-txset{:02d}.txt'.format(i)
            DualWTfile = sourcefolder + '/20210529_new_recalibration_parameter_sweep_list-WTTIP_WT-txset{:02d}-{}.txt'.format(i, TIP_metadata_suffix)
            figureName = sourcefolder + '/20210529_new_recalibration_parameter_sweep_list-WTTIP_WT-txset{:02d}-{}_WT_transmission_TIPvsNoTIP.png'.format(i, TIP_metadata_suffix)
            plot_saved_comparison_results(WTWTfile, DualWTfile, figureName)


if __name__ == '__main__':
    __main__()