import numpy as np
from scipy.integrate import solve_ivp
from numba import jit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
matplotlib.rcParams['axes.labelsize']=16
matplotlib.rcParams['xtick.labelsize']=12
matplotlib.rcParams['ytick.labelsize']=12
matplotlib.rcParams['font.sans-serif']="Arial"
matplotlib.rcParams['font.family']="sans-serif"

def get_data():
    data = pd.read_csv('patient_data.csv',)
    data.columns = pd.Index(['PatientID', 'day', 'log10VL'])

    return data

@jit
def model_ODE(t, y, params):
    gamma, beta, delta = params
    f, V = y

    dydt = np.zeros(2)

    dydt[0] = -beta*f*V

    dydt[1] = gamma*f*V - delta*V

    return dydt

#@jit
def model_ODE_TIP(t, y, params):
    gamma, beta, delta, rho, psi = params
    f, fTIP, V, VTIP = y

    dydt = np.zeros(4)

    dydt[0] = -beta*f*V - beta*f*VTIP

    dydt[1] = +beta*f*VTIP - beta*fTIP*V

    dydt[2] = gamma*f*V - delta*V + psi*gamma*fTIP*V

    dydt[3] = rho*gamma*fTIP*V - delta*VTIP

    return dydt

def simulate_single_inf(params, t0, tf):
    _, gamma, beta, delta, V0, _, _, _, _ = params
    odeparams = (gamma, beta, delta)
    Y0 = [1, V0]
    sol = solve_ivp(lambda t, y: model_ODE(t, y, odeparams),
                    [t0, tf], Y0, t_eval=np.arange(t0, tf+1))
    if not sol.success:
        print('Integration failed. params:', params)
        raise

    times = sol.t
    V = sol.y[1, :]

    cutlog10 = lambda x: np.log10(x) if x > 0 else 0

    log10V = [cutlog10(x) for x in V]

    return times, log10V

def simulate_dual_inf(params, t0, tf):
    _, gamma, beta, delta, V0, _, _, _, _, rho, psi, fracTIP = params
    odeparams = (gamma, beta, delta, rho, psi)
    Y0 = [1-fracTIP, fracTIP, V0, 0]
    sol = solve_ivp(lambda t, y: model_ODE_TIP(t, y, odeparams),
                    [t0, tf], Y0, t_eval=np.arange(t0, tf+1))
    if not sol.success:
        print('Integration failed. params:', params)
        raise

    times = sol.t
    V = sol.y[2, :]
    VTIP = sol.y[3, :]

    cutlog10 = lambda x: np.log10(x) if x > 0 else 0

    log10V = [cutlog10(x) for x in V]
    log10VTIP = [cutlog10(x) for x in VTIP]

    return times, log10V, log10VTIP

def source_params():
    #             [PatientID, gamma, beta, delta, V(0), L, R0, C*, Tp]
    all_params = [['Singapore 2', 3.998, 6.64e-6, 0.71, 6605, 1.41, 5.64, 0.82, 1.9],
                  ['Singapore 3', 3.999, 1.26e-6, 0.43, 6380, 2.35, 9.40, 0.89, 2.4],
                  ['Singapore 4', 3.999, 5.81e-6, 0.72, 6514, 1.39, 5.56, 0.82, 1.9],
                  ['Singapore 6', 3.999, 3.08e-6, 0.47, 6498, 2.11, 8.45, 0.88, 2.1],
                  ['Singapore 8', 3.997, 1.96e-5, 0.36, 6850, 2.75, 11.0, 0.91, 1.6],
                  ['Singapore 9', 3.999, 1.56e-5, 0.28, 6558, 3.56, 14.2, 0.93, 1.7],
                  ['Singapore 11', 3.999, 8.46e-6, 1.38, 6860, 0.72, 2.89, 0.65, 1.8],
                  ['Singapore 12', 3.999, 7.77e-6, 0.78, 6506, 1.28, 5.14, 0.81, 1.8],
                  ['Singapore 14', 3.997, 3.52e-7, 1.03, 5337, 0.97, 3.88, 0.74, 3.0],
                  ['Singapore 16', 3.999, 4.88e-6, 0.54, 6489, 1.85, 7.41, 0.87, 2.0],
                  ['Singapore 17', 3.998, 1.32e-6, 1.01, 5891, 0.99, 3.95, 0.75, 1.7],
                  ['Singapore 18', 3.999, 5.80e-6, 0.38, 6537, 2.61, 10.4, 0.90, 1.9],
                  ['China C', 4.002, 7.95e-6, 1.29, 6596, 0.77, 3.10, 0.68, 1.9],
                  ['China D', 3.999, 1.69e-6, 0.76, 6752, 1.31, 5.26, 0.81, 2.3],
                  ['China E', 3.999, 5.87e-6, 0.84, 6517, 1.19, 4.75, 0.79, 1.9],
                  ['China H', 3.999, 1.05e-5, 1.45, 7059, 0.69, 2.76, 0.64, 1.7],
                  ['China I', 3.999, 6.15e-7, 0.45, 6251, 2.21, 8.85, 0.89, 2.6],
                  ['China L', 3.999, 1.73e-6, 0.77, 6178, 1.29, 5.17, 0.81, 2.3],
                  ['China O', 3.999, 4.57e-5, 1.90, 8541, 0.53, 2.10, 0.52, 0.9],
                  ['China P', 3.997, 1.02e-5, 1.03, 6808, 0.97, 3.88, 0.74, 1.7],
                  ['Germany 1', 3.999, 7.28e-6, 0.98, 6610, 1.02, 4.08, 0.75, 1.9],
                  ['Germany 2', 3.999, 2.88e-6, 1.51, 6262, 0.66, 2.64, 0.62, 2.4],
                  ['Germany 3', 3.999, 1.11e-5, 1.32, 5894, 0.76, 3.02, 0.67, 1.8],
                  ['Germany 4', 3.996, 6.65e-6, 1.60, 6678, 0.63, 2.50, 0.60, 2.0],
                  ['Germany 7', 3.999, 3.08e-6, 1.11, 6259, 0.90, 3.02, 0.72, 2.2],
                  ['Germany 8', 3.999, 3.29e-6, 1.08, 6375, 0.93, 2.50, 0.73, 2.2],
                  ['Germany 10', 3.999, 2.67e-6, 0.61, 6268, 1.64, 3.62, 0.85, 2.2],
                  ['Germany 14', 3.996, 8.91e-6, 1.61, 6350, 0.62, 3.71, 0.60, 1.9],
                  ['Korea 13', 3.999, 9.67e-6, 1.15, 7674, 0.87, 6.56, 0.71, 1.7],
                  ['Korea 15', 3.999, 1.44e-5, 1.16, 5802, 0.86, 2.47, 0.71, 2.5]]
    return all_params

def validate_model_setup():
    # Load patient data
    data = get_data()

    # Load patient parameters
    parameters = source_params()

    # Simulate all patient parameters
    t0 = 0
    tf = 30

    pltrows = 6
    pltcols = 5
    fig, ax = plt.subplots(nrows=pltrows, ncols=pltcols, figsize=(10, 10))

    all_t = np.zeros((len(parameters), 31))
    all_log10V = np.zeros((len(parameters), 31))
    for index, paramset in enumerate(parameters):
        print('Working on:', paramset[0])
        t, log10V = simulate_single_inf(paramset, t0, tf)
        all_t[index, :] = t
        all_log10V[index, :] = log10V

        # LOD was region-dependent
        if 'Korea' in paramset[0] or 'Singapore' in paramset[0]:
            LOD = np.log10(68)
        elif 'China' in paramset[0]:
            LOD = np.log10(15)
        elif 'Germany' in paramset[0]:
            LOD = np.log10(33.3)

        # Plot simulation results vs. patient data
        pltrow = int(np.floor(index/pltcols))
        pltcol = np.mod(index, pltcols)

        # Extract current patient data.
        observed_t = data[data['PatientID'] == paramset[0]]['day'].values
        observed_log10VL = data[data['PatientID'] == paramset[0]]['log10VL'].values
        observed_t = observed_t.astype('float64')
        observed_log10VL = observed_log10VL.astype('float64')

        ax[pltrow, pltcol].plot(observed_t, observed_log10VL, 'ok')
        ax[pltrow, pltcol].plot(t, log10V, 'r-')
        ax[pltrow, pltcol].plot([0, 30], [LOD, LOD], 'k--')
        ax[pltrow, pltcol].set_title(paramset[0])
        ax[pltrow, pltcol].axis((-1, 31, -0.5, 8.5))
        ax[pltrow, pltcol].set_xticks((0, 10, 20, 30))
        ax[pltrow, pltcol].set_yticks((0, 2, 4, 6, 8))

    ax[5, 2].set_xlabel('days past symptom onset', fontsize=20)
    test=ax[3, 0].set_ylabel('log10 viral load', fontsize=20)
    test.set_y(1.3)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1,
                        hspace=0.5, wspace=0.4)
    plt.savefig('model_validation.png', dpi=300)
    plt.savefig('model_validation.pdf', dpi=300, transparent=True)

def predict_TIP_effect(figurename, rho=1.5, psi=0.02, fracTIP=0.5):
    # Load patient parameters
    parameters = source_params()

    # Simulate all patient parameters
    t0 = 0
    tf = 30

    pltrows = 6
    pltcols = 5
    fig, ax = plt.subplots(nrows=pltrows, ncols=pltcols, figsize=(10, 10))

    all_t = np.zeros((len(parameters), 31))
    all_log10V = np.zeros((len(parameters), 31))
    all_log10V_TIPtreat = np.zeros((len(parameters), 31))

    log10_peak_difference = np.zeros((len(parameters)))
    log10_AUC_difference = np.zeros((len(parameters)))

    for index, paramset in enumerate(parameters):
        print('Working on:', paramset[0])
        paramset_tmp = paramset.copy()

        t, log10V = simulate_single_inf(paramset, t0, tf)
        all_t[index, :] = t
        all_log10V[index, :] = log10V

        paramset.append(rho)
        paramset.append(psi)
        paramset.append(fracTIP)

        _, log10V_TIPtreatment, _ = simulate_dual_inf(paramset, t0, tf)
        all_log10V_TIPtreat[index, :] = log10V_TIPtreatment

        # Find peak value
        log10_peak_difference[index] = np.max(log10V) - np.max(log10V_TIPtreatment)
        log10_AUC_difference[index] = np.log10(np.sum(np.power(10, log10V))) - \
                                      np.log10(np.sum(np.power(10, log10V_TIPtreatment)))

        # LOD was region-dependent
        if 'Korea' in paramset[0] or 'Singapore' in paramset[0]:
            LOD = np.log10(68)
        elif 'China' in paramset[0]:
            LOD = np.log10(15)
        elif 'Germany' in paramset[0]:
            LOD = np.log10(33.3)

        # Plot simulation results vs. patient data
        pltrow = int(np.floor(index/pltcols))
        pltcol = np.mod(index, pltcols)


        ax[pltrow, pltcol].plot(t, log10V, color='#000000')
        ax[pltrow, pltcol].plot(t, log10V_TIPtreatment, color='#336699', linewidth=2)
        ax[pltrow, pltcol].plot([0, 30], [LOD, LOD], 'k--')
        ax[pltrow, pltcol].set_title(paramset[0])
        ax[pltrow, pltcol].axis((-1, 31, -0.5, 8.5))
        ax[pltrow, pltcol].set_xticks((0, 10, 20, 30))
        ax[pltrow, pltcol].set_yticks((0, 2, 4, 6, 8))


    ax[5, 2].set_xlabel('days past symptom onset', fontsize=20)
    test=ax[3, 0].set_ylabel('log10 viral load', fontsize=20)
    test.set_y(1.3)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1,
                        hspace=0.5, wspace=0.4)
    plt.savefig(figurename+'.png', dpi=300)
    plt.savefig(figurename+'.pdf', dpi=300, transparent=True)

def predict_TIP_effect_focusInd(figurename, individual_name, rho=1.5, psi=0.02, fracTIP=0.5):
    # Load patient parameters
    parameters = source_params()

    # Simulate all patient parameters
    t0 = 0
    tf = 30

    plt.figure(figsize=(7*0.65, 6*0.65))
    ax=plt.gca()
    WT_only_color = '#000000'
    WT_TIP_color = '#336699'

    for index, paramset in enumerate(parameters):
        if individual_name in paramset[0]:
            print('Working on:', paramset[0])
        else:
            continue

        paramset_tmp = paramset.copy()

        t, log10V = simulate_single_inf(paramset, t0, tf)

        paramset_tmp.append(rho)
        paramset_tmp.append(psi)
        paramset_tmp.append(fracTIP)

        _, log10V_TIPtreatment, _ = simulate_dual_inf(paramset_tmp, t0, tf)

        # LOD was region-dependent
        if 'Korea' in paramset[0] or 'Singapore' in paramset[0]:
            LOD = np.log10(68)
        elif 'China' in paramset[0]:
            LOD = np.log10(15)
        elif 'Germany' in paramset[0]:
            LOD = np.log10(33.3)

        ax.plot(t, log10V, color=WT_only_color, linewidth=2.5)
        ax.plot(t, log10V_TIPtreatment, color=WT_TIP_color, linewidth=2.5)
        ax.plot([0, 30], [LOD, LOD], 'k--')
        #ax.set_title(paramset[0])
        ax.axis((-1, 31, -0.5, 8.5))
        ax.set_xticks((0, 10, 20, 30))
        ax.set_yticks((0, 2, 4, 6, 8))
        ax.set_xticklabels(('0', '10', '20', '30'), fontsize=14)
        ax.set_yticklabels(('0', '2', '4', '6', '8'), fontsize=14)

    ax.set_xlabel('days past symptom onset', fontsize=20)
    ax.set_ylabel('log10 viral load', fontsize=20)
    plt.tight_layout(pad=1.2)
    plt.savefig(figurename+'.png', dpi=300)
    plt.savefig(figurename+'.pdf', dpi=300, transparent=True)


def TIP_heatmap(filebasename, fracTIP=0.5):
    # Load patient parameters
    parameters = source_params()

    psis = np.array([np.power(10, x) for x in np.arange(-8, 1, 1, dtype=float)])
    rhos = np.array([np.power(10, x) for x in np.arange(-6, 3, 1, dtype=float)])

    # Simulate all patient parameters
    t0 = 0
    tf = 30

    pltrows = 6
    pltcols = 5
    fig, ax = plt.subplots(nrows=pltrows, ncols=pltcols, figsize=(10, 10))

    log10_peak_difference = np.zeros((len(psis),len(rhos), len(parameters)))
    log10_AUC_difference = np.zeros((len(psis),len(rhos), len(parameters)))

    for psi_index, psi in enumerate(psis):
        for rho_index, rho in enumerate(rhos):
            for index, paramset in enumerate(parameters):
                print('Working on: psi=', psi, 'rho=', rho, ' | ', paramset[0])
                paramset_tmp = paramset.copy()

                t, log10V = simulate_single_inf(paramset, t0, tf)

                paramset_tmp.append(rho)
                paramset_tmp.append(psi)
                paramset_tmp.append(fracTIP)

                _, log10V_TIPtreatment, _ = simulate_dual_inf(paramset_tmp, t0, tf)

                # Find peak value
                log10_peak_difference[psi_index, rho_index, index] = np.max(log10V) - np.max(log10V_TIPtreatment)
                log10_AUC_difference[psi_index, rho_index, index] = np.log10(np.sum(np.power(10, log10V))) - \
                                                                    np.log10(np.sum(np.power(10, log10V_TIPtreatment)))
    # Construct a pandas dataframe
    df = pd.DataFrame(columns=('log10Difference', 'log10AUCDifference', 'fractionTIP', 'patientID', 'VLcompartment', 'psi', 'rho'))
    for i in range(len(psis)):
        for j in range(len(rhos)):
            for k, paramset in enumerate(parameters):
                df = df.append({'log10Difference': log10_peak_difference[i, j, k],
                                'log10AUCDifference': log10_AUC_difference[i, j, k],
                                'fractionTIP': fracTIP,
                                'patientID': paramset[0],
                                'psi': psis[i],
                                'rho': rhos[j]},
                               ignore_index=True)

    df.to_csv(filebasename+'.csv')

def get_change_peak_VL_heatmap_fromfile(filebasename, csvfile):
    fractionTIP = 0.5
    psis = np.array([np.power(10, x) for x in np.arange(-8, 1, 1, dtype=float)])
    rhos = np.array([np.power(10, x) for x in np.arange(-6, 3, 1, dtype=float)])

    parameters = source_params()

    log10_peak_difference = np.zeros((len(psis),len(rhos), len(parameters)))
    log10_AUC_difference = np.zeros((len(psis),len(rhos), len(parameters)))

    df = pd.read_csv(csvfile)

    for i in range(len(psis)):
        for j in range(len(rhos)):
            for k, paramset in enumerate(parameters):
                log10_peak_difference[i, j, k] = df[(df['patientID'] == paramset[0]) &
                                                    ratio_epsilon_equals(df['psi'], psis[i]) &
                                                    ratio_epsilon_equals(df['rho'], rhos[j])]['log10Difference'].values[0]
                log10_AUC_difference[i, j, k] = df[(df['patientID'] == paramset[0]) &
                                                    ratio_epsilon_equals(df['psi'], psis[i]) &
                                                    ratio_epsilon_equals(df['rho'], rhos[j])]['log10AUCDifference'].values[0]

    peak_diff_median = np.nanmedian(log10_peak_difference, axis=2)
    AUC_diff_median = np.nanmedian(log10_AUC_difference, axis=2)

    sns.set_style(rc={'font.family': 'sans-serif',
                      'font.sans-serif': 'Arial',
                      'pdf.fonttype': 42,
                      'ps.fonttype': 42,
                      'axes.labelsize': 16,
                      'xtick.labelsize':16,
                      'ytick.labelsize':16})
    plt.figure()
    ax=plt.gca()

    axsns=sns.heatmap(peak_diff_median, ax=ax,
                      vmin=0,#np.min(LRTdiff_median[LRTdiff_median > (0.0001)]),
                      vmax=np.max(peak_diff_median[peak_diff_median > (0.0001)]),
                      cbar_kws={'label': 'Median reduction in\n peak log10 SARS-CoV-2'},
                      cmap=sns.color_palette("rocket", as_cmap=True),
                      fmt='')
    cbarax = axsns.figure.axes[-1]
    cbarax.yaxis.label.set_size(15)

    psi_labels= [r'$\mathregular{10^{-8}}$', r'$\mathregular{10^{-7}}$',
                 r'$\mathregular{10^{-6}}$', r'$\mathregular{10^{-5}}$',
                 r'$\mathregular{10^{-4}}$', r'$\mathregular{10^{-3}}$',
                 r'$\mathregular{10^{-2}}$', r'$\mathregular{10^{-1}}$',
                 r'$\mathregular{10^{0}}$']
    rho_labels= [r'$\mathregular{10^{-6}}$', r'$\mathregular{10^{-5}}$',
                 r'$\mathregular{10^{-4}}$', r'$\mathregular{10^{-3}}$',
                 r'$\mathregular{10^{-2}}$', r'$\mathregular{10^{-1}}$',
                 r'$\mathregular{10^{0}}$',  r'$\mathregular{10^{1}}$',
                 r'$\mathregular{10^{2}}$']
    ax.set_xticklabels(rho_labels, fontsize=12)
    ax.set_yticklabels(psi_labels, fontsize=12, rotation=0)
    ax.set_xlabel('Ratio of TIP to SARS-CoV-2 burst size\n(in TIP-carrier cell)', fontsize=16)
    ax.set_ylabel('Ratio of SARS-CoV-2 burst size\n(TIP-carrier cell vs. non-TIP-carrier)', fontsize=16)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(filebasename+'_heatmap.png', dpi=300)
    plt.savefig(filebasename+'_heatmap.pdf', dpi=300, transparent=True)

    plt.figure()
    ax=plt.gca()

    axsns=sns.heatmap(AUC_diff_median, ax=ax,
                      vmin=0,#np.min(LRTdiff_median[LRTdiff_median > (0.0001)]),
                      vmax=np.max(AUC_diff_median[AUC_diff_median > (0.0001)]),
                      cbar_kws={'label': 'Median reduction in\n integrated log10 SARS-CoV-2'},
                      cmap=sns.color_palette("rocket", as_cmap=True),
                      fmt='')
    cbarax = axsns.figure.axes[-1]
    cbarax.yaxis.label.set_size(15)

    ax.set_xticklabels(rho_labels, fontsize=12)
    ax.set_yticklabels(psi_labels, fontsize=12, rotation=0)
    ax.set_xlabel('Ratio of TIP to SARS-CoV-2 burst size\n(in TIP-carrier cell)', fontsize=16)
    ax.set_ylabel('Ratio of SARS-CoV-2 burst size\n(TIP-carrier cell vs. non-TIP-carrier)', fontsize=16)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(filebasename+'_heatmap_AUC.png', dpi=300)
    plt.savefig(filebasename+'_heatmap_AUC.pdf', dpi=300, transparent=True)


def TIP_heatmap_zoom_upper_right(filebasename, fracTIP=0.5):
    # Load patient parameters
    parameters = source_params()

    psis = np.array([np.power(10, x) for x in np.arange(-3, 0.5, 0.5, dtype=float)])
    rhos = np.array([np.power(10, x) for x in np.arange(-1, 2.5, 0.5, dtype=float)])

    # Simulate all patient parameters
    t0 = 0
    tf = 30

    pltrows = 6
    pltcols = 5
    fig, ax = plt.subplots(nrows=pltrows, ncols=pltcols, figsize=(10, 10))

    log10_peak_difference = np.zeros((len(psis),len(rhos), len(parameters)))
    log10_AUC_difference = np.zeros((len(psis),len(rhos), len(parameters)))

    for psi_index, psi in enumerate(psis):
        for rho_index, rho in enumerate(rhos):
            for index, paramset in enumerate(parameters):
                print('Working on: psi=', psi, 'rho=', rho, ' | ', paramset[0])
                paramset_tmp = paramset.copy()

                t, log10V = simulate_single_inf(paramset, t0, tf)

                paramset_tmp.append(rho)
                paramset_tmp.append(psi)
                paramset_tmp.append(fracTIP)

                _, log10V_TIPtreatment, _ = simulate_dual_inf(paramset_tmp, t0, tf)

                # Find peak value
                log10_peak_difference[psi_index, rho_index, index] = np.max(log10V) - np.max(log10V_TIPtreatment)
                log10_AUC_difference[psi_index, rho_index, index] = np.log10(np.sum(np.power(10, log10V))) - \
                                                                    np.log10(np.sum(np.power(10, log10V_TIPtreatment)))

    # Construct a pandas dataframe
    df = pd.DataFrame(columns=('log10Difference', 'log10AUCDifference', 'fractionTIP', 'patientID', 'VLcompartment', 'psi', 'rho'))
    for i in range(len(psis)):
        for j in range(len(rhos)):
            for k, paramset in enumerate(parameters):
                df = df.append({'log10Difference': log10_peak_difference[i, j, k],
                                'log10AUCDifference': log10_AUC_difference[i, j, k],
                                'fractionTIP': fracTIP,
                                'patientID': paramset[0],
                                'psi': psis[i],
                                'rho': rhos[j]},
                               ignore_index=True)

    df.to_csv(filebasename+'.csv')

def get_change_peak_VL_heatmap_fromfile_zoom_upper_right(filebasename, csvfile):
    fractionTIP = 0.5
    psis = np.array([np.power(10, x) for x in np.arange(-3, 0.5, 0.5, dtype=float)])
    rhos = np.array([np.power(10, x) for x in np.arange(-1, 2.5, 0.5, dtype=float)])

    parameters = source_params()

    log10_peak_difference = np.zeros((len(psis),len(rhos), len(parameters)))
    log10_AUC_difference = np.zeros((len(psis),len(rhos), len(parameters)))

    df = pd.read_csv(csvfile)

    for i in range(len(psis)):
        for j in range(len(rhos)):
            for k, paramset in enumerate(parameters):
                log10_peak_difference[i, j, k] = df[(df['patientID'] == paramset[0]) &
                                                    ratio_epsilon_equals(df['psi'], psis[i]) &
                                                    ratio_epsilon_equals(df['rho'], rhos[j])]['log10Difference'].values[0]
                log10_AUC_difference[i, j, k] = df[(df['patientID'] == paramset[0]) &
                                                    ratio_epsilon_equals(df['psi'], psis[i]) &
                                                    ratio_epsilon_equals(df['rho'], rhos[j])]['log10AUCDifference'].values[0]

    peak_diff_median = np.nanmedian(log10_peak_difference, axis=2)
    AUC_diff_median = np.nanmedian(log10_AUC_difference, axis=2)

    sns.set_style(rc={'font.family': 'sans-serif',
                      'font.sans-serif': 'Arial',
                      'pdf.fonttype': 42,
                      'ps.fonttype': 42,
                      'axes.labelsize': 16,
                      'xtick.labelsize':16,
                      'ytick.labelsize':16})
    plt.figure()
    ax=plt.gca()

    axsns=sns.heatmap(peak_diff_median, ax=ax,
                      vmin=0,#np.min(LRTdiff_median[LRTdiff_median > (0.0001)]),
                      vmax=np.max(peak_diff_median[peak_diff_median > (0.0001)]),
                      cbar_kws={'label': 'Median reduction in\n peak log10 SARS-CoV-2'},
                      cmap=sns.color_palette("rocket", as_cmap=True),
                      fmt='')
    cbarax = axsns.figure.axes[-1]
    cbarax.yaxis.label.set_size(15)
    psi_labels= [r'$\mathregular{10^{-3}}$', r'$\mathregular{10^{-2.5}}$',
                 r'$\mathregular{10^{-2}}$', r'$\mathregular{10^{-1.5}}$',
                 r'$\mathregular{10^{-1}}$', r'$\mathregular{10^{-0.5}}$',
                 r'$\mathregular{10^{0}}$']
    rho_labels= [r'$\mathregular{10^{-1}}$', r'$\mathregular{10^{-0.5}}$',
                 r'$\mathregular{10^{0}}$', r'$\mathregular{10^{0.5}}$',
                 r'$\mathregular{10^{1}}$', r'$\mathregular{10^{1.5}}$',
                 r'$\mathregular{10^{2}}$']
    ax.set_xticklabels(rho_labels, fontsize=12)
    ax.set_yticklabels(psi_labels, fontsize=12, rotation=0)
    ax.set_xlabel('Ratio of TIP to SARS-CoV-2 burst size\n(in TIP-carrier cell)', fontsize=16)
    ax.set_ylabel('Ratio of SARS-CoV-2 burst size\n(TIP-carrier cell vs. non-TIP-carrier)', fontsize=16)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(filebasename+'_heatmap.png', dpi=300)
    plt.savefig(filebasename+'_heatmap.pdf', dpi=300, transparent=True)

    plt.figure()
    ax=plt.gca()

    axsns=sns.heatmap(AUC_diff_median, ax=ax,
                      vmin=0,#np.min(LRTdiff_median[LRTdiff_median > (0.0001)]),
                      vmax=np.max(AUC_diff_median[AUC_diff_median > (0.0001)]),
                      cbar_kws={'label': 'Median reduction in\n integrated log10 SARS-CoV-2'},
                      cmap=sns.color_palette("rocket", as_cmap=True),
                      fmt='')
    cbarax = axsns.figure.axes[-1]
    cbarax.yaxis.label.set_size(15)

    ax.set_xticklabels(rho_labels, fontsize=12)
    ax.set_yticklabels(psi_labels, fontsize=12, rotation=0)
    ax.set_xlabel('Ratio of TIP to SARS-CoV-2 burst size\n(in TIP-carrier cell)', fontsize=16)
    ax.set_ylabel('Ratio of SARS-CoV-2 burst size\n(TIP-carrier cell vs. non-TIP-carrier)', fontsize=16)
    ax.invert_yaxis()
    plt.tight_layout()
    #plt.show()
    plt.savefig(filebasename+'_heatmap_AUC.png', dpi=300)
    plt.savefig(filebasename+'_heatmap_AUC.pdf', dpi=300, transparent=True)


def ratio_epsilon_equals(x,y,eps=1e-10):
    # Return True if the ratio x/y is within eps of 1.
    return (1-(x/y))<eps

def __main__():
    # Simulate model, visually compare to results from Kim et al. 2020
    validate_model_setup()

    # Simulate model to compare untreated and TIP-treated patients. Saves a .csv file.
    TIP_heatmap('TIP_heatmap_results', fracTIP=0.5)

    # Replot the simulated model results by loading from csv -- can comment out the simulation call above if already have the .csv
    get_change_peak_VL_heatmap_fromfile('TIP_heatmap_effect', 'TIP_heatmap_results.csv')

    # Simulate model for the upper right corner zoom, as above. Saves a .csv file.
    TIP_heatmap_zoom_upper_right('TIP_heatmap_results_zoom_upper_right', fracTIP=0.5)
    get_change_peak_VL_heatmap_fromfile_zoom_upper_right('TIP_heatmap_effect_zoom_upper_right', 'TIP_heatmap_results_zoom_upper_right.csv')

    # Simulate model for the representative individual(s).
    predict_TIP_effect('TIP_effect_psi0-02_rho1-5',rho=1.5, psi=0.02, fracTIP=0.5)
    predict_TIP_effect_focusInd('Singapore 17 result', 'Singapore 17')

if __name__ == '__main__':
    __main__()
