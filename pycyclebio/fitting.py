import math
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, f
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt


def fourier_square_wave(t, a, gamma, omega, phi,  y):

    return a * np.exp(gamma * t) * (1 + (4/math.pi)*np.sum(
        np.sin(math.pi*omega*(t+phi))+np.sin(math.pi*3*omega*(t+phi))/3)) + y


def fourier_cycloid_wave(t, a, gamma, omega, phi, y):
    return a * np.exp(gamma * t) * (2/math.pi - (4 / math.pi) * np.sum(
        np.cos(2 * omega * (t + phi))/3 + (np.cos(4 * omega * (t + phi)) / (4*2 ^ 2-1)))) + y


def fourier_sawtooth_wave(t, a, gamma, omega, phi, y):
    return a * np.exp(gamma * t) * (0.5 - (1/math.pi) * np.sum(
        np.sin(2*math.pi*omega*(t+phi)) + np.sin(4*math.pi*omega*(t+phi))/2)) + y


def p_harmonic_oscillator(t, a, gamma, omega, phi, y):
    return a * np.exp(gamma*t)*np.cos((omega*t)+phi)+y


def p_square_wave(t, a, gamma, omega, phi, y):
    return a * np.exp(gamma * t) * (np.sin((omega*t)+phi) + 0.25*np.sin(((omega*t)+phi)*3.0)) + y


def p_cycloid_wave(t, a, gamma, omega, phi, y):
    return a * np.exp(gamma * t) * (np.cos((omega*t) + phi)-(np.cos(2*((omega*t)+phi)))/2) + y


def p_transient_impulse(t, a, gamma,  period, width, phi,  y):
    p_tau = (period/24) * (2.0*np.pi)
    t_mod = np.mod(t - phi, 2*np.pi - 1e-7)
    impulse = np.where(t_mod - p_tau >= 0, np.exp(-0.5*((t_mod - p_tau)/width)**2), 0.0)
    return a * np.exp(gamma * t) * impulse + y


def calculate_variances(data):
    # Extract ZT times and replicate numbers from the column names
    zt_replicates = data.index.str.extract(r'(\d+)_\D*(\d+)')
    zt_times = zt_replicates[0].str.extract(r'(\d+)').astype(int)[0].values

    # Group by ZT times and calculate variances
    variances = {}
    for i, zt in enumerate(np.unique(zt_times)):
        # Find columns corresponding to this ZT time
        zt_columns = data.index[zt_times == zt]
        # Calculate variance across these columns, ignoring NaNs
        zt_var = data[zt_columns].var(ddof=1)
        variances[zt] = zt_var if zt_var else 0  # Replace NaN variances with 0
    return variances


def lack_of_fit_f(timepoints, y_obs, y_fit, params, reps):
    n = len(y_obs)
    p = len(params)
    k = len(np.unique(timepoints))

    sspure = 0
    sslof = 0
    lof_degfre = k - p
    pure_degfre = n - k
    timepoint_bins = int(len(timepoints)/reps)
    for i in range(0, timepoint_bins):
        y_at_t = np.array(y_obs[int(reps*range(0, timepoint_bins)[i]):int(reps*range(0, timepoint_bins)[i]+reps)])
        y_mean_at_t = np.array([np.mean(y_at_t)]*int(reps))
        t_sspure = np.sum((y_at_t - y_mean_at_t)**2)
        y_fit_at_t = np.array(y_fit[int(reps*range(0, timepoint_bins)[i]):int(reps*range(0, timepoint_bins)[i]+reps)])
        t_sslof = np.sum((y_mean_at_t - y_fit_at_t)**2)
        sspure = sspure + t_sspure
        sslof = sslof + t_sslof
    mean_sspure = sspure / pure_degfre
    mean_sslof = sslof / lof_degfre
    f_stat = mean_sslof / mean_sspure
    p_value = 1 - f.cdf(f_stat, lof_degfre, pure_degfre)
    return f_stat, p_value


# noinspection DuplicatedCode
def fit_best_waveform(df_row, period, models, timepoints, reps, lbound):
    """
    Fits all three waveform models to the data and determines the best fit.

    :param df_row: A DataFrame row containing the data to fit.
    :param period: (float) The period of oscillatory behaviour of interest
    :param models: (list) a list of models to fit. Typically not changed by user.
    :param timepoints: (array) timepoints, extracted from column labels of input df.
    :param reps: (int) number of replicates per timepoint, calculated from column labels.
    :param lbound: (binary) boolean to identify if dataframe contains normalised (negative) values,
    which require models to be rebounded.
    :return: A tuple containing the best-fit parameters, the waveform type, and the covariance of the fit.
    """
    amplitudes = df_row.values

    if lbound == 1:  # 1 = neg values present
        lbound = np.min(amplitudes)
    if lbound == 0:  # 0 = neg values absent
        lbound = 0
    cyclbound = np.abs(amplitudes[np.argmax(np.abs(amplitudes))])

#     variances = calculate_variances(df_row)
#     weights = np.array([1 / variances[tp] if tp in variances and variances[tp] != 0 else 0.0001 for tp in timepoints]
    # 0 variance messes model selection up, so a negligible value is used in above line

    # Fit extended harmonic oscillator
    # (t, a, gamma, omega, phi, y):
    if 'harmonic' in models:
        harmonic_initial_params = [np.median(amplitudes), 0, 1, 0, np.mean(amplitudes)]
        lower_bounds = [lbound, -0.2, 0.9, -period/2, -np.abs(amplitudes[np.argmax(np.abs(amplitudes))])]
        upper_bounds = [np.max(amplitudes), 0.2, 1.1, period/2, np.max(amplitudes)]
        harmonic_bounds = (lower_bounds, upper_bounds)
        try:
            harmonic_res = curve_fit(
                p_harmonic_oscillator,
                timepoints,
                amplitudes,
                bounds=harmonic_bounds,
                # sigma=weights,
                p0=harmonic_initial_params,
                maxfev=10000000,
                ftol=0.0001,
                xtol=0.0001
            )
            harmonic_params = harmonic_res[0]
            harmonic_covariance = harmonic_res[1]
            harmonic_fitted_values = p_harmonic_oscillator(timepoints, *harmonic_params)
            harmonic_residuals = amplitudes - harmonic_fitted_values
            harmonic_sse = np.sum(harmonic_residuals ** 2)
            harmonic_rmse = np.sqrt(harmonic_sse/len(harmonic_residuals))
        except RuntimeError:
            harmonic_params = np.nan
            harmonic_covariance = np.nan
            harmonic_fitted_values = [0] * len(df_row)
            harmonic_sse = np.inf
            harmonic_rmse = np.inf
    else:
        harmonic_params = np.nan
        harmonic_covariance = np.nan
        harmonic_fitted_values = [0] * len(df_row)
        harmonic_sse = np.inf
        harmonic_rmse = np.inf

    if 'square' in models:
        # Fit square oscillator
        # (t, a, gamma, omega, phi, y):
        square_initial_params = [np.median(amplitudes), 0, 1, 0, np.mean(amplitudes)]
        square_lower_bounds = [lbound, -0.2, 0.9, -period/2, -np.abs(amplitudes[np.argmax(np.abs(amplitudes))])]
        square_upper_bounds = [np.max(amplitudes), 0.2, 1.1, period/2, np.max(amplitudes)]
        square_bounds = (square_lower_bounds, square_upper_bounds)
        try:
            square_res = curve_fit(
                p_square_wave,
                timepoints,
                amplitudes,
                bounds=square_bounds,
                # sigma=weights,
                p0=square_initial_params,
                maxfev=10000000,
                ftol=0.0001,
                xtol=0.0001
            )
            square_params = square_res[0]
            square_covariance = square_res[1]
            square_fitted_values = p_square_wave(timepoints, *square_params)
            square_residuals = amplitudes - square_fitted_values
            square_sse = np.sum(square_residuals ** 2)
            square_rmse = np.sqrt(square_sse/len(square_residuals))
        except RuntimeError:
            square_params = np.nan
            square_covariance = np.nan
            square_fitted_values = [0] * len(df_row)
            square_sse = np.inf
            square_rmse = np.inf
    else:
        square_params = np.nan
        square_covariance = np.nan
        square_fitted_values = [0] * len(df_row)
        square_sse = np.inf
        square_rmse = np.inf

    if 'cycloid' in models:
        # Fit cycloid oscillators
        # (t, a, gamma, omega, phi, y):
        cycloid_initial_params = [np.median(amplitudes), 0, 1, 0, np.mean(amplitudes)]
        cycloid_lower_bounds = [-cyclbound, -0.2, 0.9, -period/2,
                                -cyclbound]
        cycloid_upper_bounds = [cyclbound, 0.2, 1.1, period/2, cyclbound]
        cycloid_bounds = (cycloid_lower_bounds, cycloid_upper_bounds)
        try:
            cycloid_res = curve_fit(
                p_cycloid_wave,
                timepoints,
                amplitudes,
                bounds=cycloid_bounds,
                # sigma=weights,
                p0=cycloid_initial_params,
                maxfev=10000000,
                ftol=0.0001,
                xtol=0.0001
            )
            cycloid_params = cycloid_res[0]
            cycloid_covariance = cycloid_res[1]
            cycloid_fitted_values = p_cycloid_wave(timepoints, *cycloid_params)
            cycloid_residuals = amplitudes - cycloid_fitted_values
            cycloid_sse = np.sum(cycloid_residuals ** 2)
            cycloid_rmse = np.sqrt(cycloid_sse/len(cycloid_residuals))
        except RuntimeError:
            cycloid_params = np.nan
            cycloid_covariance = np.nan
            cycloid_fitted_values = [0] * len(df_row)
            cycloid_sse = np.inf
            cycloid_rmse = np.inf
    else:
        cycloid_params = np.nan
        cycloid_covariance = np.nan
        cycloid_fitted_values = [0] * len(df_row)
        cycloid_sse = np.inf
        cycloid_rmse = np.inf

    if 'transient' in models:
        # Fit transient oscillator
        # (t, a, gamma,  period, width, phi,  y):

        # Lower bounds of p and w need to be adjusted with experimental resolution (in extreme cases),
        # if they are too small compared to measurements they will produce a flat line
        # (transient occurring for very small duration between points) which breaks the statistical corrections
        transient_initial_params = [np.median(amplitudes), 0, 1, 1, 0,  np.min(amplitudes)]
        transient_lower_bounds = [lbound, -0.2, 0.1, 0.1, -period, lbound]
        transient_upper_bounds = [np.max(amplitudes), 0.2, 24, 4, period, np.max(amplitudes)]
        transient_bounds = (transient_lower_bounds, transient_upper_bounds)
        try:
            transient_res = curve_fit(
                p_transient_impulse,
                timepoints,
                amplitudes,
                bounds=transient_bounds,
                # sigma=weights,
                p0=transient_initial_params,
                maxfev=10000000,
                ftol=0.0001,
                xtol=0.0001
            )
            transient_params = transient_res[0]
            transient_covariance = transient_res[1]
            transient_fitted_values = p_transient_impulse(timepoints, *transient_params)
            transient_residuals = amplitudes - transient_fitted_values
            transient_sse = np.sum(transient_residuals ** 2)
            transient_rmse = np.sqrt(transient_sse/len(transient_residuals))
        except RuntimeError:
            transient_params = np.nan
            transient_covariance = np.nan
            transient_fitted_values = [0] * len(df_row)
            transient_sse = np.inf
            transient_rmse = np.inf
    else:
        transient_params = np.nan
        transient_covariance = np.nan
        transient_fitted_values = [0] * len(df_row)
        transient_sse = np.inf
        transient_rmse = np.inf
    # Determine best fit
    sse_values = [harmonic_sse, square_sse, cycloid_sse, transient_sse]
    best_fit_index = np.argmin(sse_values)
    if sse_values == [np.inf, np.inf, np.inf, np.inf]:
        best_params = np.nan
        best_waveform = 'unsolved'
        best_covariance = np.nan
        best_fitted_values = np.nan
        best_rmse = np.nan
    elif best_fit_index == 0:
        test_params = copy.deepcopy(harmonic_params)
        test_params[0] = 0
        test_values = p_harmonic_oscillator(timepoints, test_params[0], test_params[1], test_params[2], test_params[3],
                                            test_params[4])
        if lack_of_fit_f(timepoints, amplitudes, test_values, test_params, reps)[1] > 0.05:
            # Note, a cutoff og 0.05 seemed extremely strict.
            # Relaxed to .2 but should look for non-arbitrary ways of doing this
            best_params = np.nan
            best_waveform = 'non-rhythmic'
            best_covariance = np.nan
            best_fitted_values = np.nan
            best_rmse = np.nan
        else:
            best_params = harmonic_params
            best_waveform = 'sinusoidal'
            best_covariance = harmonic_covariance
            best_fitted_values = harmonic_fitted_values
            best_rmse = harmonic_rmse
    elif best_fit_index == 1:
        test_params = copy.deepcopy(square_params)
        test_params[0] = 0
        test_values = p_square_wave(timepoints, test_params[0], test_params[1], test_params[2], test_params[3],
                                    test_params[4])
        if lack_of_fit_f(timepoints, amplitudes, test_values, test_params, reps)[1] > 0.05:
            best_params = np.nan
            best_waveform = 'non-rhythmic'
            best_covariance = np.nan
            best_fitted_values = np.nan
            best_rmse = np.nan
        else:
            best_params = square_params
            best_waveform = 'square'
            best_covariance = square_covariance
            best_fitted_values = square_fitted_values
            best_rmse = square_rmse
    elif best_fit_index == 2:
        test_params = copy.deepcopy(cycloid_params)
        test_params[0] = 0
        test_values = p_cycloid_wave(timepoints, test_params[0], test_params[1], test_params[2], test_params[3],
                                     test_params[4])
        if lack_of_fit_f(timepoints, amplitudes, test_values, test_params, reps)[1] > 0.05:
            best_params = np.nan
            best_waveform = 'non-rhythmic'
            best_covariance = np.nan
            best_fitted_values = np.nan
            best_rmse = np.nan
        else:
            best_params = cycloid_params
            best_waveform = 'cycloid'
            best_covariance = cycloid_covariance
            best_fitted_values = cycloid_fitted_values
            best_rmse = cycloid_rmse
    else:
        test_params = copy.deepcopy(transient_params)
        test_params[0] = 0
        test_values = p_transient_impulse(timepoints, test_params[0], test_params[1], test_params[2], test_params[3],
                                          test_params[4], test_params[5])
        if lack_of_fit_f(timepoints, amplitudes, test_values, test_params, reps)[1] > 0.05:
            best_params = np.nan
            best_waveform = 'non-rhythmic'
            best_covariance = np.nan
            best_fitted_values = np.nan
            best_rmse = np.nan
        else:
            best_params = transient_params
            best_waveform = 'transient'
            best_covariance = transient_covariance
            best_fitted_values = transient_fitted_values
            best_rmse = transient_rmse

    return best_waveform, best_params, best_covariance, best_fitted_values, best_rmse


def categorize_rhythm(gamma):
    """
    Categorizes the rhythm based on the value of γ.

    :param gamma: The γ value from the fitted parameters.
    :return: A string describing the rhythm category.
    """
    if 0.15 >= gamma >= 0.03:
        return 'forced'
    elif -0.15 <= gamma <= -0.03:
        return 'damped'
    elif -0.03 <= gamma <= 0.03:
        return 'stable'
    else:
        return 'overexpressed' if gamma > 0.15 else 'repressed'


def variance_based_filtering(df, min_feature_variance=0.05):
    """Variance-based filtering of features
    Arguments:

    :param df: (dataframe) containing molecules by row and samples by columns
    :param min_feature_variance: (float) Minimum variance to include a feature in the analysis; default: 5%
    Returns:

    :return variant_df (DataFrame): DataFrame with variant molecules (variance > min_feature_variance)
    :return invariant_df (DataFrame): DataFrame with invariant molecules (variance <= min_feature_variance)
    """
    variances = df.var(axis=1)
    variant_df = df.loc[variances > min_feature_variance]
    invariant_df = df.loc[variances <= min_feature_variance]
    return variant_df, invariant_df


# noinspection DuplicatedCode
def get_pycycle(df_in, period, models=None):
    """
    Models expression data using 4 equations.

    :param df_in: A dataframe organised with samples defined by columns and molecules defined by rows.
                    The first column and row should contain strings identifying samples or molecules.
                    Samples should be organised in ascending time order (all reps per timepoint should be together)
    :param period: An integer indicating the primary period length of interest (in same AU as timepoints)
    :param models: A list of strings identifying the models to be used. ->['harmonic', 'square', 'cycloid', 'transient']
    :return: df_out: A dataframe containing the best-fitting model, with parameters that produced the best fit,
                        alongside statistics indicating the robustness of the model's fit compared to input data.
    """
    df_in = df_in.set_index(df_in.columns[0])
    df, df_invariant = variance_based_filtering(df_in)  # Filtering removes invariant molecules from analysis
    pvals = []
    osc_type = []
    mod_type = []
    parameters = []
    fitted_model = []
    model_rmse = []
    if models is None:
        models = ['harmonic', 'square', 'cycloid', 'transient']
    timepoints = np.array([float(re.findall(r'\d+', col)[0]) for col in df.columns])
    timepoints = (timepoints / period * (2 * math.pi))
    reps = len(timepoints) / len(np.unique(timepoints))
    if np.any(df.to_numpy() < 0):
        lbound = 1  # 1 = neg values present
    else:
        lbound = 0  # 0 = neg values absent
    if isinstance(df.iloc[0, 0], str):
        df = df.set_index(df.columns.tolist()[0])
    for i in tqdm(range(df.shape[0])):
        waveform, params, covariance, fitted_values, rmse = fit_best_waveform(df.iloc[i, :], period, models, timepoints,
                                                                              reps, lbound)
        if waveform == 'unsolved' or waveform == 'non-rhythmic':
            tau, p_value = np.nan, np.nan
            modulation = np.nan
        else:
            tau, p_value = kendalltau(fitted_values, df.iloc[i, :].values)
            modulation = categorize_rhythm(params[1])
        oscillation = waveform
        if math.isnan(p_value):
            p_value = 1
        pvals.append(p_value)
        osc_type.append(oscillation)
        mod_type.append(modulation)
        parameters.append(params)
        fitted_model.append(fitted_values)
        model_rmse.append(rmse)
    corr_pvals = multipletests(pvals, alpha=0.001, method='fdr_tsbh')[1]
    cap_bh_pvals = np.where(pvals > corr_pvals, pvals, corr_pvals)
    df_out = pd.DataFrame({"Feature": df.index.tolist(), "p-val": pvals, "BH-padj": cap_bh_pvals, "Waveform": osc_type,
                           "Modulation": mod_type, "parameters": parameters, "Fitted_values": fitted_model,
                          "RMSE": model_rmse})
    invariant_features = df_invariant.index.tolist()
    invariant_rows = pd.DataFrame({
        "Feature": invariant_features,
        "p-val": [np.nan] * len(invariant_features),
        "BH-padj": [np.nan] * len(invariant_features),
        "Type": ['invariant'] * len(invariant_features),
        "parameters": [np.nan] * len(invariant_features)
    })
    # Concatenate variant and invariant rows
    df_out = pd.concat([df_out, invariant_rows], ignore_index=False)
    return df_out.sort_values(by='p-val').sort_values(by='BH-padj')

# Todo: Differential expression
# Todo: Visualisation functions for dataset-wide phases ect
# Todo: Phase-set enrichment tools for transcripts / proteins / glycans?
# Todo: Include compositional transforms + uncertainty scale model
# Todo: introduce modifier to y term (baseline) to capture general trends in expression?
# Todo: Is an integral/dot product a better index of similarity than residuals?
# Todo: improve stats, KT needs an upgrade


# noinspection DuplicatedCode
def fit_repro(df_in):
    df_in = df_in.set_index(df_in.columns[0])
    df, df_invariant = variance_based_filtering(df_in)
    pvals = []
    osc_type = []
    mod_type = []
    parameters = []
    fitted_model = []
    rmse = []
    period = 4
    models = ['harmonic', 'transient']
    timepoints = np.array([float(re.findall(r'\d+', col)[0]) for col in df.columns])
    timepoints = (timepoints / period * (2 * math.pi))
    reps = len(timepoints) / len(np.unique(timepoints))
    if np.any(df.to_numpy() < 0):
        lbound = 1  # 1 = neg values present
    else:
        lbound = 0  # 0 = neg values absent
    if isinstance(df.iloc[0, 0], str):
        df = df.set_index(df.columns.tolist()[0])
    for i in tqdm(range(df.shape[0])):
        waveform, params, covariance, fitted_values, rmse = fit_best_waveform(df.iloc[i, :], period, models, timepoints,
                                                                              reps, lbound)
        if waveform == 'unsolved' or waveform == 'non-rhythmic':
            tau, p_value = np.nan, np.nan
            modulation = np.nan
        else:
            tau, p_value = kendalltau(fitted_values, df.iloc[i, :].values)
            modulation = categorize_rhythm(params[1])
        oscillation = waveform
        if math.isnan(p_value):
            p_value = 1
        pvals.append(p_value)
        osc_type.append(oscillation)
        mod_type.append(modulation)
        parameters.append(params)
        fitted_model.append(fitted_values)
    corr_pvals = multipletests(pvals, alpha=0.001, method='fdr_tsbh')[1]
    cap_bh_pvals = np.where(pvals > corr_pvals, pvals, corr_pvals)
    df_out = pd.DataFrame({"Feature": df.index.tolist(), "p-val": pvals, "BH-padj": cap_bh_pvals, "Waveform": osc_type,
                           "Modulation": mod_type, "parameters": parameters, "Fitted_values": fitted_model,
                          "RMSE": rmse})
    invariant_features = df_invariant.index.tolist()
    invariant_rows = pd.DataFrame({
        "Feature": invariant_features,
        "p-val": [np.nan] * len(invariant_features),
        "BH-padj": [np.nan] * len(invariant_features),
        "Waveform": ['invariant'] * len(invariant_features),
        "parameters": [np.nan] * len(invariant_features)
    })
    # Concatenate variant and invariant rows
    df_out = pd.concat([df_out, invariant_rows], ignore_index=False)
    return df_out.sort_values(by='p-val').sort_values(by='BH-padj')


def pcbplot(data, res, molecule, period=24, colour=None, shading=None, title=True, path=None):

    data = data.set_index(data.columns[0])
    res = res.set_index(res.columns[0])
    measurements = data.loc[molecule, :]
    waveform = res.loc[molecule][2]
    params = res.loc[molecule][4]

    timepoints = np.array([float(re.findall(r'\d+', col)[0]) for col in data.columns])
    timepoints = (timepoints / period * (2 * math.pi))  # Plots against time constant rather than time
    t = np.linspace(0, np.max(timepoints), 500)

    if waveform == 'sinusoidal':
        eq = p_harmonic_oscillator(t, params[0], params[1], params[2], params[3], params[4])
        wavecolour = '#1EA896'
    elif waveform == 'cycloid':
        eq = p_cycloid_wave(t, params[0], params[1], params[2], params[3], params[4])
        wavecolour = "#81A68C"
    elif waveform == 'square':
        eq = p_square_wave(t, params[0], params[1], params[2], params[3], params[4])
        wavecolour = "#545775"
    elif waveform == 'transient':
        eq = p_transient_impulse(t, params[0], params[1], params[2], params[3], params[4], params[5])
        wavecolour = "#AFAFEB"
    else:
        return 'No suitable model found'

    if colour is not None:
        wavecolour = colour

    timepoints_plot = (timepoints * period) / (2 * math.pi)
    t_plot = (t * period) / (2 * math.pi)

    plt.scatter(x=timepoints_plot, y=measurements.transpose(), zorder=3, color="#000000")
    plt.plot(t_plot, eq, color=wavecolour, linewidth=3, zorder=2)
    plt.xticks(np.arange(0, np.max(timepoints_plot), 6))
    if title is True:
        plt.title(molecule)
    xmin, xmax = plt.xlim()
    if shading:
        shading_starts = []
        shading_ends = []
        for i in range(0, int(len(shading)/2)):
            shading_starts.append(shading[i*2])
            shading_ends.append(shading[i*2+1])
            for start, end in [(shading_starts[i], shading_ends[i])]:
                plt.axvspan(start, end, color='gray', alpha=0.4, zorder=1)

    plt.xlim(xmin, xmax)
    if path:
        plt.savefig(path)
    plt.show()


def get_regs(data, res, target, regs, period=24):

    rhythmic_regs = pd.DataFrame()
    complex_wave = None
    sub_res = None

    row = res[res.iloc[:, 0] == target]
    if row.iloc[0, 3] == 'square':
        complex_wave = 0
    if row.iloc[0, 3] == 'cycloid':
        complex_wave = 1

    for i in range(0, len(regs)):

        reg_iter = regs[i]
        element = res[res.iloc[:, 0].str.lower() == reg_iter.lower()]
        try:
            if element.iloc[0, 2] < 0.05:
                element = element.assign(Likely_Component='Carrier')
                rhythmic_regs = pd.concat([rhythmic_regs, element])
        except IndexError:
            pass

    if complex_wave == 0:
        print("Finding high-frequency components, "
              "access regulatory elements with 'results[0]' and this analysis with 'results[1]'")
        sub_res = get_pycycle(data, period/3)

    if complex_wave == 1:
        print("Finding high-frequency components, "
              "access regulatory elements with 'results[0]' and this analysis with 'results[1]'")
        sub_res = get_pycycle(data, period/2)

    if complex_wave is not None:

        for i in range(0, len(regs)):

            reg_iter = regs[i]
            element = sub_res[sub_res.iloc[:, 0].str.lower() == reg_iter.lower()]
            try:
                if element.iloc[0, 2] < 0.05:
                    element = element.assign(Likely_Component='Modulator')
                    rhythmic_regs = pd.concat([rhythmic_regs, element])
            except IndexError:
                pass
        rhythmic_regs.sort_values(by='BH-padj')
        rhythmic_regs = rhythmic_regs[~rhythmic_regs.duplicated(subset=rhythmic_regs.columns[0], keep='first')]
        return [rhythmic_regs, sub_res]

    else:
        rhythmic_regs.sort_values(by='BH-padj')
        rhythmic_regs = rhythmic_regs[~rhythmic_regs.duplicated(subset=rhythmic_regs.columns[0], keep='first')]
        return rhythmic_regs.sort_values(by='BH-padj')
