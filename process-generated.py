from CosinorPy import file_parser, cosinor, cosinor1, cosinor_nonlin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import os
import warnings
import time
from joblib import Parallel, delayed

def get_elapsed(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

def get_plot_diff(df1, df2, test):
    y1 = np.array(df1[df1['test']==test].y)
    y2 = np.array(df2[df2['test']==test].y)
    return y2 - y1

def draw_graph_compare(df1, df2, test, color='black', save=None):
    diff = get_plot_diff(df1, df2, test)
    x1, y1 = np.array(df1[df1['test']==test].x), np.array(df1[df1['test']==test].y)
    x2, y2 = np.array(df2[df2['test']==test].x), np.array(df2[df2['test']==test].y)

    min_y = np.min([np.min(y1), np.min(y2)])
    max_y = np.max([np.max(y1), np.max(y2)])
    margin = max_y * 0.1
    low_limit = min_y - margin
    high_limit = max_y + margin

    fig, (plt1, plt2, plt_diff) = plt.subplots(1, 3, figsize=(24, 5))
    plt1.set_ylim([low_limit, high_limit])
    plt1.plot(x1, y1, 'o', color=color, markersize=1)

    plt2.set_ylim([low_limit, high_limit])
    plt2.plot(x2, y2, 'o', color=color, markersize=1)

    # plt_diff.set_ylim([0 - margin, max_y - min_y + margin])
    # print(x1, diff)
    plt_diff.plot(x1, diff, 'o', color=color, markersize=1)

    if save is not None:
        plt.tight_layout()
        plt.savefig(save)

    plt.show()


def process_file(filepath):
    df = pd.read_csv(filepath)

    df = df.sort_values(by=['test', 'x'])
    filename = os.path.basename(filepath)
    filename_no_ext = os.path.splitext(filename)[0]

    start = time.time()
    print('starting regular for', filename)

    raw_file = "{filename_no_ext}_none.csv".format(filename_no_ext=filename_no_ext)
    try:
        if not os.path.exists(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), raw_file)):
            df_results = cosinor.fit_group(df, n_components = [1,2,3,4,5], period=24, plot=False)
            df_best_models = cosinor.get_best_models(df, df_results, n_components = [1,2,3,4,5])
            df_best_models.to_csv(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), raw_file), index=False)
    except Exception as error:
        print('error during', raw_file, error)
    reg_time = time.time()
    print('{filename_no_ext} reg took: '.format(filename_no_ext=filename_no_ext), get_elapsed(start, reg_time))

    # linear detrend
    print('starting linear for', filename)
    lin_file = "{filename_no_ext}_lin.csv".format(filename_no_ext=filename_no_ext)
    try:
        if not os.path.exists(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), lin_file)):
            df_lin = cosinor.remove_lin_comp_df(df, n_components = 2)
            df_lin_results = cosinor.fit_group(df_lin, n_components = [1,2,3,4,5], period=24, plot=False)
            df_lin_best_models = cosinor.get_best_models(df_lin, df_lin_results, n_components = [1,2,3,4,5])
            df_lin_best_models.to_csv(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), lin_file), index=False)
    except Exception as error:
        print('error during', lin_file, error)
    # print('lin results', df_lin_best_models)
    lin_time = time.time()
    print('{filename_no_ext} lin took: '.format(filename_no_ext=filename_no_ext), get_elapsed(reg_time, lin_time))

    # qubic
    print('starting qubic for', filename)
    qub_file = "{filename_no_ext}_qub.csv".format(filename_no_ext=filename_no_ext)
    try:
        if not os.path.exists(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), qub_file)):
            df_qub = cosinor.remove_polin_comp_df(df, degree = 3)
            df_qub_results = cosinor.fit_group(df_qub, n_components = [1,2,3,4,5], period=24, plot=False)
            df_qub_best_models = cosinor.get_best_models(df_qub, df_qub_results, n_components = [1,2,3,4,5])
            df_qub_best_models.to_csv(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), qub_file), index=False)
    except Exception as error:
        print('error during', qub_file, error)
    qub_time = time.time()
    print('{filename_no_ext} qub took: '.format(filename_no_ext=filename_no_ext), get_elapsed(lin_time, qub_time))

    # baseline
    print('starting baseline for', filename)
    baseline_file = "{filename_no_ext}_baseline.csv".format(filename_no_ext=filename_no_ext)
    try:
        if not os.path.exists(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), baseline_file)) :
            df_baseline = cosinor.remove_baseline_comp_df(df, width=2)
            df_baseline_results = cosinor.fit_group(df_baseline, n_components = [1,2,3,4,5], period=24, plot=False)
            df_baseline_best_models = cosinor.get_best_models(df_baseline, df_baseline_results, n_components = [1,2,3,4,5])
            df_baseline_best_models.to_csv(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), baseline_file), index=False)
    except Exception as error:
        print('error during', baseline_file, error)
    baseline_time = time.time()
    print('{filename_no_ext} baseline took: '.format(filename_no_ext=filename_no_ext), get_elapsed(qub_time, baseline_time))

    # amp
    print('starting amp for', filename)
    amp_file = "{filename_no_ext}_amp.csv".format(filename_no_ext=filename_no_ext)
    try:
        if not os.path.exists(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), amp_file)):
            df_amp = cosinor.remove_amp_comp_df(df, phase=0, period=24)
            df_amp_results = cosinor.fit_group(df_amp, n_components = [1,2,3,4,5], period=24, plot=False)
            df_amp_best_models = cosinor.get_best_models(df_amp, df_amp_results, n_components = [1,2,3,4,5])
            df_amp_best_models.to_csv(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), amp_file), index=False)
    except Exception as error:
        print('error during', amp_file, error)
    amp_time = time.time()
    print('{filename_no_ext} amp took: '.format(filename_no_ext=filename_no_ext), get_elapsed(baseline_time, amp_time))

    # z-score
    print('starting z-score for', filename)
    z_score_file = "{filename_no_ext}_z_score.csv".format(filename_no_ext=filename_no_ext)
    try:
        if not os.path.exists(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), z_score_file)):
            df_z_score = cosinor.remove_z_score_comp_df(df)
            df_z_score_results = cosinor.fit_group(df_z_score, n_components = [1,2,3,4,5], period=24, plot=False)
            df_z_score_best_models = cosinor.get_best_models(df_z_score, df_z_score_results, n_components = [1,2,3,4,5])
            df_z_score_best_models.to_csv(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), z_score_file), index=False)
    except Exception as error:
        print('error during', z_score_file, error)
    z_score_time = time.time()
    print('{filename_no_ext} z_score took: '.format(filename_no_ext=filename_no_ext), get_elapsed(amp_time, z_score_time))

    # min-max abs
    print('starting min max abs for', filename)
    min_max_abs_file = "{filename_no_ext}_min_max_abs.csv".format(filename_no_ext=filename_no_ext)
    try:
        if not os.path.exists(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), min_max_abs_file)) :
            df_min_max_abs = cosinor.remove_min_max_abs_comp_df(df)
            df_min_max_abs_results = cosinor.fit_group(df_min_max_abs, n_components = [1,2,3,4,5], period=24, plot=False)
            df_min_max_abs_best_models = cosinor.get_best_models(df_min_max_abs, df_min_max_abs_results, n_components = [1,2,3,4,5])
            df_min_max_abs_best_models.to_csv(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), min_max_abs_file), index=False)
    except Exception as error:
        print('error during', min_max_abs_file, error)
    min_max_abs_time = time.time()
    print('{filename_no_ext} min_max_abs took: '.format(filename_no_ext=filename_no_ext), get_elapsed(z_score_time, min_max_abs_time))

    # arima
    print('starting arima for', filename)
    warnings.filterwarnings('ignore')

    arima_file = "{filename_no_ext}_arima.csv".format(filename_no_ext=filename_no_ext)
    try:
        if not os.path.exists(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), arima_file)):
            df_arima = cosinor.remove_arima_comp_df(df)
            df_arima_results = cosinor.fit_group(df_arima, n_components = [1,2,3,4,5], period=24, plot=False)
            df_arima_best_models = cosinor.get_best_models(df_arima, df_arima_results, n_components = [1,2,3,4,5])
            df_arima_best_models.to_csv(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), arima_file), index=False)
    except Exception as error:
        print('error during', arima_file, error)

    warnings.filterwarnings('default')
    arima_time = time.time()
    print('{filename_no_ext} arima took: '.format(filename_no_ext=filename_no_ext), get_elapsed(min_max_abs_time, arima_time))

    # TODO: Extract stats, are almost certainly in df_lin_results


# filenames = [
#     'data/raw/generated_data_1_1697891954.csv',
#     'data/raw/generated_data_2_1697891954.csv',
#     'data/raw/generated_data_3_1697891954.csv'
#     ]

timestamp = "1711791535"
if not os.path.exists("data/results/direct/{timestamp}".format(timestamp=timestamp)):
    os.makedirs("data/results/direct/{timestamp}".format(timestamp=timestamp))

filenames = [
    'data/raw/{timestamp}/generated_data_1_24_0_0_{timestamp}.csv'.format(timestamp=timestamp),
    'data/raw/{timestamp}/generated_data_2_24_0_0_{timestamp}.csv'.format(timestamp=timestamp),
    'data/raw/{timestamp}/generated_data_3_24_0_0_{timestamp}.csv'.format(timestamp=timestamp),
    'data/raw/{timestamp}/generated_data_1_24_5_0_{timestamp}.csv'.format(timestamp=timestamp),
    'data/raw/{timestamp}/generated_data_2_24_5_0_{timestamp}.csv'.format(timestamp=timestamp),
    'data/raw/{timestamp}/generated_data_3_24_5_0_{timestamp}.csv'.format(timestamp=timestamp),
    'data/raw/{timestamp}/generated_data_1_24_0_1_{timestamp}.csv'.format(timestamp=timestamp),
    'data/raw/{timestamp}/generated_data_2_24_0_1_{timestamp}.csv'.format(timestamp=timestamp),
    'data/raw/{timestamp}/generated_data_3_24_0_1_{timestamp}.csv'.format(timestamp=timestamp),
    'data/raw/{timestamp}/generated_data_1_24_5_1_{timestamp}.csv'.format(timestamp=timestamp),
    'data/raw/{timestamp}/generated_data_2_24_5_1_{timestamp}.csv'.format(timestamp=timestamp),
    'data/raw/{timestamp}/generated_data_3_24_5_1_{timestamp}.csv'.format(timestamp=timestamp)
    ]

# 

results = Parallel(n_jobs=4)(delayed(process_file)(i) for i in filenames)

# process_file('data/raw/generated_data_1697824781.csv')

# TO FIX:
# error during generated_data_1_24_5_0_1697967968_amp.csv float division by zero
# generated_data_1_24_5_0_1697967968 amp took:  00:19:44.39
# starting z-score for generated_data_1_24_5_0_1697967968.csv
# generated_data_1_24_5_0_1697967968 z_score took:  00:00:00.00
# starting min max abs for generated_data_1_24_5_0_1697967968.csv
# generated_data_1_24_5_0_1697967968 min_max_abs took:  00:00:00.00
# starting arima for generated_data_1_24_5_0_1697967968.csv
# error during generated_data_1_24_5_0_1697967968_arima.csv float division by zero
# generated_data_1_24_5_0_1697967968 arima took:  02:17:41.13
    