from CosinorPy import file_parser, cosinor, cosinor1, cosinor_nonlin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import os
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt

timestamp = "1711791535"
report_combos = True
if not os.path.exists("data/results/graphs/{timestamp}".format(timestamp=timestamp)):
    os.makedirs("data/results/graphs/{timestamp}".format(timestamp=timestamp))

processed_file="generated_data_{timestamp}.csv".format(timestamp=timestamp)
df = pd.read_csv(os.path.join("data/results/processed/{timestamp}".format(timestamp=timestamp), processed_file))

# remove entries without noise
mask = df["noise"] == 0
df = df[~mask]

# for now only lin and amp trends
mask = df["quad"] == 0
df = df[mask]

# todo compute error
if ("test" in df.columns):
    df = df.drop("test", axis=1)
df.replace("raw", "none", inplace=True)

# df["amp_rel_err"] = np.abs(df["amplitude"] - df["amplitude_ref"]) / df["amplitude_ref"]
# df["amp_err"] = np.abs(df["amplitude"] - df["amplitude_ref"])
# df["mesor_err"] = np.abs(df["mesor"] - df["mesor_ref"])
# df["phase_err"] = np.abs(df["acrophase"] - df["phase_ref"])
# print(df)
# print(df.describe())


print("columns", df.columns)
methods = df["method"].unique()
print('methods', methods)


# plot_path = "data/results/graphs/{timestamp}".format(timestamp=timestamp)
# if not os.path.exists(plot_path):
#     os.makedirs(plot_path)
# # y os napaka v amplitudi, x os razlicne barve bojo razlicne metode detrendinga, razlicni stolpci bodo pa razlicne vrednosti parametrov pri najboljšem bo violina zelo dol, najslabši pa najbolj gor
# for lin_comp in [0,1]:
#     for amplif in [-0.05, 0, 0.05]:
#         df_filtered = df[(df["lin_comp"] == lin_comp) & (df["amplification"] == amplif)]
#         for p in ["n_components","baseline","lin_comp","amplification","max_time","time_step","noise","replicates"]:
#             for error in ["amp_err", "phase_err", "mesor_err"]:
#                 plt.clf()
#                 plot_name = "violin_{lin_comp}_{amplif}_{p}_{error}.pdf".format(lin_comp=lin_comp,amplif=amplif,p=p,error=error)
#                 violin_plot = sns.violinplot(data=df_filtered, x=p, y=error, hue="method", log_scale=True)
#                 violin_plot.axhline(1, color="lightgrey", ls='--')
#                 fig = violin_plot.get_figure()
#                 fig.savefig(os.path.join(plot_path, plot_name))
#                 # exit(0)

# graph combos for report touples
if report_combos:
    graph_combos = {
            'amp_err' :
                {
                    # none
                    '0_0_all': [
                        'violin_0_0_1_4_0_0.5_0_amp_err_24_1',
                        'violin_0_0_1_4_0_0.5_0_amp_err_96_1',
                        'violin_0_0_1_4_0_0.5_3.141592653589793_amp_err_24_1'
                    ],
                    # lin
                    '1_0_no': [
                        'violin_1_0_1_4_0_0.5_0_amp_err_24_1',
                        'violin_1_0_1_4_0_0.5_0_amp_err_24_4',
                        'violin_1_0_1_4_0_0.5_0_amp_err_96_4'
                    ],
                    '1_0_phas': [
                        'violin_1_0_1_4_0_0.5_3.141592653589793_amp_err_24_1',
                        'violin_1_0_1_4_0_0.5_3.141592653589793_amp_err_24_8',
                        'violin_1_0_1_4_0_0.5_3.141592653589793_amp_err_96_8'
                    ],
                    # neg amp
                    '0_-0.05_all': [
                        'violin_0_-0.05_1_4_0_0.5_0_amp_err_24_1',
                        'violin_0_-0.05_1_4_0_0.5_0_amp_err_96_1',
                        'violin_0_-0.05_1_4_0_0.5_3.141592653589793_amp_err_24_1'
                    ],
                    # pos amp
                    '0_0.05_all': [
                        'violin_0_0.05_1_4_0_0.5_0_amp_err_24_1',
                        'violin_0_0.05_1_4_0_0.5_0_amp_err_96_1',
                        'violin_0_0.05_1_4_0_0.5_3.141592653589793_amp_err_24_1'
                    ],
                    # both neg amp
                    '1_-0.05_all': [
                        'violin_1_-0.05_1_4_0_0.5_0_amp_err_24_1',
                        'violin_1_-0.05_1_4_0_0.5_0_amp_err_96_8',
                        'violin_1_-0.05_1_4_0_0.5_3.141592653589793_amp_err_24_1'
                    ],
                    # both pos amp
                    '1_0.05_all': [
                        'violin_1_0.05_1_4_0_0.5_0_amp_err_24_1',
                        'violin_1_0.05_1_4_0_0.5_0_amp_err_96_1',
                        'violin_1_0.05_1_4_0_0.5_3.141592653589793_amp_err_24_1'
                    ]
                },
            'mesor_err': {
                    # none
                    '0_0_all': [
                        'violin_0_0_1_4_0_0.5_0_mesor_err_24_1',
                        'violin_0_0_1_4_0_0.5_0_mesor_err_24_8',
                        'violin_0_0_1_4_0_0.5_3.141592653589793_mesor_err_24_1'
                    ],
                    # lin
                    '1_0_all': [
                        'violin_1_0_1_4_0_0.5_0_mesor_err_24_1',
                        'violin_1_0_1_4_0_0.5_0_mesor_err_96_1',
                        'violin_1_0_1_4_0_0.5_3.141592653589793_mesor_err_24_1'
                    ],
                    # neg amp
                    '0_-0.05_all': [
                        'violin_0_-0.05_1_4_0_0.5_0_mesor_err_24_1',
                        'violin_0_-0.05_1_4_0_0.5_0_mesor_err_24_8',
                        'violin_0_-0.05_1_4_0_0.5_3.141592653589793_mesor_err_24_1'
                    ],
                    # pos amp
                    '0_0.05_all': [
                        'violin_0_0.05_1_4_0_0.5_0_mesor_err_24_1',
                        'violin_0_0.05_1_4_0_0.5_0_mesor_err_48_8',
                        'violin_0_0.05_1_4_0_0.5_3.141592653589793_mesor_err_24_1'
                    ],
                    # both neg amp
                    '1_-0.05_all': [
                        'violin_1_-0.05_1_4_0_0.5_0_mesor_err_24_1',
                        'violin_1_-0.05_1_4_0_0.5_0_mesor_err_48_1',
                        'violin_1_-0.05_1_4_0_0.5_3.141592653589793_mesor_err_24_1'
                    ],
                    # both pos amp
                    '1_0.05_all': [
                        'violin_1_0.05_1_4_0_0.5_0_mesor_err_24_1',
                        'violin_1_0.05_1_4_0_0.5_3.141592653589793_mesor_err_24_1',
                        'violin_1_0.05_1_4_0_0.5_3.141592653589793_mesor_err_24_4'
                    ]
                },
            'phase_err': {
                    # none
                    '0_0_all': [
                        'violin_0_0_1_4_0_0.5_0_phase_err_24_1',
                        'violin_0_0_1_4_0_0.5_0_phase_err_96_1',
                        'violin_0_0_1_4_0_0.5_3.141592653589793_phase_err_96_1'
                    ],
                    # lin
                    '1_0_no': [
                        'violin_1_0_1_4_0_0.5_0_phase_err_24_1',
                        'violin_1_0_1_4_0_0.5_0_phase_err_24_4',
                        'violin_1_0_1_4_0_0.5_0_phase_err_48_4'
                    ],
                    '1_0_phas': [
                        'violin_1_0_1_4_0_0.5_3.141592653589793_phase_err_24_1',
                        'violin_1_0_1_4_0_0.5_3.141592653589793_phase_err_24_4',
                        'violin_1_0_1_4_0_0.5_3.141592653589793_phase_err_24_8'

                    ],
                    # neg amp
                    '0_-0.05_all': [
                        'violin_0_-0.05_1_4_0_0.5_0_phase_err_24_1',
                        'violin_0_-0.05_1_4_0_0.5_3.141592653589793_phase_err_24_1',
                        'violin_0_-0.05_1_4_0_0.5_3.141592653589793_phase_err_24_4'

                    ],
                    # pos amp
                    '0_0.05_all': [
                        'violin_0_0.05_1_4_0_0.5_0_phase_err_24_1',
                        'violin_0_0.05_1_4_0_0.5_0_phase_err_24_8',
                        'violin_0_0.05_1_4_0_0.5_3.141592653589793_phase_err_24_1'
                    ],
                    # both neg amp
                    '1_-0.05_all': [
                        'violin_1_-0.05_1_4_0_0.5_0_phase_err_24_1',
                        'violin_1_-0.05_1_4_0_0.5_0_phase_err_96_4',
                        'violin_1_-0.05_1_4_0_0.5_3.141592653589793_phase_err_24_1'
                    ],
                    # both pos amp
                    '1_0.05_no': [
                        'violin_1_0.05_1_4_0_0.5_0_phase_err_24_1',
                        'violin_1_0.05_1_4_0_0.5_0_phase_err_24_2',
                        'violin_1_0.05_1_4_0_0.5_0_phase_err_48_1'
                    ],
                    '1_0.05_phas': [
                        'violin_1_0.05_1_4_0_0.5_3.141592653589793_phase_err_24_1',
                        'violin_1_0.05_1_4_0_0.5_3.141592653589793_phase_err_24_2',
                        'violin_1_0.05_1_4_0_0.5_3.141592653589793_phase_err_96_1'
                    ]
            },

        }

    res_cache = {
        'amp_err': {

        },
        'phase_err': {
            
        },
        'mesor_err': {
            
        }
    }

plot_path = "data/results/graphs/{timestamp}/specific".format(timestamp=timestamp)
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

for n_components in [1]:
    for replicates in [4]:
        for baseline in [0]:
            for noise in [0.5]:
                for lin_comp in [
                    0,
                    1
                    ]:
                    for phase_ref in [0, np.pi]:
                        for amplif in [
                            -0.05,
                            0,
                            0.05
                            ]:
                            for max_time in [24, 48, 96]:
                                for time_step in [1, 2, 4, 8]:
                                    for error in [
                                        "amp_err",
                                        "phase_err",
                                        "mesor_err"
                                        ]:
                                        if not os.path.exists(os.path.join(plot_path, error)):
                                            os.makedirs(os.path.join(plot_path, error))
                                        if report_combos and not os.path.exists(os.path.join(plot_path, error, 'joined')):
                                            os.makedirs(os.path.join(plot_path, error, 'joined'))
                                        df_filtered = df[
                                            (df["lin_comp"] == lin_comp) &
                                            (df["phase_ref"] == phase_ref) &
                                            (df["amplification"] == amplif) &
                                            (df["n_components"] == n_components) &
                                            (df["replicates"] == replicates) &
                                            (df["baseline"] == baseline) &
                                            (df["noise"] == noise) &
                                            (df["max_time"] == max_time) &
                                            (df["time_step"] == time_step)
                                        ]

                                        df_filtered = df_filtered.copy()
                                        
                                        # error_z = "{error}_z".format(error=error)
                                        # df_filtered[error_z] = sp.stats.zscore(df_filtered[error])

                                        # for p in [
                                        #     # "n_components",
                                        #     # "baseline",
                                        #     "lin_comp",
                                        #     # "amplification",
                                        #     # "max_time",
                                        #     # "time_step",
                                        #     # "noise",
                                        #     # "replicates"
                                        #     ]:
                                        # plt.clf()
                                        # # Create a figure with a specific size # TODO: look into plot sizing
                                        # plt.figure(figsize=(8, 6))

                                        # plot_name = "violin_{n_components}_{replicates}_{baseline}_{noise}_{lin_comp}_{amplif}_{p}_{error}_{max_time}_{time_step}.pdf".format(n_components=n_components, replicates=replicates, baseline=baseline, noise=noise, lin_comp=lin_comp,amplif=amplif,
                                        #                                                                                                                                       p=p,
                                        #                                                                                                                                       error=error,max_time=max_time,time_step=time_step)
                                        plot_name = "violin_{lin_comp}_{amplif}_{n_components}_{replicates}_{baseline}_{noise}_{phase_ref}_{error}_{max_time}_{time_step}".format(n_components=n_components, replicates=replicates, baseline=baseline, noise=noise, lin_comp=lin_comp,
                                                                                                                                                                                      phase_ref=phase_ref,amplif=amplif,
                                                                                                                                                                                      error=error,max_time=max_time,time_step=time_step)
                                        
                                        if report_combos:
                                            # combo graphs
                                            group_id = '{lin_comp}_{amplif}'.format(lin_comp=lin_comp, amplif=amplif)
                                            if (group_id + '_all') in graph_combos[error]:
                                                group_id += '_all'
                                            else:
                                                if phase_ref == 0:
                                                    group_id += '_no'
                                                else:
                                                    group_id += '_phas'

                                            is_subplot = False
                                            if (group_id in graph_combos[error]):
                                                is_subplot = plot_name in graph_combos[error][group_id]
                                                # print('checking group id', group_id, is_subplot, plot_name)
                                                # if (not is_subplot):
                                                #     print(plot_name, 'not in', graph_combos[error][group_id])

                                            if (group_id in graph_combos[error]) and (group_id not in res_cache[error]) and is_subplot:
                                                res_cache[error][group_id] = {
                                                    'count': 0,
                                                    'axes': plt.subplots(nrows=1, ncols=len(graph_combos[error][group_id]), squeeze=True, figsize=(14, 4))
                                                }

                                        plot_name_raw = plot_name
                                        # add extension
                                        plot_name += '.pdf'

                                        # violin_plot = sns.violinplot(data=df_filtered, x=p, y=error_z, hue="method", log_scale=True)
                                        # violin_plot.axhline(1, color="lightgrey", ls='--')

                                        fig = plt.figure()
                                        violin_plot = sns.violinplot(data=df_filtered,
                                                                    #  x=p,
                                                                    #  y=error_z, hue="method", log_scale=False)
                                                                    y=error, hue="method", log_scale=False)
                                        violin_plot.axhline(0, color="lightgrey", ls='--')


                                        fig = violin_plot.get_figure()
                                        fig.tight_layout()
                                        fig.savefig(os.path.join(plot_path, error, plot_name))
                                        plt.close(fig)

                                        # print('graphmaker', is_subplot, report_combos and is_subplot)
                                        if (report_combos and is_subplot):
                                            res_cache[error][group_id]['count'] += 1
                                            plt_idx = graph_combos[error][group_id].index(plot_name_raw)
                                            sub_fig = res_cache[error][group_id]['axes'][0]
                                            sub_axes = res_cache[error][group_id]['axes'][1]
                                            sub_ax = sub_axes[plt_idx]
                                            sub_ax.clear()
                                            # print('group idx', group_id, plt_idx)
                                            phase_text = 0
                                            if (phase_ref == np.pi):
                                                phase_text = 'pi'
                                            sub_ax.set_title('Duration: {max_time}h, Sample: {time_step}h, Phase: {phase_text}'.format(max_time=max_time,time_step=time_step, phase_text=phase_text)) # , fontsize=16)
                                            violin_subplot = sns.violinplot(data=df_filtered,
                                                                    #  x=p,
                                                                    #  y=error_z, hue="method", log_scale=False)
                                                                    y=error, hue="method", log_scale=False, ax=sub_ax)
                                            violin_subplot.axhline(0, color="lightgrey", ls='--')

                                            # set common legend
                                            if (plt_idx == len(graph_combos[error][group_id]) - 1):
                                                handles, labels = sub_ax.get_legend_handles_labels()
                                                sub_fig.legend(handles, labels, loc='center right')

                                            # remove individual ax legends
                                            sub_ax.get_legend().remove()

                                            if (res_cache[error][group_id]['count'] == len(graph_combos[error][group_id])):
                                                sub_fig.tight_layout()
                                                sub_fig.savefig(os.path.join(plot_path, error, 'joined', group_id + '.pdf'))
                                                plt.close(sub_fig)
                                                del res_cache[error][group_id]
                                        # exit(0)

print("done")
