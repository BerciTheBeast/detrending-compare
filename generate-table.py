from CosinorPy import file_parser, cosinor, cosinor1, cosinor_nonlin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import os
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import re

timestamp = "1711791535"
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



print("columns", df.columns)
methods = df["method"].unique()
print('methods', methods)

tables_path = "data/results/processed/{timestamp}/tables".format(timestamp=timestamp)
if not os.path.exists(tables_path):
    os.makedirs(tables_path)

# agg_dict = {
#     "amp_err": ["mean", "std"],
#     "mesor_err": ["mean", "std"],
#     "phase_err": ["mean", "std"]
# }

amp_dict = {
    "amp_err": ["mean", "std"]
}

mesor_dict = {
    "mesor_err": ["mean", "std"]
}

phase_dict = {
    "phase_err": ["mean", "std"]
}

splits = {
    "A": [
        ("amp", "mean"),
        ("amp", "std"),
        ("arima", "mean"),
        ("arima", "std"),
        ("baseline", "mean"),
        ("baseline", "std")
    ],
    "B": [
        ("lin", "mean"),
        ("lin", "std"),
        ("min_max_abs", "mean"),
        ("min_max_abs", "std")
    ],
    "C": [
        ("none", "mean"),
        ("none", "std"),
        ("qub", "mean"),
        ("qub", "std"),
        ("z_score", "mean"),
        ("z_score", "std")
    ]
}

group_components = ["n_components", "baseline", "phase_ref", "mesor_ref", "amplitude_ref", "max_time", "time_step", "noise", "replicates", "lin_comp", "amplification", "method"]
out_components = ["l", "p", "a", "t", "s"]


# for method in methods:
# amp_dict["amp_err"]["method"] = ["mean", "std"]
# mesor_dict["mesor_err"]["method"] = ["mean", "std"]
# phase_dict["phase_err"]["method"] = ["mean", "std"]

# 41k entries...
# new_df = df.groupby(by=["n_components", "baseline", "phase_ref", "mesor_ref", "amplitude_ref", "max_time", "time_step", "noise", "replicates", "lin_comp", "amplification", "method"], as_index=False).aggregate(agg_dict)
# print(new_df)
# exit(0)

count = 0

# df_list = []
df_amp = []
df_mesor = []
df_phase = []

# mathods_df = pd.DataFrame(columns=pd.MultiIndex.from_product(methods, ["mean", "std"], names=["methods", ""]))

def find_min_method(row, methods):
    min_method = None

    for method in methods:
        if min_method is None or np.float64(row[method]["mean"]) < np.float64(row[min_method]["mean"]):
            min_method = method

    return min_method

def handle_num_format(num):
    if np.abs(num) >= 10_000 or (np.abs(num) <= 0.00001 and num != 0):
        return '{:.0e}'.format(num)
    return '{:.2f}'.format(num)

def flatten(arr):
    out_arr = []
    for subarr in arr:
        for elem in subarr:
            out_arr.append(elem)

    return out_arr


for n_components in [1]:
    for replicates in [4]:
        for baseline in [0]:
            for noise in [0.5]:
                for lin_comp in [0,1]:
                    for phase_ref in [0, np.pi]:
                        for amplif in [-0.05, 0, 0.05]:
                            for max_time in [24, 48, 96]:
                                for time_step in [1, 2, 4, 8]:
                                    # for error in ["amp_err", "phase_err", "mesor_err"]:
                                    # NONE OF THESE SHOULD HAVE THE SAME NAMES AS METHODS
                                    # OUTPUT LATEX NEEDS 0 EXTRA "&" signs for proper alignment of subcolumns
                                    # TODO: Maybe give these props shorter names?
                                    phase_str = str(phase_ref)

                                    if (phase_ref == np.pi):
                                        phase_str = 'pi'


                                    temp_df_amp = pd.DataFrame({
                                       "l#": [lin_comp], # lin_comp
                                       "p#": [phase_str], # phase_ref
                                       "a#": [amplif], # amplification
                                    #    "comp#": [n_components], # n_components
                                    #    "rep#": [replicates], # replicates
                                    #    "base#": [baseline], # baseline_val
                                    #    "noise#": [noise], # noise
                                       "t#": [max_time], # max_time
                                       "s#": [time_step] # time_step
                                    })
                                    temp_df_mesor = pd.DataFrame({
                                       "l#": [lin_comp], # lin_comp
                                       "p#": [phase_str], # phase_ref
                                       "a#": [amplif], # amplification
                                    #    "comp#": [n_components], # n_components
                                    #    "rep#": [replicates], # replicates
                                    #    "base#": [baseline], # baseline_val
                                    #    "noise#": [noise], # noise
                                       "t#": [max_time], # max_time
                                       "s#": [time_step] # time_step
                                    })
                                    temp_df_phase = pd.DataFrame({
                                       "l#": [lin_comp], # lin_comp
                                       "p#": [phase_str], # phase_ref
                                       "a#": [amplif], # amplification
                                    #    "comp#": [n_components], # n_components
                                    #    "rep#": [replicates], # replicates
                                    #    "base#": [baseline], # baseline_val
                                    #    "noise#": [noise], # noise
                                       "t#": [max_time], # max_time
                                       "s#": [time_step] # time_step
                                    })

                                    for method in methods:
                                        temp_df_amp["{method}#mean".format(method=method)] = 0
                                        temp_df_amp["{method}#std".format(method=method)] = 0
                                        temp_df_mesor["{method}#mean".format(method=method)] = 0
                                        temp_df_mesor["{method}#std".format(method=method)] = 0
                                        temp_df_phase["{method}#mean".format(method=method)] = 0
                                        temp_df_phase["{method}#std".format(method=method)] = 0

                                    temp_df_amp = temp_df_amp.set_axis(temp_df_amp.columns.str.split('#',expand=True),axis=1)
                                    temp_df_mesor = temp_df_mesor.set_axis(temp_df_mesor.columns.str.split('#',expand=True),axis=1)
                                    temp_df_phase = temp_df_phase.set_axis(temp_df_phase.columns.str.split('#',expand=True),axis=1)

                                    
                                    # https://stackoverflow.com/questions/40225683/how-to-simply-add-a-column-level-to-a-pandas-dataframe
                                    # print(temp_df_amp.index)
                                    # temp_df_amp = temp_df_amp.set_axis(pd.MultiIndex.from_product([temp_df_amp.columns, ['']]), axis=1)
                                    # print(temp_df_amp)
                                    # print(temp_df_amp.index)
                                    # temp_df_mesor = temp_df_mesor.set_axis(pd.MultiIndex.from_product([temp_df_mesor.columns, ['']]), axis=1)
                                    # temp_df_phase = temp_df_phase.set_axis(pd.MultiIndex.from_product([temp_df_phase.columns, ['']]), axis=1)

                                    for method in methods:
                                        count = count + 1
                                        df_filtered = df[
                                            (df["lin_comp"] == lin_comp) &
                                            (df["phase_ref"] == phase_ref) &
                                            (df["amplification"] == amplif) &
                                            (df["n_components"] == n_components) &
                                            (df["replicates"] == replicates) &
                                            (df["baseline"] == baseline) &
                                            (df["noise"] == noise) &
                                            (df["max_time"] == max_time) &
                                            (df["time_step"] == time_step) &
                                            (df["method"] == method)
                                        ]

                                        df_filtered = df_filtered.copy()
                                        # print(df_filtered)
                                        # new_df = pd.DataFrame(columns=["n_components", "baseline", "phase_ref", "mesor_ref", "amplitude_ref", "max_time", "time_step", "noise", "replicates", "lin_comp", "amplification", "method", "amp_err", "mesor_err", "phase_err"])
                                        # new_df = df_filtered.groupby(by=out_components, as_index=False).aggregate(agg_dict)
                                        new_df_amp = df_filtered.groupby(by=group_components, as_index=False).aggregate(amp_dict)
                                        new_df_mesor = df_filtered.groupby(by=group_components, as_index=False).aggregate(mesor_dict)
                                        new_df_phase = df_filtered.groupby(by=group_components, as_index=False).aggregate(phase_dict)
                                        # print(new_df_amp)
                                        # print(new_df_amp["amp_err"])



                                        # print("count", count)
                                        # if (temp_df_amp.columns.nlevels > 1):
                                        #     temp_df_amp = pd.merge(temp_df_amp, new_df_amp["amp_err"], left_index=True, right_index=True)
                                        #     temp_df_mesor = pd.merge(temp_df_mesor, new_df_mesor["mesor_err"], left_index=True, right_index=True)
                                        #     temp_df_phase = pd.merge(temp_df_phase, new_df_phase["phase_err"], left_index=True, right_index=True)
                                        #     print('idx 1', temp_df_amp.index)
                                        # else:
                                        #     temp_df_amp = pd.merge(temp_df_amp.set_axis(pd.MultiIndex.from_product([temp_df_amp.columns, ['']]), axis=1), new_df_amp["amp_err"], left_index=True, right_index=True)
                                        #     temp_df_mesor = pd.merge(temp_df_mesor.set_axis(pd.MultiIndex.from_product([temp_df_mesor.columns, ['']]), axis=1), new_df_mesor["mesor_err"], left_index=True, right_index=True)
                                        #     temp_df_phase = pd.merge(temp_df_phase.set_axis(pd.MultiIndex.from_product([temp_df_phase.columns, ['']]), axis=1), new_df_phase["phase_err"], left_index=True, right_index=True)
                                        #     print('idx 2', temp_df_amp.index)

                                        temp_df_amp[method] = new_df_amp["amp_err"]
                                        temp_df_mesor[method] = new_df_mesor["mesor_err"]
                                        temp_df_phase[method] = new_df_phase["phase_err"]

                                        # print(temp_df_amp)
                                        # print(temp_df_amp)
                                        # temp_df_amp[method] = new_df_amp["amp_err"]
                                        # temp_df_mesor[method] = new_df_mesor["mesor_err"]
                                        # temp_df_phase[method] = new_df_phase["phase_err"]
                                        # df_list.append(new_df)
                                        # print(new_df)
                                        # exit(0)

                                    
                                    # count = count + 1
                                    # df_filtered = df[
                                    #     (df["lin_comp"] == lin_comp) &
                                    #     (df["phase_ref"] == phase_ref) &
                                    #     (df["amplification"] == amplif) &
                                    #     (df["n_components"] == n_components) &
                                    #     (df["replicates"] == replicates) &
                                    #     (df["baseline"] == baseline) &
                                    #     (df["noise"] == noise) &
                                    #     (df["max_time"] == max_time) &
                                    #     (df["time_step"] == time_step)
                                    # ]

                                    # df_filtered = df_filtered.copy()
                                    df_amp.append(temp_df_amp)
                                    df_mesor.append(temp_df_mesor)
                                    df_phase.append(temp_df_phase)
                                    # TODO: make this work so each row has column for all 8 methods, not a row for each method

print("count", count)

outfile_base = "table"
try:

    df_amp_end = pd.concat(df_amp, ignore_index=True)
    df_mesor_end = pd.concat(df_mesor, ignore_index=True)
    df_phase_end = pd.concat(df_phase, ignore_index=True)

    for method in methods:
        df_amp_end[method, "mean"] = df_amp_end[method, "mean"].map(handle_num_format)
        df_amp_end[method, "std"] = df_amp_end[method, "std"].map(handle_num_format)
        df_mesor_end[method, "mean"] = df_mesor_end[method, "mean"].map(handle_num_format)
        df_mesor_end[method, "std"] = df_mesor_end[method, "std"].map(handle_num_format)
        df_phase_end[method, "mean"] = df_phase_end[method, "mean"].map(handle_num_format)
        df_phase_end[method, "std"] = df_phase_end[method, "std"].map(handle_num_format)

        
        # df_amp_end[method, "mean"] = df_amp_end[method, "mean"].map('{:.2E}'.format)
        # df_amp_end[method, "std"] = df_amp_end[method, "std"].map('{:.2E}'.format)
        # df_mesor_end[method, "mean"] = df_mesor_end[method, "mean"].map('{:.2E}'.format)
        # df_mesor_end[method, "std"] = df_mesor_end[method, "std"].map('{:.2E}'.format)
        # df_phase_end[method, "mean"] = df_phase_end[method, "mean"].map('{:.2E}'.format)
        # df_phase_end[method, "std"] = df_phase_end[method, "std"].map('{:.2E}'.format)

    for i, row in df_amp_end.iterrows():
        min_method = find_min_method(row, methods)
        if min_method is not None:
            df_amp_end.loc[i, (min_method, "mean")] = "\\b{" + row[min_method, "mean"] + "}"

    for i, row in df_mesor_end.iterrows():
        min_method = find_min_method(row, methods)
        if min_method is not None:
            df_mesor_end.loc[i, (min_method, "mean")] = "\\b{" + row[min_method, "mean"] + "}"

    for i, row in df_phase_end.iterrows():
        min_method = find_min_method(row, methods)
        if min_method is not None:
            df_phase_end.loc[i, (min_method, "mean")] = "\\b{" + row[min_method, "mean"] + "}"

    # df_amp_end = df_amp_end.apply(lambda x: mark_row_max(x, methods), axis=1)
    # df_mesor_end = df_mesor_end.apply(lambda x: mark_row_max(x, methods), axis=1)
    # df_phase_end = df_phase_end.apply(lambda x: mark_row_max(x, methods), axis=1)

    # pd.set_option('display.float_format', '{:.2E}'.format)
    # df_mesor_end.set_option('display.float_format', '{:.2E}'.format)
    # df_phase_end.set_option('display.float_format', '{:.2E}'.format)

    # amp
    df_amp_end.to_csv(os.path.join(tables_path, "{outfile_base}_amp.csv".format(outfile_base=outfile_base)), index=False)

    # mesor
    df_mesor_end.to_csv(os.path.join(tables_path, "{outfile_base}_mesor.csv".format(outfile_base=outfile_base)), index=False)

    # phase
    df_phase_end.to_csv(os.path.join(tables_path, "{outfile_base}_phase.csv".format(outfile_base=outfile_base)), index=False)

    out_filenames = []

    # split into multiple graphs - current table too big. maybe 3-2-3 split?
    for key, val in splits.items():
        out_cols = []
        for key_other, val_other in splits.items():
            if (key != key_other):
                for entry in val_other:
                    out_cols.append(entry)
        new_split_amp = df_amp_end.drop(columns=out_cols)
        new_split_mesor = df_mesor_end.drop(columns=out_cols)
        new_split_phase = df_phase_end.drop(columns=out_cols)

        # latex parts
        amp_filename = os.path.join(tables_path, "{outfile_base}_amp_{key}.tex".format(outfile_base=outfile_base, key=key))
        new_split_amp.to_latex(amp_filename, index=False)
        out_filenames.append(amp_filename)

        mesor_filename = os.path.join(tables_path, "{outfile_base}_mesor_{key}.tex".format(outfile_base=outfile_base, key=key))
        new_split_mesor.to_latex(mesor_filename, index=False)
        out_filenames.append(mesor_filename)

        phase_filename = os.path.join(tables_path, "{outfile_base}_phase_{key}.tex".format(outfile_base=outfile_base, key=key))
        new_split_phase.to_latex(phase_filename, index=False)
        out_filenames.append(phase_filename)
        
    title_formatter = {
        "amp": "Rezultati amplitudne napake, del ##part##/1",
        "mesor": "Rezultati napake MESOR, del ##part##/1",
        "phase": "Rezultati fazne napake, del ##part##/1"
    }

    label_formatter = {
        "amp": "tab:amp:##part##_1",
        "mesor": "tab:mesor:##part##_1",
        "phase": "tab:phase:##part##_1"
    }

    # latex
    for filename in out_filenames:
        name_split = filename.split("_")
        part = name_split.pop().split('.')[-2]
        err_type = name_split.pop()
        print('parts', err_type, part)

        header = """\\begin{table}[!h]
\\begin{center}
\\caption{##caption##}
\\label{##label##}
\\begin{tabular}{""".replace("##caption##", title_formatter[err_type], 1).replace("##label##", label_formatter[err_type]).replace("##part##", part, 2)

        with open(filename, 'r+') as file:
            data = file.read()
            file.seek(0)
            data = data.replace(
            """\\begin{tabular}{""",
            header,
            1
            )
            data = data.replace("""\end{tabular}""", """\end{tabular}
\end{center}
\end{table}""", 1)
            data = data.replace(' mean', ' & & & & mean', 1)
            data = data.replace('\\textbackslash b\\{', '\\textbf{')
            data = data.replace('\\}', '}')
            data = data.replace('pi', '$\\pi$')
            file.write(data)
            file.truncate()

    # TODO: split by hand, breaker (every 24-ish lines, look for line count +25 AKA VSAKE 2 razdelka [24-96], se pravi 2x od 24-96 max time, potem pa break):

    # \bottomrule
    # \end{tabular}
    # \end{center}
    # \end{table}

    # \begin{table}[!h]
    # \begin{center}
    # \caption{Rezultati amplitudne napake, del A}
    # \label{tab:amp:A}
    # \begin{tabular}{rlrrrllllll}
    # \toprule
    # l &  p &     a &  t & s & \multicolumn{2}{l}{amp} & \multicolumn{2}{l}{arima} & \multicolumn{2}{l}{baseline} \\
    # &     & & & & mean &   std &     mean &   std &      mean &  std \\
    # \midrule



    # latex

    # TMP?
    # df_amp_end.to_latex(os.path.join(tables_path, "{outfile_base}_amp.tex".format(outfile_base=outfile_base)), index=False)
    # df_mesor_end.to_latex(os.path.join(tables_path, "{outfile_base}_mesor.tex".format(outfile_base=outfile_base)), index=False)
    # df_phase_end.to_latex(os.path.join(tables_path, "{outfile_base}_phase.tex".format(outfile_base=outfile_base)), index=False)


    # return df
except KeyboardInterrupt:
    print('Interrupted')
    sys.exit(1)
except Exception as exc:
    print('Error saving: ' + "{outfile_base}.csv", exc)

print("done")