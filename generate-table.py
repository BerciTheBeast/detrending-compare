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

agg_dict = {
    "amp_err": "mean",
    "mesor_err": "mean",
    "phase_err": "mean"
}

# 41k entries...
# new_df = df.groupby(by=["n_components", "baseline", "phase_ref", "mesor_ref", "amplitude_ref", "max_time", "time_step", "noise", "replicates", "lin_comp", "amplification", "method"], as_index=False).aggregate(agg_dict)
# print(new_df)
# exit(0)

count = 0

df_list = []
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
                                        new_df = df_filtered.groupby(by=["n_components", "baseline", "phase_ref", "mesor_ref", "amplitude_ref", "max_time", "time_step", "noise", "replicates", "lin_comp", "amplification", "method"], as_index=False).aggregate(agg_dict)
                                        df_list.append(new_df)
                                        # print(new_df)
                                        # exit(0)

print("count", count)

outfile_base = "table"
try:
    df = pd.concat(df_list, ignore_index=True)
    # print(df)
    # exit(0)
    # df = df.sort_values(by=['test', 'x'])
    df.to_csv(os.path.join(tables_path, "{outfile_base}.csv".format(outfile_base=outfile_base)), index=False)
    df.to_latex(os.path.join(tables_path, "{outfile_base}.tex".format(outfile_base=outfile_base)), index=False)
    # return df
except KeyboardInterrupt:
    print('Interrupted')
    sys.exit(1)
except Exception as exc:
    print('Error saving: ' + "{outfile_base}.csv", exc)

print("done")