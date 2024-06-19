import os
import pandas as pd
import tqdm

import numpy as np
import math
import scipy.stats as sp


# timestamp="1698760411"
timestamp="1711791535"

if not os.path.exists("data/results/direct/{timestamp}".format(timestamp=timestamp)):
    os.makedirs("data/results/direct/{timestamp}".format(timestamp=timestamp))
if not os.path.exists("data/results/processed/{timestamp}".format(timestamp=timestamp)):
    os.makedirs("data/results/processed/{timestamp}".format(timestamp=timestamp))

result_list = os.listdir('data/results/direct/{timestamp}'.format(timestamp=timestamp))
out_files = list(set([x.split('_')[0] + '_' + x.split('_')[1] + '_' + x.split('_')[6] for x in result_list]))
print('out files', out_files)
for out_file in tqdm.tqdm(out_files, desc="out_file", position=0):
    out_filename = out_file + '.csv'
    if not os.path.exists(os.path.join("data/results/processed/{timestamp}".format(timestamp=timestamp), out_filename)):
        print('generating processed', out_filename)
        df_arr = []
        timestamp = out_file.split('_')[2]
        for direct_file in tqdm.tqdm(result_list, desc="included file", position=1, leave=False):
            if timestamp in direct_file:
                method = "_".join(direct_file.split('_')[7:]).split('.')[0]
                if method == "raw":
                    method = "none"
                df_tmp = pd.read_csv(os.path.join("data/results/direct/{timestamp}".format(timestamp=timestamp), direct_file))
                df_tmp = df_tmp[["test", "amplitude", "acrophase", "mesor"]]
                df_tmp["method"] = method
                df_arr.append(df_tmp)
        df = pd.concat(df_arr, ignore_index=True)
        # {n_components}_{baseline}_{lin_comp}_{amplification}_{quad}_{phase}_{min_time}_{max_time}_{time_step}_{noise}_{noise_simple}_{replicates}_{experiment}
        for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], miniters=100, desc="row", position=1, leave=False):
            test = row["test"]
            _, n_components, baseline, lin_comp, amplification, quad, phase, _, max_time, time_step, noise, _, replicates, experiment, mesor, amplitude = test.split("_")
            df.loc[index, "n_components"] = int(n_components)
            df.loc[index, "baseline"] = float(baseline)
            df.loc[index, "lin_comp"] = float(lin_comp)
            df.loc[index, "amplification"] = float(amplification)
            df.loc[index, "quad"] = float(quad)
            df.loc[index, "phase_ref"] = float(phase)
            diff = row["acrophase"] - float(phase)
            diff = np.mod(diff, (2 * np.pi))
            diff = np.mod((diff + (2 * np.pi)), (2 * np.pi))
            if (diff > np.pi):
                diff = 2 * np.pi - diff
            diff = np.abs(diff)
            # diff = math.fmod(diff, (2 * np.pi))
            # diff = math.fmod((diff + 2 * np.pi), (2 * np.pi))
            df.loc[index, "phase_err"] = diff
            # df.loc[index, "phase_err"] = sp.circmean(float(phase) - row["acrophase"])
            df.loc[index, "mesor_ref"] = float(mesor)
            df.loc[index, "amplitude_ref"] = float(amplitude)
            df.loc[index, "max_time"] = float(max_time)
            df.loc[index, "time_step"] = float(time_step)
            df.loc[index, "noise"] = float(noise)
            df.loc[index, "replicates"] = int(replicates)
            df.loc[index, "experiment"] = int(experiment)
        
        df = df.drop("test", axis=1)
        
        df["amp_rel_err"] = np.abs(df["amplitude"] - df["amplitude_ref"]) / df["amplitude_ref"]
        df["amp_err"] = np.abs(df["amplitude"] - df["amplitude_ref"])
        df["mesor_err"] = np.abs(df["mesor"] - df["mesor_ref"])
        # df["phase_err"] = np.abs(df["acrophase"] - df["phase_ref"])

        df.to_csv(os.path.join(os.path.join("data/results/processed/{timestamp}".format(timestamp=timestamp), out_filename)), index=False)
