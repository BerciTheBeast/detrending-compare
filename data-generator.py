from CosinorPy import file_parser, cosinor, cosinor1, cosinor_nonlin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import tqdm
import os
from time import time
import multiprocessing
from joblib import Parallel, delayed
import sys
import traceback

# df = file_parser.generate_test_data(phase = 0, n_components = 1, name="test1", lin_comp = 0.1, noise=0.5, replicates = 3, independent=False)
# df2 = file_parser.generate_test_data(phase = np.pi, n_components = 1, name="test3", lin_comp = 0, noise=0.5, replicates = 3, independent=False)
# df = df.append(df2, ignore_index=True)

# df2 = file_parser.generate_test_data(phase = 0, n_components = 3, name="test4", amplification = -0.04, noise=0.5, replicates = 3, time_step=1,  max_time = 72, independent=False)
# df = df.append(df2, ignore_index=True)
# df2 = file_parser.generate_test_data(phase = np.pi, n_components = 3, name="test2", amplification = 0.04, noise=0.5, replicates = 3, time_step=1, max_time = 72, independent=False)
# df = df.append(df2, ignore_index=True)

# n_components: int = 1,
# period: int = 24,
# amplitudes: Any | None = None,
# baseline: int = 0,
# lin_comp: int = 0,
# amplification: int = 0,
# phase: int = 0,
# min_time: int = 0,
# max_time: int = 48,
# time_step: int = 2,
# replicates: int = 1,
# independent: bool = True,

# noise: int = 0,
# noise_simple: int = 1,
# characterize_data: bool = False

timestamp = int(time())

if not os.path.exists("data/raw/{timestamp}".format(timestamp=timestamp)):
    os.makedirs("data/raw/{timestamp}".format(timestamp=timestamp))

def generate_df_for_n(n_components):
    # df_list = []
    for period in [24]:
        for baseline in [0, 5]:
            for lin_comp in [0, 1]:
                df_list = []
                for amplification in [-0.05, 0, 0.05]:
                    for quad in [
                        0,
                        # 0.05
                    ]:
                        for phase in [
                            0, np.pi
                            ]:
                            for min_time in [0]:
                                for max_time in [24, 2*24, 4*24]:
                                    for time_step in [1, 2, 4, 8]:
                                        for noise in [
                                            # 0,
                                            0.5,
                                            3
                                        ]:
                                            for noise_simple in [0]:
                                                for replicates in [1, 2, 4]: # add to test name (or nah?)
                                                    for experiment in range(5): # add to test name
                                                        try:
                                                            df2 = file_parser.generate_test_data(
                                                                n_components = n_components,
                                                                period = period,
                                                                # amplitudes = amplitudes,
                                                                amplitudes = np.multiply(5, np.array([1,1/2,1/3,1/4])),
                                                                baseline = baseline,
                                                                lin_comp = lin_comp,
                                                                amplification = amplification,
                                                                phase = phase,
                                                                min_time = min_time,
                                                                max_time = max_time,
                                                                time_step = time_step,
                                                                noise = noise,
                                                                noise_simple = noise_simple,
                                                                name="test_{n_components}_{baseline}_{lin_comp}_{amplification}_{quad}_{phase}_{min_time}_{max_time}_{time_step}_{noise}_{noise_simple}_{replicates}_{experiment}".format(
                                                                    n_components = n_components,
                                                                    period = period,
                                                                    # amplitudes = amplitudes,
                                                                    baseline = baseline,
                                                                    lin_comp = lin_comp,
                                                                    amplification = amplification,
                                                                    phase = phase,
                                                                    min_time = min_time,
                                                                    max_time = max_time,
                                                                    time_step = time_step,
                                                                    noise = noise,
                                                                    noise_simple = noise_simple,
                                                                    replicates=replicates,
                                                                    experiment=experiment,
                                                                    quad=quad
                                                                ),
                                                                replicates=replicates,
                                                                independent=independent,
                                                                quad = quad,
                                                                add_stats=True)
                                                            # appending to dataframes is very expensive, concat at the end is waaay cheaper
                                                            df_list.append(df2)
                                                        except Exception as exc_gen:
                                                            print('Error generating: ' + "test_{n_components}_{baseline}_{lin_comp}_{amplification}_{quad}_{phase}_{min_time}_{max_time}_{time_step}_{noise}_{noise_simple}_{replicates}_{experiment}".format(n_components = n_components,
                                                                    period = period,
                                                                    # amplitudes = amplitudes,
                                                                    baseline = baseline,
                                                                    lin_comp = lin_comp,
                                                                    amplification = amplification,
                                                                    phase = phase,
                                                                    min_time = min_time,
                                                                    max_time = max_time,
                                                                    time_step = time_step,
                                                                    noise = noise,
                                                                    noise_simple = noise_simple,
                                                                    replicates=replicates,
                                                                    experiment=experiment,
                                                                    quad=quad))
                                                            print(exc_gen)
                                                            print(traceback.format_exc())
                                                            exit(1)
                        print("\tfinished n:{n_components}, p: {period}, b: {baseline}, lc: {lin_comp}, amp: {amplification}".format(n_components=n_components, period=period, baseline=baseline, lin_comp=lin_comp, amplification=amplification, phase=phase))
                try:
                    df = pd.concat(df_list, ignore_index=True)
                    df = df.sort_values(by=['test', 'x'])
                    if save and saveByComponent:
                        df.to_csv(os.path.join("data/raw/{timestamp}".format(timestamp=timestamp), "generated_data_{n_components}_{period}_{baseline}_{lin_comp}_{timestamp}.csv".format(n_components=n_components, timestamp=timestamp, period=period, baseline=baseline, lin_comp=lin_comp, amplification=amplification)), index=False)
                    # return df
                except KeyboardInterrupt:
                    print('Interrupted')
                    sys.exit(1)
                except Exception as exc:
                    print('Error saving: ' + "generated_data_{n_components}_{period}_{baseline}_{lin_comp}_{timestamp}.csv".format(n_components=n_components, timestamp=timestamp, period=period, baseline=baseline, lin_comp=lin_comp, amplification=amplification), exc)
            print("finished n:{n_components}, p: {period}, b: {baseline}".format(n_components=n_components, period=period, baseline=baseline))
    # try:
    #     df = pd.concat(df_list, ignore_index=True)
    #     if save and saveByComponent:
    #         df.to_csv(os.path.join("data/raw/{timestamp}".format(timestamp=timestamp), "generated_data_{n_components}_{timestamp}.csv".format(n_components=n_components, timestamp=timestamp, period=period, baseline=baseline, lin_comp=lin_comp, amplification=amplification)), index=False)
    #     return df
    # except KeyboardInterrupt:
    #     print('Interrupted')
    #     sys.exit(1)
    # except:
    #     print('Error saving: ' + "generated_data_{n_components}_{timestamp}.csv".format(n_components=n_components, timestamp=timestamp, period=period, baseline=baseline, lin_comp=lin_comp, amplification=amplification))
            
save=True
saveByComponent=True
# replicates = 5
independent = True # Nočemo posebi replikatov
# for outer in tqdm.tqdm([10, 20, 30, 40, 50], desc=" outer", position=0):
#     for inner in tqdm.tqdm(range(outer), desc=" inner loop", position=1, leave=False):
#         time.sleep(0.05)
# components = np.arange(1, 11) # anything above 3 goes out of bounds in "y += amplitudes[j] * np.cos((x/periods[j])*np.pi*2 + phases[j])"
components = np.arange(1, 4)

print('starting for timestamp', timestamp)
results = Parallel(n_jobs=4)(delayed(generate_df_for_n)(i) for i in components)
# print(results)
if save and not saveByComponent:
    res_df = pd.concat(results, ignore_index=True)
    res_df = res_df.sort_values(by=['test', 'x'])
    res_df.to_csv(os.path.join("data/raw/{timestamp}".format(timestamp=timestamp), "generated_data_{timestamp}.csv".format(timestamp=timestamp)), index=False)

# with multiprocessing.Pool(len(components)) as pool:
# 	# call the function for each item in parallel
# 	pool.map(generate_df_for_n, components)

# for n_components in tqdm.tqdm(, desc='components', position=0):
    # df = pd.DataFrame(columns=['test','x','y'], dtype=float)
    
# df
    

# 1711791535