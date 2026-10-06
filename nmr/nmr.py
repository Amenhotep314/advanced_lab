"""nmr.py: Utility functions for the NMR experiment."""

import numpy as np
import scipy as sp
import spinmob as s
import csv
from collections import defaultdict


def load_data(data_pattern: str) -> list:

    datas = []

    for i in range(3):
        data = s.data.load(f"{data_pattern}/000{i} {data_pattern}.txt")
        baseline = data[1][0]
        data[1] = data[1] - baseline
        datas.append(data)

    return datas


def load_data_t1(data_name: str) -> dict:

    data_dict = defaultdict(list)
    data = np.genfromtxt(data_name, delimiter=',', names=True, dtype=None, encoding='utf-8')
    field_names = data.dtype.names

    for row in data:
        key = row[field_names[0]]
        data_dict[key].append([row[name] for name in field_names[1:]])

    data_dict = {key: np.array(value).T for key, value in data_dict.items()}
    return data_dict


def detect_resolution(data):

    deltas = np.diff(data[0][1])
    deltas = deltas[deltas != 0]
    resolution = np.min(np.abs(deltas))

    return resolution


def find_peaks(datas, n_pulses=20, min_height=0.1, max_points=20):

    peak_points = []
    n_peaks = []

    for data in datas:
        peaks = sp.signal.find_peaks(
            data[1],
            height=min_height,
            distance=len(data[1]) / (n_pulses + 1),
            # threshold=0.6,
        )
        times = data[0][peaks[0][1:]]
        times -= times[0]
        amplitudes = data[1][peaks[0][1:]]

        ans = np.vstack([times[:max_points], amplitudes[:max_points]])
        peak_points.append(ans)
        n_peaks.append(len(times))

    for i in range(len(peak_points)):
        peak_points[i] = peak_points[i][:, : min(n_peaks)]

    for peak in peak_points:
        print(peak.shape)

    return peak_points


def get_plot_vals(peak_points, resolution):

    all_point_vals = np.vstack([points[1] for points in peak_points])
    means = np.mean(all_point_vals, axis=0)
    times = peak_points[0][0]

    sems = np.std(all_point_vals, axis=0, ddof=1) / np.sqrt(len(peak_points))
    resolution_error = resolution / np.sqrt(12)
    uncertainties = np.sqrt(sems**2 + resolution_error**2)

    ans = {"times": times, "means": means, "uncertainties": uncertainties}

    return ans


def fit_t2_model(plot_values):

    model = lambda t, M_0, T_2, b: M_0 * np.exp(-t / T_2) + b

    fit = sp.optimize.curve_fit(
        model,
        plot_values["times"],
        plot_values["means"],
        p0=(14, 0.1, -0.5),
        bounds=((0, 0, -2), (20, 0.2, 2)),
        sigma=plot_values["uncertainties"],
        absolute_sigma=True,
    )

    return fit
