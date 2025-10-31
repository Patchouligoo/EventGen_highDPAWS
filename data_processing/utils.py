import numpy as np
import awkward as ak
import vector

vector.register_awkward()
import pandas as pd

import numpy as np
from config.lhco_signal_parameters import mass_upper_bound, mass_lower_bound


def get_dijetmass(jets):
    jet_e = np.sqrt(
        jets[:, 0, 3] ** 2 + jets[:, 0, 0] ** 2 * np.cosh(jets[:, 0, 1]) ** 2
    )
    jet_e += np.sqrt(
        jets[:, 1, 3] ** 2 + jets[:, 1, 0] ** 2 * np.cosh(jets[:, 1, 1]) ** 2
    )
    jet_px = jets[:, 0, 0] * np.cos(jets[:, 0, 2]) + jets[:, 1, 0] * np.cos(
        jets[:, 1, 2]
    )
    jet_py = jets[:, 0, 0] * np.sin(jets[:, 0, 2]) + jets[:, 1, 0] * np.sin(
        jets[:, 1, 2]
    )
    jet_pz = jets[:, 0, 0] * np.sinh(jets[:, 0, 1]) + jets[:, 1, 0] * np.sinh(
        jets[:, 1, 1]
    )
    mjj = np.sqrt(np.abs(jet_px**2 + jet_py**2 + jet_pz**2 - jet_e**2))
    return mjj


def get_mjj_mask(mjj):
    from config.lhco_signal_parameters import SR_mjj_min, SR_mjj_max

    mask_region = (mjj > SR_mjj_min) & (mjj < SR_mjj_max)
    return mask_region


def normalize_mass(mass):
    return (mass - mass_lower_bound) / (mass_upper_bound - mass_lower_bound)


def unnormalize_mass(mass):
    return mass * (mass_upper_bound - mass_lower_bound) + mass_lower_bound


def process_highlevel_features(filepath):
    # ---------------- process high level features ----------------
    df_highlevel = pd.read_hdf(filepath)

    j_p4 = {}
    # compute missing features
    for ji in ["j1", "j2"]:
        j_p4[ji] = ak.zip(
            {
                "px": df_highlevel[f"px{ji}"],
                "py": df_highlevel[f"py{ji}"],
                "pz": df_highlevel[f"pz{ji}"],
                "m": df_highlevel[f"m{ji}"],
            },
            with_name="Momentum4D",
        )
        if f"pt{ji}" not in df_highlevel.columns:
            df_highlevel[f"pt{ji}"] = j_p4[ji].pt
            df_highlevel[f"eta{ji}"] = j_p4[ji].eta
            df_highlevel[f"phi{ji}"] = j_p4[ji].phi
            df_highlevel[f"N{ji}"] = 0
        tau_mask = np.array(
            (df_highlevel[f"tau1{ji}"] > 0) & (df_highlevel[f"tau2{ji}"] > 0)
        )
        df_highlevel[f"tau21{ji}"] = np.where(
            tau_mask,
            np.divide(
                df_highlevel[f"tau2{ji}"], df_highlevel[f"tau1{ji}"], where=tau_mask
            ),
            0,
        )
        df_highlevel[f"tau32{ji}"] = np.where(
            tau_mask,
            np.divide(
                df_highlevel[f"tau3{ji}"], df_highlevel[f"tau2{ji}"], where=tau_mask
            ),
            0,
        )

    # filter by label for mixed dataset
    if "label" in df_highlevel.columns:
        df_highlevel = df_highlevel[df_highlevel["label"] == 0]

    highlevel_features = df_highlevel[
        [
            "mj1",
            "mj2",
            "tau2j1",
            "tau1j1",
            "tau2j2",
            "tau1j2",
            "tau3j1",
            "tau2j1",
            "tau3j2",
            "tau2j2",
        ]
    ].values

    # negative jet masses can occur due to jets represented as a single massless particle. We will set the negative masses to zero.
    highlevel_features[:, :2][highlevel_features[:, :2] < 0] = 0.0

    tau2j1_tau1j1 = highlevel_features[:, 2] / highlevel_features[:, 3]
    tau2j2_tau1j2 = highlevel_features[:, 4] / highlevel_features[:, 5]
    tau3j1_tau2j1 = highlevel_features[:, 6] / highlevel_features[:, 7]
    tau3j2_tau2j2 = highlevel_features[:, 8] / highlevel_features[:, 9]

    highlevel_features = np.concatenate(
        [
            highlevel_features[:, :2],
            tau2j1_tau1j1[:, None],
            tau2j2_tau1j2[:, None],
            tau3j1_tau2j1[:, None],
            tau3j2_tau2j2[:, None],
        ],
        axis=-1,
    )

    # set mass to log scale
    highlevel_features[:, 0] = np.log(highlevel_features[:, 0])
    highlevel_features[:, 1] = np.log(highlevel_features[:, 1])

    print(
        "num of nan or inf values: ",
        np.sum(np.isnan(highlevel_features)),
        np.sum(np.isinf(highlevel_features)),
    )
    # set any nan or inf values to 0
    highlevel_features[np.isnan(highlevel_features)] = 0.0
    highlevel_features[np.isinf(highlevel_features)] = 0.0

    return highlevel_features
