from typing import Optional, Tuple, List
from itertools import repeat
import os
import sys
import glob
import json
import subprocess

import numpy as np
import pandas as pd
import awkward as ak

from aliad.interface.awkward import Momentum4DArrayBuilder
from quickstats import stdout
from quickstats.utils.string_utils import split_str
from quickstats.maths.numerics import cartesian_product


def _get_unit_scale(unit: str = "GeV"):
    if unit == "GeV":
        return 1.0
    elif unit == "TeV":
        return 0.001
    raise ValueError(f'invalid unit: {unit} (choose between "GeV" and "TeV")')


def get_flattened_arrays(arrays: "awkward.Array", pad_size: int = 200):
    jet_keys = arrays.fields
    jet_arrays = {}
    part_arrays = {}
    array_builder = Momentum4DArrayBuilder("PtEtaPhiM")
    for jet_key in jet_keys:
        if jet_key == "jj":
            continue
        jet_p4 = array_builder.get_array_from_dict(
            {
                "pt": arrays[jet_key]["jet_pt"],
                "eta": arrays[jet_key]["jet_eta"],
                "phi": arrays[jet_key]["jet_phi"],
                "m": arrays[jet_key]["jet_m"],
            }
        )
        jet_arrays_i = {
            f"px{jet_key}": jet_p4.px,
            f"py{jet_key}": jet_p4.py,
            f"pz{jet_key}": jet_p4.pz,
        }
        part_arrays_i = {}
        for field in arrays[jet_key].fields:
            array = arrays[jet_key][field]
            # particle features
            if array.ndim == 2:
                padded_arrays = ak.to_numpy(
                    ak.fill_none(ak.pad_none(array, pad_size, clip=True), 0)
                ).T
                for i, padded_array in enumerate(padded_arrays):
                    key = field.replace("part_", "") + f"{jet_key}p{i + 1}"
                    part_arrays_i[key] = padded_array
            # jet features
            else:
                key = field.replace("jet_", "") + jet_key
                jet_arrays_i[key] = array
        for key in jet_arrays_i:
            jet_arrays_i[key] = ak.to_numpy(jet_arrays_i[key])
        # for key in part_arrays_i:
        #    part_arrays_i[key] = ak.to_numpy(part_arrays_i[key])
        jet_arrays.update(jet_arrays_i)
        part_arrays.update(part_arrays_i)
    if "jj" in jet_keys:
        for field in arrays["jj"].fields:
            key = field.replace("dijet_", "") + "jj"
            jet_arrays[key] = ak.to_numpy(arrays["jj"][field])
    return {**jet_arrays, **part_arrays}


def RnD_txt_to_arrays(
    filename: str,
    sortby: Optional[str] = "mass",
    unit: str = "GeV",
    feature_level: str = "low_level",
    extra_vars: bool = True,
    flatten: bool = False,
    pad_size: int = 200,
):
    assert sortby in [None, "mass", "pt"]
    assert unit in ["GeV", "TeV"]
    assert feature_level in ["low_level", "high_level"]
    stdout.info(f'Reading file "{filename}"')
    event_numbers = []
    jet_indices = []
    jet_features = []
    jet_N = []
    part_features = []
    with open(filename, "r") as f:
        data = f.readlines()
    low_level = feature_level == "low_level"
    scale = _get_unit_scale(unit)
    ntaus = 5 if extra_vars else 3
    for line in data:
        tokens = line.split()
        event_number = int(tokens[0])
        jet_index = int(tokens[1])
        jet_features_i = [float(token) for token in tokens[3:13]]
        event_numbers.append(event_number)
        jet_indices.append(jet_index)
        part_tokens = line.split("P")[1:]
        jet_N = len(part_tokens)
        jet_features_i.append(jet_N)
        jet_features.append(jet_features_i)
        if low_level:
            # the last element of the tuple is the constituent mass (= 0)
            part_features_i = [
                tuple(split_str(tokens, cast=float) + [0.0]) for tokens in part_tokens
            ]
            part_features_i = np.array(
                part_features_i,
                dtype=[
                    ("pT", "float64"),
                    ("eta", "float64"),
                    ("phi", "float64"),
                    ("mass", "float64"),
                ],
            )
            part_features.append(part_features_i)
    event_numbers = np.array(event_numbers)
    jet_indices = np.array(jet_indices)
    jet_size = np.unique(jet_indices).shape[0]
    jet_features = np.array(jet_features)
    record = {
        "event_number": event_numbers,
        "jet_index": jet_indices,
        "jet_features": jet_features,
    }
    if low_level:
        array_builder = Momentum4DArrayBuilder("PtEtaPhiM")
        part_features = array_builder.get_array_from_list(part_features)
        record["part_features"] = part_features
    if sortby is not None:
        feature_idx_map = {"pt": 0, "mass": 3}
        feature_idx = feature_idx_map[sortby]
        feature_size = jet_features.shape[0]
        sort_idx = np.argsort(
            -jet_features[:, feature_idx].reshape(feature_size // jet_size, jet_size),
            axis=-1,
        )
        sort_idx = (
            sort_idx
            + np.arange(feature_size // jet_size).reshape(feature_size // jet_size, 1)
            * jet_size
        )
        sort_idx = sort_idx.flatten()
        for key in ["jet_features", "part_features"]:
            if key not in record:
                continue
            record[key] = record[key][sort_idx]
    record = ak.Record(record)
    arrays = {}
    for jet_index in range(jet_size):
        jet_key = f"j{jet_index + 1}"
        jet_mask = record["jet_index"] == jet_index
        jet_features = record["jet_features"][jet_mask]
        arrays[jet_key] = {}
        # jet features
        arrays[jet_key]["jet_pt"] = jet_features[:, 0] * scale
        arrays[jet_key]["jet_eta"] = jet_features[:, 1]
        arrays[jet_key]["jet_phi"] = jet_features[:, 2]
        arrays[jet_key]["jet_m"] = jet_features[:, 3] * scale
        # arrays[jet_key]["N"]        = ak.num(part_features, axis=-1)
        arrays[jet_key]["N"] = jet_features[:, -1]
        if extra_vars:
            arrays[jet_key]["jet_btag"] = jet_features[:, 4]
        for i in range(ntaus):
            arrays[jet_key][f"tau{i + 1}"] = jet_features[:, 5 + i]
        if extra_vars:
            tau_mask = np.array(((jet_features[:, 5] > 0) & (jet_features[:, 6] > 0)))
            arrays[jet_key]["tau12"] = np.where(
                tau_mask,
                np.divide(jet_features[:, 6], jet_features[:, 5], where=tau_mask),
                0,
            )
            arrays[jet_key]["tau23"] = np.where(
                tau_mask,
                np.divide(jet_features[:, 7], jet_features[:, 6], where=tau_mask),
                0,
            )
        jet_p4 = Momentum4DArrayBuilder.get_array_from_dict(
            {
                "pt": jet_features[:, 0],
                "eta": jet_features[:, 1],
                "phi": jet_features[:, 2],
                "m": jet_features[:, 3],
            }
        )
        # particle features
        if low_level:
            part_features = record["part_features"][jet_mask]
            arrays[jet_key]["part_pt"] = part_features.pt * scale
            arrays[jet_key]["part_eta"] = part_features.eta
            arrays[jet_key]["part_phi"] = part_features.phi
            if extra_vars:
                arrays[jet_key]["part_e"] = part_features.e * scale
                arrays[jet_key]["part_relpt"] = part_features.pt / jet_p4.pt
                arrays[jet_key]["part_deltaeta"] = part_features.deltaeta(jet_p4)
                arrays[jet_key]["part_deltaphi"] = part_features.deltaphi(jet_p4)
                arrays[jet_key]["part_deltaR"] = part_features.deltaR(jet_p4)
    ak_arrays = ak.Array(arrays)
    # dijet features
    if extra_vars and (jet_size == 2):
        # compute mjj
        j1_p4 = ak.zip(
            {
                "pt": ak_arrays["j1"]["jet_pt"],
                "eta": ak_arrays["j1"]["jet_eta"],
                "phi": ak_arrays["j1"]["jet_phi"],
                "m": ak_arrays["j1"]["jet_m"],
            },
            with_name="Momentum4D",
        )
        j2_p4 = ak.zip(
            {
                "pt": ak_arrays["j2"]["jet_pt"],
                "eta": ak_arrays["j2"]["jet_eta"],
                "phi": ak_arrays["j2"]["jet_phi"],
                "m": ak_arrays["j2"]["jet_m"],
            },
            with_name="Momentum4D",
        )
        jj_p4 = j1_p4.add(j2_p4)
        ak_arrays["jj"] = ak.Array({"dijet_m": jj_p4.m, "dijet_pt": jj_p4.pt})
    if flatten:
        return get_flattened_arrays(ak_arrays, pad_size=pad_size)
    return ak_arrays
