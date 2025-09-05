import numpy as np
import awkward as ak
import vector
import gc

vector.register_awkward()
import pandas as pd
import energyflow as ef
import fastjet as fj
import h5py

from data_processing.utils import get_dijetmass, get_mjj_mask


def process_extra_qcdbkg(filename, outputpath, pad_size=-1, region="SR"):

    df = pd.read_hdf(filename)

    # drop all columns with high level features
    columns = df.columns
    high_level_features = [
        column for column in columns if "j1" in column or "j2" in column
    ]
    df = df.drop(columns=high_level_features)
    assert len(df.columns) == 2100

    # the column of https://zenodo.org/records/8370758/files/events_anomalydetection_qcd_extra_inneronly_4vecs_and_features.h5
    # is falsly set to be px0, py0, ... , px699, py699, pz699
    # but if we check the scale, it should be pt0, eta0, ... , pt699, eta699, phi699
    # so we rename the columns to avoid confusion
    df.rename(columns={f"px{i}": f"pt{i}" for i in range(700)}, inplace=True)
    df.rename(columns={f"py{i}": f"eta{i}" for i in range(700)}, inplace=True)
    df.rename(columns={f"pz{i}": f"phi{i}" for i in range(700)}, inplace=True)

    print(df.describe())

    # run jet clustering to calculate all necessary features
    out_dict = jet_clustering(df, outputpath, pad_size)

    # clean the memory
    del df
    gc.collect()

    # apply SR mask, shuffle the data, and save to outputpath
    final_processing(out_dict, outputpath, region)


def process_qcdbkg(filename, outputpath, pad_size=-1, region="SR", saved_result="bkg"):

    processed_qcd_lowlevel = pd.read_hdf(filename)

    if saved_result == "bkg":
        # mask for bkg events with last columns to be 0
        mask = processed_qcd_lowlevel.iloc[:, -1] == 0
        processed_qcd_lowlevel = processed_qcd_lowlevel[mask]
        # drop the last column
        processed_qcd_lowlevel = processed_qcd_lowlevel.iloc[:, :-1]
    elif saved_result == "signal":
        # mask for signal events with last columns to be 1
        mask = processed_qcd_lowlevel.iloc[:, -1] == 1
        processed_qcd_lowlevel = processed_qcd_lowlevel[mask]
        # drop the last column
        processed_qcd_lowlevel = processed_qcd_lowlevel.iloc[:, :-1]
    else:
        raise ValueError("saved_result should be either 'bkg' or 'signal'")

    print(processed_qcd_lowlevel.describe())

    out_dict = jet_clustering(processed_qcd_lowlevel, outputpath, pad_size)

    # clean the memory
    del processed_qcd_lowlevel
    gc.collect()

    final_processing(out_dict, outputpath, region)


# based on https://github.com/uhh-pd-ml/FastJet-LHCO/blob/main/fastjet_awkward.ipynb
def jet_clustering(df, outputpath, pad_size=-1):

    # change the shape of the data to (n_events, n_particles, n_features)
    data_full = np.array(df).reshape(len(df), -1, 3)
    print("input shape: ", data_full.shape)

    # save lengths of signal and background data for later use
    len_data_full = len(data_full)

    print(f"events to be processed: {len_data_full}")

    # to awkard array
    zrs = np.zeros((data_full.shape[0], data_full.shape[1], 1))
    data_with_mass = np.concatenate((data_full, zrs), axis=2)
    awkward_data = ak.from_numpy(data_with_mass)

    # tell awkward that the data is in eta, phi, pt, mass format
    vector.register_awkward()
    unmasked_data = ak.zip(
        {
            "pt": awkward_data[:, :, 0],
            "eta": awkward_data[:, :, 1],
            "phi": awkward_data[:, :, 2],
            "mass": awkward_data[:, :, 3],
        },
        with_name="Momentum4D",
    )

    # remove the padded data points
    data = ak.drop_none(ak.mask(unmasked_data, unmasked_data.pt != 0))

    # here we use anti-Kt algorithm with R=1.0
    jetdef = fj.JetDefinition(fj.antikt_algorithm, 1.0)
    cluster = fj.ClusterSequence(data, jetdef)

    # get jets and constituents
    jets_out = cluster.inclusive_jets()
    consts_out = cluster.constituents()

    # sort jets by pt and only keep the leading 2 jets and their constituents
    jets_sorted, idxs = sort_by_pt(jets_out, return_indices=True)
    consts_sorted_jets = consts_out[idxs]
    consts_sorted = sort_by_pt(consts_sorted_jets)

    n_jets = 2
    jets_awk = jets_sorted[:, :n_jets]
    consts_awk = consts_sorted[:, :n_jets]

    # get max. number of constituents in an event
    if pad_size == -1:
        max_consts = int(ak.max(ak.num(consts_awk, axis=-1)))
        print("pad with max counts", max_consts)
    else:
        max_consts = pad_size
        print("pad with fixed size", max_consts)

    # pad the data with zeros to make them all the same length
    zero_padding = ak.zip(
        {"pt": 0.0, "eta": 0.0, "phi": 0.0, "mass": 0.0}, with_name="Momentum4D"
    )
    padded_consts1 = ak.fill_none(
        ak.pad_none(consts_awk, max_consts, clip=True, axis=-1), zero_padding, axis=-1
    )
    # pad the data on the jet axis to make sure to at least have n_jets jets
    zero_padding_jet = ak.zip(
        {
            "pt": [0.0] * max_consts,
            "eta": [0.0] * max_consts,
            "phi": [0.0] * max_consts,
            "mass": [0.0] * max_consts,
        },
        with_name="Momentum4D",
    )
    padded_consts = ak.fill_none(
        ak.pad_none(padded_consts1, n_jets, clip=True, axis=1), zero_padding_jet, axis=1
    )

    # go back to numpy arrays
    energy_const = ak.to_numpy(padded_consts.energy)
    pt, eta, phi, mass = ak.unzip(padded_consts)
    pt_np = ak.to_numpy(pt)
    eta_np = ak.to_numpy(eta)
    phi_np = ak.to_numpy(phi)
    e_np = ak.to_numpy(energy_const)
    consts = np.stack((pt_np, eta_np, phi_np), axis=-1)
    print("constituents shape: ", consts.shape)  # should be N x 2 x max_consts x 3

    # calculate mask for jet constituents
    mask = np.expand_dims((consts[..., 0] > 0).astype(int), axis=-1)

    # get numpy arrays for jet data
    jets_pt_np = ak.to_numpy(jets_awk.pt)
    jets_eta_np = ak.to_numpy(jets_awk.eta)
    jets_phi_np = ak.to_numpy(jets_awk.phi)
    jets_m_np = ak.to_numpy(jets_awk.m)
    jets_e_np = ak.to_numpy(jets_awk.energy)
    jets = np.stack((jets_pt_np, jets_eta_np, jets_phi_np, jets_m_np), axis=-1)
    print("jets shape: ", jets.shape)

    # Negative jet masses can occur due to jets represented as a single massless particle. We will set the negative masses to zero.
    jets[:, :, -1][jets[:, :, -1] < 0] = 0.0

    # Now need to further convert the data to the format that is used in the training of Omnilearn
    # for jet it is (N, 2, 5), 5 features are pt, eta, phi, mass, multiplicity (num of non zero constituents)
    jet_data = np.concatenate([jets, np.sum(mask, axis=-2)], axis=-1)

    # for constituents it is (N, 2, max_consts, 4), 4 features are dEta, dPhi, log(pt), log(E)
    mask = mask.reshape(-1, 2, max_consts)
    jets_eta_np = np.expand_dims(jets_eta_np, axis=-1)
    jets_phi_np = np.expand_dims(jets_phi_np, axis=-1)
    jets_pt_np = np.expand_dims(jets_pt_np, axis=-1)
    jets_e_np = np.expand_dims(jets_e_np, axis=-1)

    consts_dEta = eta_np - jets_eta_np
    consts_dEta = consts_dEta * mask
    consts_dPhi = phi_np - jets_phi_np
    consts_dPhi = consts_dPhi * mask
    consts_logPt = np.where(pt_np > 0, np.log(pt_np), 0)
    consts_logPt = consts_logPt * mask
    consts_logE = np.where(e_np > 0, np.log(e_np), 0)
    consts_logE = consts_logE * mask

    # wrap phi between -pi and pi
    consts_dPhi = np.where(
        consts_dPhi > np.pi,
        consts_dPhi - 2 * np.pi,
        consts_dPhi,
    )
    consts_dPhi = np.where(
        consts_dPhi < -np.pi,
        consts_dPhi + 2 * np.pi,
        consts_dPhi,
    )

    consts = np.concatenate(
        [
            np.expand_dims(consts_dEta, axis=-1),
            np.expand_dims(consts_dPhi, axis=-1),
            np.expand_dims(consts_logPt, axis=-1),
            np.expand_dims(consts_logE, axis=-1),
        ],
        axis=-1,
    )

    dijet_mass = get_dijetmass(jet_data)

    # preprocessing of nan and inf values
    jet_data[np.isnan(jet_data)] = 0.0
    jet_data[np.isinf(jet_data)] = 0.0
    consts[np.isnan(consts)] = 0.0
    consts[np.isinf(consts)] = 0.0

    print("after final processing:")
    print("jets shape: ", jets.shape)
    print("constituents shape: ", consts.shape)

    return {
        "constituents": consts,
        "global": jet_data,
        "condition": dijet_mass,
    }


def final_processing(out_dict, output_path, region="SR"):

    constituents = out_dict["constituents"]
    jet_data = out_dict["global"]
    dijet_mass = out_dict["condition"]

    mask_region = get_mjj_mask(dijet_mass)

    if region == "SR":
        constituents = constituents[mask_region]
        jet_data = jet_data[mask_region]
        dijet_mass = dijet_mass[mask_region]

    else:
        constituents = constituents[~mask_region]
        jet_data = jet_data[~mask_region]
        dijet_mass = dijet_mass[~mask_region]

    print(f"after applying {region} mask:")
    print("jets shape: ", jet_data.shape)
    print("constituents shape: ", constituents.shape)

    np.random.seed(42)
    indices = np.arange(jet_data.shape[0])
    np.random.shuffle(indices)
    jet_data = jet_data[indices]
    constituents = constituents[indices]
    dijet_mass = dijet_mass[indices]

    # save to outputpath
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("constituents", data=constituents)
        hf.create_dataset("global", data=jet_data)
        hf.create_dataset("condition", data=dijet_mass)


# define a helper function to sort ak.Array by pt
def sort_by_pt(data: ak.Array, ascending: bool = False, return_indices: bool = False):
    """Sort ak.Array by pt

    Args:
        data (ak.Array): array that should be sorted by pt. It should have a pt attribute.
        ascending (bool, optional): If True, the first value in each sorted group will be smallest; if False, the order is from largest to smallest. Defaults to False.
        return_indices (bool, optional): If True, the indices of the sorted array are returned. Defaults to False.

    Returns:
        ak.Array: sorted array
        ak.Array (optional): indices of the sorted array
    """
    if isinstance(data, ak.Array):
        try:
            temppt = data.pt
        except AttributeError:
            raise AttributeError(
                "Needs either correct coordinates or embedded vector backend"
            ) from None
    tmpsort = ak.argsort(temppt, axis=-1, ascending=ascending)
    if return_indices:
        return data[tmpsort], tmpsort
    else:
        return data[tmpsort]
