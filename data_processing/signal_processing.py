import pandas as pd
import numpy as np
import h5py
import awkward as ak
from scipy.stats import norm
from data_processing.utils import get_dijetmass, get_mjj_mask


def signal_prep(inputpath, outputpath, pad_size):
    """
    Output of signal from Delphes has format in h5 file:
    pxj1, pyj1, ...., tau23j1, pxj2, pyj2, ...., tau23j2, mjj, ptjj, ptj1p1, ptj1p2, ..., etaj1p1, ..., deltaRj2pN
    when pad size is 150 for both of 2 jets, we have
    16 * 2 + mjj + ptjj + (pt, eta, phi, e, relpt, deltaeta, deltaphi, deltaR) * 2 * 200 = 3234 columns
    so shape is (N, 3234)

    what we need for Omnilearn to train is:
    key = 'constituents', shape = N, 2, 150, 4 (dEta, dPhi, log(pt_constituent), log(e_constituent))
    key = 'global', shape = N, 2, 5 (pt, eta, phi, mass, multiplicity)
    key = 'condition, shape = N, (dijet mass)

    Convert the format
    """

    input_df = pd.read_hdf(inputpath)

    # first create jet_data
    jet_data = input_df[
        ["ptj1", "etaj1", "phij1", "mj1", "ptj2", "etaj2", "phij2", "mj2"]
    ].values.reshape(-1, 2, 4)

    # Negative jet masses can occur due to jets represented as a single massless particle. We will set the negative masses to zero.
    jet_data[:, :, -1][jet_data[:, :, -1] < 0] = 0.0

    jet_pt = np.expand_dims(jet_data[..., 0], axis=-1)
    jet_eta = np.expand_dims(jet_data[..., 1], axis=-1)
    jet_phi = np.expand_dims(jet_data[..., 2], axis=-1)
    jet_mass = np.expand_dims(jet_data[..., 3], axis=-1)
    jet_awk = ak.zip(
        {"pt": jet_pt, "eta": jet_eta, "phi": jet_phi, "mass": jet_mass},
        with_name="Momentum4D",
    )
    jet_energy = ak.to_numpy(jet_awk.e)

    # then create constituents
    j1_columns_to_read = [
        [f"ptj1p{i}", f"etaj1p{i}", f"phij1p{i}"] for i in range(1, pad_size + 1)
    ]
    j1_columns_to_read = list(np.array(j1_columns_to_read).flatten())

    j2_columns_to_read = [
        [f"ptj2p{i}", f"etaj2p{i}", f"phij2p{i}"] for i in range(1, pad_size + 1)
    ]
    j2_columns_to_read = list(np.array(j2_columns_to_read).flatten())

    constituents = input_df[j1_columns_to_read + j2_columns_to_read].values

    constituents = constituents.reshape(-1, 2, pad_size, 3)

    # sort all constituents by pt, pt is at [:, :, :, 0]
    sorted_indices = np.argsort(constituents[:, :, :, 0], axis=2)[:, :, ::-1]
    constituents = np.take_along_axis(
        constituents, sorted_indices[:, :, :, np.newaxis], axis=2
    )

    const_mass = np.zeros_like(constituents[..., 0])
    constituents_awk = ak.zip(
        {
            "pt": constituents[..., 0],
            "eta": constituents[..., 1],
            "phi": constituents[..., 2],
            "mass": const_mass,
        },
        with_name="Momentum4D",
    )
    energy_const = ak.to_numpy(constituents_awk.e)
    pt_const = ak.to_numpy(constituents_awk.pt)
    eta_const = ak.to_numpy(constituents_awk.eta)
    phi_const = ak.to_numpy(constituents_awk.phi)

    # then create mask
    mask = np.expand_dims((constituents[..., 0] > 0).astype(int), axis=-1)

    # Now need to further convert the data to the format that is used in the training of Omnilearn
    # for jet it is (N, 2, 5), 5 features are pt, eta, phi, mass, multiplicity (num of non zero constituents)
    jet_data = np.concatenate([jet_data, np.sum(mask, axis=-2)], axis=-1)

    # for constituents it is (N, 2, pad_size, 4), 4 features are dEta, dPhi, log(pt_constituent), log(e_constituent)
    # also we only apply this calculation to masked constituents
    mask = mask.reshape(-1, 2, pad_size)
    const_dEta = eta_const - jet_eta
    const_dEta = const_dEta * mask
    const_dPhi = phi_const - jet_phi
    const_dPhi = const_dPhi * mask
    const_logPt = np.where(pt_const > 0, np.log(pt_const), 0)
    const_logPt = const_logPt * mask
    const_logE = np.where(energy_const > 0, np.log(energy_const), 0)
    const_logE = const_logE * mask

    # wrap phi between -pi and pi
    const_dPhi = np.where(
        const_dPhi > np.pi,
        const_dPhi - 2 * np.pi,
        const_dPhi,
    )
    const_dPhi = np.where(
        const_dPhi < -np.pi,
        const_dPhi + 2 * np.pi,
        const_dPhi,
    )

    # concate in shape of (N, 2, pad_size, 4)
    constituents = np.concatenate(
        [
            np.expand_dims(const_dEta, axis=-1),
            np.expand_dims(const_dPhi, axis=-1),
            np.expand_dims(const_logPt, axis=-1),
            np.expand_dims(const_logE, axis=-1),
        ],
        axis=-1,
    )

    print("after preprocessing:")
    print("jets shape: ", jet_data.shape)
    print("constituents shape: ", constituents.shape)

    dijet_mass = get_dijetmass(jet_data)
    mask_region = get_mjj_mask(dijet_mass)

    jet_data = jet_data[mask_region]
    constituents = constituents[mask_region]
    dijet_mass = dijet_mass[mask_region]

    print("after applying SR mask:")
    print("jets shape: ", jet_data.shape)
    print("constituents shape: ", constituents.shape)

    # preprocessing of nan and inf values
    jet_data[np.isnan(jet_data)] = 0.0
    jet_data[np.isinf(jet_data)] = 0.0
    constituents[np.isnan(constituents)] = 0.0
    constituents[np.isinf(constituents)] = 0.0

    # shuffle, take first 50k events as train, 25k as val, 25k as test
    np.random.seed(42)
    indices = np.arange(jet_data.shape[0])
    np.random.shuffle(indices)
    jet_data = jet_data[indices]
    constituents = constituents[indices]
    dijet_mass = dijet_mass[indices]

    events_to_keep = 50000

    jet_data = jet_data[:events_to_keep]
    constituents = constituents[:events_to_keep]
    dijet_mass = dijet_mass[:events_to_keep]

    # save to outputpath
    with h5py.File(outputpath, "w") as hf:
        hf.create_dataset("constituents", data=constituents)
        hf.create_dataset("global", data=jet_data)
        hf.create_dataset("condition", data=dijet_mass)
