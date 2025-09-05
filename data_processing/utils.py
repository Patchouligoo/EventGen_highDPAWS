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
