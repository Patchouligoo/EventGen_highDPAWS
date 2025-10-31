import luigi
import law
import pandas as pd
import numpy as np
import h5py as h5
from tqdm import tqdm

from tasks.signal_generation import (
    OmniLearnSignalPrep,
    NEventsMixin,
    DecayChannelMixin,
    BaseTask,
)
from tasks.bkg_generation import BackgroundProcessing


class FinalOutput(
    BaseTask,
):

    pad_size = luigi.IntParameter(default=150)
    mx_values = np.linspace(50, 600, 12)
    my_values = np.linspace(50, 600, 12)

    def requires(self):

        reqs = {}
        for mx in self.mx_values:
            for my in self.my_values:
                req = OmniLearnSignalPrep.req(
                    self, mx=mx, my=my, n_events=500000, process="qq"
                )
                key = f"mx_{mx}_my_{my}"
                reqs[key] = req

        reqs["SR_data_bkg"] = BackgroundProcessing.req(
            self,
            data_type="qcd",
            region="SR",
        )
        reqs["SB_data_bkg"] = BackgroundProcessing.req(
            self,
            data_type="qcd",
            region="SB",
        )
        reqs["SR_MC_bkg"] = BackgroundProcessing.req(
            self,
            data_type="extra_qcd",
            region="SR",
        )

        return reqs

    def output(self):
        return {
            # used to form pseudo data, use bkg from LHCO original samples
            "SR_data_bkg": self.local_target(f"SR_data_bkg.h5"),
            "SR_data_sig": self.local_target(f"SR_data_sig.h5"),
            "SB_data_bkg": self.local_target(f"SB_data_bkg.h5"),
            # used to form MC simulation, use bkg from extra QCD samples
            "SR_MC_bkg": self.local_target(f"SR_MC_bkg.h5"),
            "SR_MC_sig": self.local_target(f"SR_MC_sig.h5"),
        }

    @law.decorator.safe_output
    def run(self):

        np.random.seed(42)

        # --------------------- signals for data ---------------------
        df_SR_data_sig_constituents = None
        df_SR_data_sig_global = None
        df_SR_data_sig_condition = None
        df_SR_data_highlevel_features = None
        df_SR_data_sig_param_mx = None
        df_SR_data_sig_param_my = None

        pbar = tqdm(self.mx_values, desc="Processing signals for data")
        for mx in pbar:
            for my in self.my_values:
                key = f"mx_{mx}_my_{my}"
                SR_data_sig_path_mxmy = self.input()[key]["signals_for_data"].path

                if df_SR_data_sig_constituents is None:
                    with h5.File(SR_data_sig_path_mxmy, "r") as hf:
                        df_SR_data_sig_constituents = hf["constituents"][:]
                        df_SR_data_sig_global = hf["global"][:]
                        df_SR_data_sig_condition = hf["condition"][:]
                        df_SR_data_highlevel_features = hf["highlevel_features"][:]

                        df_SR_data_sig_param_mx = np.full(
                            hf["constituents"][:].shape[0], mx
                        )
                        df_SR_data_sig_param_my = np.full(
                            hf["constituents"][:].shape[0], my
                        )

                else:
                    with h5.File(SR_data_sig_path_mxmy, "r") as hf:
                        df_SR_data_sig_constituents = np.concatenate(
                            (df_SR_data_sig_constituents, hf["constituents"][:]), axis=0
                        )
                        df_SR_data_sig_global = np.concatenate(
                            (df_SR_data_sig_global, hf["global"][:]), axis=0
                        )
                        df_SR_data_sig_condition = np.concatenate(
                            (df_SR_data_sig_condition, hf["condition"][:]), axis=0
                        )
                        df_SR_data_highlevel_features = np.concatenate(
                            (
                                df_SR_data_highlevel_features,
                                hf["highlevel_features"][:],
                            ),
                            axis=0,
                        )

                        df_SR_data_sig_param_mx = np.concatenate(
                            (
                                df_SR_data_sig_param_mx,
                                np.full(hf["constituents"][:].shape[0], mx),
                            ),
                            axis=0,
                        )
                        df_SR_data_sig_param_my = np.concatenate(
                            (
                                df_SR_data_sig_param_my,
                                np.full(hf["constituents"][:].shape[0], my),
                            ),
                            axis=0,
                        )

        print(
            "Total number of available signal events for data: ",
            df_SR_data_sig_constituents.shape[0],
        )

        # shuffle signals for data
        indices = np.arange(df_SR_data_sig_constituents.shape[0])
        np.random.shuffle(indices)
        df_SR_data_sig_constituents = df_SR_data_sig_constituents[indices]
        df_SR_data_sig_global = df_SR_data_sig_global[indices]
        df_SR_data_sig_condition = df_SR_data_sig_condition[indices]
        df_SR_data_sig_param_mx = df_SR_data_sig_param_mx[indices]
        df_SR_data_sig_param_my = df_SR_data_sig_param_my[indices]
        df_SR_data_highlevel_features = df_SR_data_highlevel_features[indices]

        # store signals for data
        self.output()["SR_data_sig"].parent.touch()
        with h5.File(self.output()["SR_data_sig"].path, "w") as hf:
            hf.create_dataset("constituents", data=df_SR_data_sig_constituents)
            hf.create_dataset("global", data=df_SR_data_sig_global)
            hf.create_dataset("condition", data=df_SR_data_sig_condition)
            hf.create_dataset("highlevel_features", data=df_SR_data_highlevel_features)
            hf.create_dataset("param_mx", data=df_SR_data_sig_param_mx)
            hf.create_dataset("param_my", data=df_SR_data_sig_param_my)

        # --------------------- signals for MC ---------------------
        df_SR_MC_sig_constituents = None
        df_SR_MC_sig_global = None
        df_SR_MC_sig_condition = None
        df_SR_MC_highlevel_features = None
        df_SR_MC_sig_param_mx = None
        df_SR_MC_sig_param_my = None

        pbar = tqdm(self.mx_values, desc="Processing signals for MC")
        for mx in pbar:
            for my in self.my_values:
                key = f"mx_{mx}_my_{my}"
                SR_MC_sig_path_mxmy = self.input()[key]["signals_for_mc"].path

                if df_SR_MC_sig_constituents is None:
                    with h5.File(SR_MC_sig_path_mxmy, "r") as hf:
                        df_SR_MC_sig_constituents = hf["constituents"][:]
                        df_SR_MC_sig_global = hf["global"][:]
                        df_SR_MC_sig_condition = hf["condition"][:]
                        df_SR_MC_highlevel_features = hf["highlevel_features"][:]

                        df_SR_MC_sig_param_mx = np.full(
                            hf["constituents"][:].shape[0], mx
                        )
                        df_SR_MC_sig_param_my = np.full(
                            hf["constituents"][:].shape[0], my
                        )

                else:
                    with h5.File(SR_MC_sig_path_mxmy, "r") as hf:
                        df_SR_MC_sig_constituents = np.concatenate(
                            (df_SR_MC_sig_constituents, hf["constituents"][:]), axis=0
                        )
                        df_SR_MC_sig_global = np.concatenate(
                            (df_SR_MC_sig_global, hf["global"][:]), axis=0
                        )
                        df_SR_MC_sig_condition = np.concatenate(
                            (df_SR_MC_sig_condition, hf["condition"][:]), axis=0
                        )
                        df_SR_MC_highlevel_features = np.concatenate(
                            (
                                df_SR_MC_highlevel_features,
                                hf["highlevel_features"][:],
                            ),
                            axis=0,
                        )

                        df_SR_MC_sig_param_mx = np.concatenate(
                            (
                                df_SR_MC_sig_param_mx,
                                np.full(hf["constituents"][:].shape[0], mx),
                            ),
                            axis=0,
                        )
                        df_SR_MC_sig_param_my = np.concatenate(
                            (
                                df_SR_MC_sig_param_my,
                                np.full(hf["constituents"][:].shape[0], my),
                            ),
                            axis=0,
                        )

        print(
            "Total number of available signal events for MC: ",
            df_SR_MC_sig_constituents.shape[0],
        )
        # shuffle signals for MC
        indices = np.arange(df_SR_MC_sig_constituents.shape[0])
        np.random.shuffle(indices)
        df_SR_MC_sig_constituents = df_SR_MC_sig_constituents[indices]
        df_SR_MC_sig_global = df_SR_MC_sig_global[indices]
        df_SR_MC_sig_condition = df_SR_MC_sig_condition[indices]
        df_SR_MC_highlevel_features = df_SR_MC_highlevel_features[indices]
        df_SR_MC_sig_param_mx = df_SR_MC_sig_param_mx[indices]
        df_SR_MC_sig_param_my = df_SR_MC_sig_param_my[indices]

        # store signals for MC
        with h5.File(self.output()["SR_MC_sig"].path, "w") as hf:
            hf.create_dataset("constituents", data=df_SR_MC_sig_constituents)
            hf.create_dataset("global", data=df_SR_MC_sig_global)
            hf.create_dataset("condition", data=df_SR_MC_sig_condition)
            hf.create_dataset("highlevel_features", data=df_SR_MC_highlevel_features)
            hf.create_dataset("param_mx", data=df_SR_MC_sig_param_mx)
            hf.create_dataset("param_my", data=df_SR_MC_sig_param_my)

        # --------------------- backgrounds for data ---------------------
        SR_data_bkg_path = self.input()["SR_data_bkg"].path
        # No need to do any processing, just copy the file
        with h5.File(SR_data_bkg_path, "r") as hf_in:
            with h5.File(self.output()["SR_data_bkg"].path, "w") as hf_out:
                for key in hf_in.keys():
                    hf_out.create_dataset(key, data=hf_in[key][:])
                print(
                    "Total number of available background events for data in SR: ",
                    hf_in["constituents"][:].shape[0],
                )

        SB_data_bkg_path = self.input()["SB_data_bkg"].path
        # No need to do any processing, just copy the file
        with h5.File(SB_data_bkg_path, "r") as hf_in:
            with h5.File(self.output()["SB_data_bkg"].path, "w") as hf_out:
                for key in hf_in.keys():
                    hf_out.create_dataset(key, data=hf_in[key][:])
                print(
                    "Total number of available background events for data in SB: ",
                    hf_in["constituents"][:].shape[0],
                )

        # --------------------- backgrounds for MC ---------------------
        SR_MC_bkg_path = self.input()["SR_MC_bkg"].path
        # No need to do any processing, just copy the file
        with h5.File(SR_MC_bkg_path, "r") as hf_in:
            with h5.File(self.output()["SR_MC_bkg"].path, "w") as hf_out:
                for key in hf_in.keys():
                    hf_out.create_dataset(key, data=hf_in[key][:])
                print(
                    "Total number of available background events for MC in SR: ",
                    hf_in["constituents"][:].shape[0],
                )
