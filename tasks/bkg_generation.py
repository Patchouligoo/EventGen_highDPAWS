from tasks.signal_generation import BaseTask
import os
import importlib
import subprocess
import shutil

import luigi
import law
import pandas as pd


class BackgroundDownloading(BaseTask):

    def output(self):
        return {
            "lhco_extra_qcd_bkg": {
                "low_level": self.local_target(
                    "events_anomalydetection_qcd_extra_inneronly_4vecs_and_features.h5"
                ),
                "high_level": self.local_target(
                    "events_anomalydetection_qcd_extra_inneronly_features.h5"
                ),
            },
            "lhco_qcd_bkg": {
                "low_level": self.local_target("events_anomalydetection_v2.h5"),
                "high_level": self.local_target(
                    "events_anomalydetection_v2.features.h5"
                ),
            },
        }

    @law.decorator.safe_output
    def run(self):
        from share.path import lhco_extra_qcd_bkg, lhco_qcd_mixed
        from utils.download import download_file

        self.output()["lhco_extra_qcd_bkg"]["low_level"].parent.touch()
        downloading_dir = self.output()["lhco_extra_qcd_bkg"]["low_level"].parent.path

        if not os.path.exists(
            f"{downloading_dir}/{lhco_extra_qcd_bkg['low_level']['name']}"
        ):
            download_file(lhco_extra_qcd_bkg["low_level"]["path"], downloading_dir)

        if not os.path.exists(
            f"{downloading_dir}/{lhco_extra_qcd_bkg['high_level']['name']}"
        ):
            download_file(lhco_extra_qcd_bkg["high_level"]["path"], downloading_dir)

        if not os.path.exists(
            f"{downloading_dir}/{lhco_qcd_mixed['low_level']['name']}"
        ):
            download_file(lhco_qcd_mixed["low_level"]["path"], downloading_dir)

        if not os.path.exists(
            f"{downloading_dir}/{lhco_qcd_mixed['high_level']['name']}"
        ):
            download_file(lhco_qcd_mixed["high_level"]["path"], downloading_dir)


class BackgroundProcessing(BaseTask):

    pad_size = luigi.IntParameter(default=150)
    data_type = luigi.ChoiceParameter(choices=["qcd", "extra_qcd"], default="qcd")

    region = luigi.ChoiceParameter(choices=["SR", "SB"], default="SR")

    def requires(self):
        return BackgroundDownloading.req(self)

    def store_parts(self):

        out = (
            super().store_parts()
            + (f"data_type_{self.data_type}",)
            + (f"region_{self.region}",)
            + (f"constituent_pad_{self.pad_size}",)
        )

        return out

    def output(self):
        return self.local_target(f"processed_data_bkg.h5")

    @law.decorator.safe_output
    def run(self):

        from data_processing.bkg_processing import (
            # process_lhco_highlevel_qcdbkg,
            process_extra_qcdbkg,
            process_qcdbkg,
        )

        self.output().parent.touch()

        if self.data_type == "extra_qcd":
            # 2214 columns
            # 700 particles, each with 3 features (pt, eta, phi) + 14 high level features in  px py pz (can ignore)
            # Need to run jet clustering
            process_extra_qcdbkg(
                self.input()["lhco_extra_qcd_bkg"],
                self.output().path,
                pad_size=self.pad_size,
                region=self.region,
            )

        elif self.data_type == "qcd":
            # bkg and signal, in shape of pt, eta, phi for 700 parts so 2100 columns
            # need to first select bkg only, then run jet clustering
            process_qcdbkg(
                self.input()["lhco_qcd_bkg"],
                self.output().path,
                pad_size=self.pad_size,
                region=self.region,
            )

        else:
            raise ValueError("data_type must be qcd or extra_qcd")
