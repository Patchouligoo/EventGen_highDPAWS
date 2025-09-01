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
