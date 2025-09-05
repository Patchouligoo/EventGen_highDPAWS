import os
import importlib
import subprocess
import shutil

import luigi
import law
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import dask
from dask.distributed import Client
from dask import delayed

from utils.numpy import NumpyEncoder
from utils.infrastructure import ClusterMixin
from utils.utils import RnD_txt_to_arrays
import share.features as features


class BaseTask(law.Task):
    """
    Base task which provides some convenience methods
    """

    version = law.Parameter(default="dev")

    def store_parts(self):
        task_name = self.__class__.__name__
        return (
            os.getenv("GEN_OUT"),
            f"version_{self.version}",
            task_name,
        )

    def local_path(self, *path):
        sp = self.store_parts()
        sp += path
        return os.path.join(*(str(p) for p in sp))

    def local_target(self, *path, **kwargs):
        return law.LocalFileTarget(self.local_path(*path), **kwargs)

    def local_directory_target(self, *path, **kwargs):
        return law.LocalDirectoryTarget(self.local_path(*path), **kwargs)


class DecayChannelMixin:
    process = luigi.ChoiceParameter(choices=["qq", "qqq"], default="qq")

    def store_parts(self):
        sp = super().store_parts()
        return sp + (f"process_{self.process}",)


class ProcessMixin(DecayChannelMixin):
    mx = luigi.FloatParameter(default=500.0)
    my = luigi.FloatParameter(default=100.0)

    def store_parts(self):
        sp = super().store_parts()
        return sp + (
            f"mx_{self.mx}",
            f"my_{self.my}",
        )


class DetectorMixin:
    @property
    def detector_config(self):
        return os.getenv("GEN_CODE") + "/config/detector/delphes_card_RnD.dat"


class NEventsMixin:
    n_events = luigi.IntParameter(default=1000)

    def store_parts(self):
        sp = super().store_parts()
        return sp + (f"n_events_{self.n_events}",)


class PythiaConfig(ProcessMixin, BaseTask):

    def output(self):
        return self.local_target(
            f"pythia_RnD_Z_XY_{self.process}_{self.mx}_{self.my}.cmnd"
        )

    @law.decorator.safe_output
    def run(self):
        base_process_card = (
            os.getenv("GEN_CODE")
            + f"/config/processes/LHCO/pythia_RnD_Z_XY_{self.process}.cmnd"
        )

        with open(base_process_card, "r") as f:
            pythia_config = f.read()

        pythia_config = pythia_config.replace("mx_PLACEHOLDER", str(self.mx))
        pythia_config = pythia_config.replace("my_PLACEHOLDER", str(self.my))

        # write to file
        self.output().parent.touch()
        self.output().dump(pythia_config, formatter="text")


class ChunkedEventsTask(NEventsMixin):
    n_max = luigi.IntParameter(default=1000000)

    @property
    def brakets(self):
        n_events = int(self.n_events)
        starts = range(0, n_events, self.n_max)
        stops = list(starts)[1:] + [n_events]
        brakets = zip(starts, stops)
        return list(brakets)

    @property
    def n_brakets(self):
        return len(self.brakets)

    @property
    def identifiers(self):
        return list(f"{i}_with_{int(self.n_max)}" for i in range(self.n_brakets))


class DelphesPythia8(
    DetectorMixin,
    ChunkedEventsTask,
    ProcessMixin,
    ClusterMixin,
    BaseTask,
):
    # SLURM Configuration
    cores = 1
    memory = "1GB"
    walltime = "24:00:00"
    qos = "shared"

    def output(self):
        return {
            identifier: {
                "config": self.local_target(f"{identifier}/config.txt"),
                "events": self.local_target(f"{identifier}/events.root"),
                "out": self.local_target(f"{identifier}/out.txt"),
            }
            for identifier in self.identifiers
        }

    def requires(self):
        return {"pythia_config": PythiaConfig.req(self)}

    @property
    def executable(self):
        return "DelphesPythia8"

    @staticmethod
    def call_with_output(cmd, out_path):
        with open(out_path, "w") as out_file:
            result = subprocess.call(cmd, stdout=out_file, stderr=out_file)
        return result

    @law.decorator.safe_output
    def run(self):
        detector_config = self.detector_config
        pythia_config = self.input()["pythia_config"].load(formatter="text")

        # Set up the tasks to compute
        cmds = []
        for identifier, (start, stop) in zip(self.identifiers, self.brakets):
            config_target = self.output()[identifier]["config"]
            events_target = self.output()[identifier]["events"]
            out_target = self.output()[identifier]["out"]
            # In case the task already successfully finished an identifier
            if events_target.exists():
                continue

            config_target.parent.touch()
            events_target.parent.touch()
            out_target.parent.touch()

            n_events = stop - start
            pythia_config = pythia_config.replace(
                "NEVENTS_PLACEHOLDER", str(int(n_events))
            )

            config_target.dump(pythia_config, formatter="text")

            cmd = [
                self.executable,
                detector_config,
                config_target.path,
                events_target.path,
            ]
            cmds.append((cmd, out_target.path))

        # Connect to the cluster
        cluster = self.start_cluster(len(cmds))
        client = Client(cluster)

        # Submit tasks
        tasks = [delayed(self.call_with_output)(cmd, out) for (cmd, out) in cmds]
        results = client.compute(tasks)

        # Gather the results
        results = client.gather(results)

        # Scale down and close the cluster
        cluster.scale(0)
        client.close()
        cluster.close()


class DelphesPythia8ROOTtoTXT(NEventsMixin, ProcessMixin, BaseTask):

    cluster_mode = luigi.ChoiceParameter(choices=["local", "slurm"], default="local")

    def requires(self):
        return DelphesPythia8.req(self)

    def output(self):
        return self.local_target(f"processed_{self.process}_{self.mx}_{self.my}.txt")

    @law.decorator.safe_output
    def run(self):

        inputs_root = self.input().values()
        inputs_root_paths = [input_root["events"].path for input_root in inputs_root]

        self.output().parent.touch()
        parent_dir = self.output().parent.path

        # hadd all root files in inputs_root_paths to one root file
        os.system(f"hadd -f {parent_dir}/all_events.root {' '.join(inputs_root_paths)}")

        os.system(
            f"root -b -x -q '{os.getenv('GEN_CODE')}/share/Delphes_files/LHCO_process.C(\"{parent_dir}/all_events.root\",\"{self.output().path}\")'"
        )

        # remove the all_events.root file
        os.system(f"rm {parent_dir}/all_events.root")


class DelphesPythia8TXTtoH5(NEventsMixin, ProcessMixin, BaseTask):
    feature_level = luigi.ChoiceParameter(
        choices=["low_level", "high_level"], default="low_level"
    )
    pad_size = luigi.IntParameter(default=150)

    cluster_mode = luigi.ChoiceParameter(choices=["local", "slurm"], default="local")

    def store_parts(self):
        if self.feature_level == "low_level":
            return (
                super().store_parts()
                + (f"feature_level_{self.feature_level}",)
                + (f"constituent_pad_{self.pad_size}",)
            )
        else:
            return super().store_parts() + (f"feature_level_{self.feature_level}",)

    def requires(self):
        return DelphesPythia8ROOTtoTXT.req(self)

    def output(self):
        return self.local_target(f"processed_{self.process}_{self.mx}_{self.my}.h5")

    @law.decorator.safe_output
    def run(self):

        output_array = RnD_txt_to_arrays(
            self.input().path,
            feature_level=self.feature_level,
            flatten=True,
            sortby="pt",
            pad_size=self.pad_size,
        )

        df = pd.DataFrame(output_array)

        # save only the features in the feature list
        if self.feature_level == "high_level":
            features_list = df.columns
            features_list = [
                f
                for f in features_list
                if f in features.high_level_features_signal_lhco
            ]
            df = df[features_list]
            df["mx"] = self.mx
            df["my"] = self.my

        self.output().parent.touch()
        df.to_hdf(self.output().path, key="output", mode="w")


class OmniLearnSignalPrep(NEventsMixin, ProcessMixin, BaseTask):
    pad_size = luigi.IntParameter(default=150)

    cluster_mode = luigi.ChoiceParameter(choices=["local", "slurm"], default="local")

    def store_parts(self):
        return super().store_parts() + (f"constituent_pad_{self.pad_size}",)

    def requires(self):
        return DelphesPythia8TXTtoH5.req(
            self, feature_level="low_level", pad_size=self.pad_size
        )

    def output(self):
        return self.local_target(
            f"processed_data_signal_{self.process}_{self.mx}_{self.my}.h5"
        )

    @law.decorator.safe_output
    def run(self):
        from data_processing.signal_processing import signal_prep

        self.output().parent.touch()
        signal_prep(self.input().path, self.output().path, self.pad_size)


# class GenerateSignalsAllMass(
#     NEventsMixin,
#     DecayChannelMixin,
#     BaseTask,
# ):
#     feature_level = luigi.ChoiceParameter(
#         choices=["low_level", "high_level"], default="low_level"
#     )
#     pad_size = luigi.IntParameter(default=150)

#     cluster_mode = luigi.ChoiceParameter(choices=["local", "slurm"], default="local")

#     mx_values = np.linspace(50, 600, 12)
#     my_values = np.linspace(50, 600, 12)

#     def requires(self):
#         reqs = {}
#         for mx in self.mx_values:
#             for my in self.my_values:
#                 req = DelphesPythia8TXTtoH5.req(
#                     self,
#                     mx=mx,
#                     my=my,
#                 )
#                 key = f"mx_{mx}_my_{my}"
#                 reqs[key] = req
#         return reqs

#     def output(self):
#         return self.local_directory_target(f"job_status.txt")

#     @law.decorator.safe_output
#     def run(self):

#         print("All tasks finished")
#         self.output().parent.touch()
#         with open(self.output().path, "w") as f:
#             f.write("All tasks finished\n")
