import os
import sys
import glob
import json
import subprocess
from quickstats import stdout


def download_file(url: str, outdir: str):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, os.path.basename(url))
    command = ["wget", "-O", outpath, url]

    try:
        subprocess.run(command, check=True)
        stdout.info(f"File downloaded to {outpath}")
    except subprocess.CalledProcessError as e:
        stdout.error(f"An error occurred: {e}")
