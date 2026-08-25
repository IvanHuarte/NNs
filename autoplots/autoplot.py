#!/home/ihuarte/Escritorio/Ivan/NNs/.venv/bin/python

import argparse
import glob
import json
import os
from pathlib import Path

from NNs.autoplots.plot_from_artifact import crossed_artifact_plots

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--directory",
    required=True,
    help="Directory of MPS artifacts (.json)",
)

args = parser.parse_args()
directory = args.directory

project_folder = str(Path(__file__).parent) + "/"
with open(project_folder + "/config_plots.json", "r") as f:
    config_plot = json.load(f)

files = glob.glob(f"{directory}/**/*results*.json", recursive=True)

directory += "Figures/"
os.makedirs(directory, exist_ok=True)


crossed_artifact_plots(files, config_plot, directory)
