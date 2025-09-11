# EventGenDelphes
Minimal python-based workflow to run particle physics simulations on the Perlmutter cluster using Madgraph, Pythia, and Delphes.
It also includes software to skim and plot key features of the events.

## Setup
To setup your working area do:
```bash
source setup.sh
```

When not done yet, this command installs the necessary software environment, pythia, and delphes.
Also it activates the software environment and sets the necessary environment variables.

## Run:

To generate signals at one mass
```
law run OmniLearnSignalPrep --version eventgen_production --mx 300 --my 300 --process qq --n-events 1000 --cluster-mode slurm (or local)
```

To get all inputs for Omnilearn PAWS, run each of the following 3 steps:

1. Generate all signal mass grids used in PAWS study

first run
```
sbatch slurm_submission.sh
```
This will do all the delphes + pythia works to get root files on slurm

then run
```
law run OmniLearnSignalPrepAllMass --version eventgen_production --process qq --n-events 500000 --workers 10 
```
to further process outputs in Ominlearn format, need ~ 200 GB of memory, recommanded to run on perlmutter cpu node

2. Generate backgrounds, run:
```
law run BackgroundProcessing --version eventgen_production --data-type qcd --region SR
law run BackgroundProcessing --version eventgen_production --data-type qcd --region SB
law run BackgroundProcessing --version eventgen_production --data-type extra_qcd --region SR
```
Need ~ 300 GB of memory, recommanded to run on perlmutter cpu node

3. For final outputs together, run:
```
law run FinalOutput --version eventgen_production
```
Need ~ 128GB of memory, recommanded to run on perlmutter cpu node