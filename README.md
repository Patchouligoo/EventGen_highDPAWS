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

To generate all signal mass grids used in PAWS study

first run
```
law run SubmitSignalsAllMass --version eventgen_production --process qq --n-events 500000 --cluster-mode slurm --workers 144
```
needs to be run on perlmutter cpu node with slurm. This will do all the delphes + pythia works to get root files

then run
```
law run OmniLearnSignalPrepAllMass --version eventgen_production --process qq --n-events 500000 --workers 10 
```
to further process outputs in Ominlearn format
