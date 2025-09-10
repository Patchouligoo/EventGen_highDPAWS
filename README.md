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

to generate signals at one mass
```
law run DelphesPythia8TXTtoH5 --version eventgen_test_0 --mx 300 --my 300 --process qq --n-events 1000 --cluster-mode slurm (or local)
```

to generate all signal mass grids used in PAWS study, run

```
law run SubmitSignalsAllMass --version eventgen_test_0 --process qq --n-events 500000 --cluster-mode slurm --workers 144
```
needs to be run on perlmutter cpu node with slurm
