# BIC coursework: ANN trained by PSO (banknote authentication)

Pair coursework for Biologically-Inspired Computation. A feedforward ANN, written from
scratch, is trained by a from-scratch Particle Swarm Optimisation (Algorithm 39,
"Essentials of Metaheuristics", with informants) to classify forged banknotes.

## Files
| File | Purpose |
|---|---|
| `ann.py` | ANN (configurable layers, nodes, activations: logistic/ReLU/tanh). `ANNLearnedActs` = extension where PSO also chooses the hidden activations |
| `pso.py` | PSO, Algorithm 39 with informants; comments give Algorithm 39 line numbers; boundary handling: none / clip / reflect |
| `data.py` | load data, stratified 70/30 split, standardise (train statistics only) |
| `run.py` | couples PSO to the ANN (a particle = all weights and biases; fitness = training accuracy) and runs one configuration |
| `sweep.py` | runs a named grid of configurations x 10 runs, saves mean/sd and convergence data to `results/` |
| `plot.py` | bar charts (mean ± sd) and convergence curves from `results/` |
| `compare.py` | Welch t-test between two configurations |
| `BIC_coursework.ipynb` | Colab notebook that runs everything |

## How to run (Google Colab, easiest)
1. Open `BIC_coursework.ipynb` in Colab (colab.research.google.com -> File -> Upload notebook).
2. Run cells top to bottom. Cell 1 asks you to upload `data_banknote_authentication.txt`
   (UCI banknote authentication dataset: https://archive.ics.uci.edu/dataset/267/banknote+authentication).

## How to run (own computer)
```
pip install numpy scipy matplotlib
python run.py --runs 10                                   # one configuration
python run.py --layers 4 4 1 --learn_acts --runs 10       # PSO also picks hidden activations
python sweep.py list                                      # list experiments
python sweep.py budget && python plot.py budget           # run + plot one experiment
python compare.py budget swarm10x50it swarm50x10it        # significance test
```
Put `data_banknote_authentication.txt` in the same folder.

## Team
- Jashwanth Ijjarotu
- Nourin Nawshad

## Declaration
Code was developed with help from generative AI (Claude Sonnet 5.5). See `AI_USE_LOG.md`
and the report appendix. Experiments, analysis and the report are our own work.
