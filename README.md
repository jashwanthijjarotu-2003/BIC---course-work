# BIC Coursework (F20BC/F21BC): ANN trained by PSO

A feedforward neural network and a Particle Swarm Optimiser (Algorithm 39, Luke,
*Essentials of Metaheuristics*, Section 3.5), both written from scratch in Python/NumPy.
PSO trains the ANN's weights and biases to classify the UCI banknote authentication dataset.

## Requirements

- Python 3.9 or later
- numpy, scipy, matplotlib

```
pip install -r requirements.txt
```

NumPy is used only for numerical work. No ANN or PSO library is used. SciPy is used only
for the t-test and F-test, and matplotlib only for the plots.

## Data

Download `data_banknote_authentication.txt` from
https://archive.ics.uci.edu/dataset/267/banknote+authentication
and put it in the same folder as the `.py` files. To use another location, pass `--data <path>`.

## Files

| File | Purpose |
|---|---|
| `data.py` | Loads the data, makes a stratified 70/30 train/test split, and standardises the features using training statistics only |
| `ann.py` | Feedforward ANN with configurable layers, neurons and activation per layer (logistic, ReLU, tanh). Exposes all weights and biases as one flat vector. `ANNLearnedActs` adds one gene per hidden layer so PSO can also choose the activation function |
| `pso.py` | PSO with informants (Algorithm 39). Comments give the matching algorithm line numbers (`Alg39 L<n>`). Extensions: boundary handling (none / clip / reflect) and informant strategy (random / nearest-neighbour) |
| `adapter.py` | Couples PSO to the ANN. Loads a particle's vector into the ANN and returns the training loss (MSE, BCE or 0–1 error) for PSO to minimise |
| `metrics.py` | Accuracy, precision, recall and F1 |
| `run.py` | Trains one configuration for N runs and prints mean ± SD |
| `sweep.py` | Runs a named experiment (a grid of configurations × N runs) and saves results to `results/` |
| `plot.py` | Makes bar charts and convergence plots from the saved results |
| `compare.py` | Welch's t-test between two configurations of one experiment |
| `BIC_coursework-2.ipynb` | Colab notebook that writes all the files above and runs every experiment in order |
| report_figures.py | Redraws the report's Figures 1–3 from the notebook outputs |

## How to run

**Option A: Google Colab (easiest).** Open `BIC_coursework-2.ipynb`, upload the data
file when asked, and run all cells from top to bottom.

**Option B: command line.** Run these from the folder that holds the `.py` files.

Check that PSO works on its own (no ANN involved; minimises f(x) = sum(x²)):
```
python pso.py
```

Train the default configuration (4-4-1 network, ReLU hidden, logistic output, swarm 20,
25 iterations) for 10 runs:
```
python run.py --runs 10
```

Change any setting from the command line, for example:
```
python run.py --layers 4 8 1 --acts tanh logistic --swarm 10 --iters 50
python run.py --alpha 0.5 --beta 2.0 --gamma 1.0 --delta 0.5 --epsilon 1.0
python run.py --informants 5 --informant_strategy nearest --boundary reflect
python run.py --loss error          # train on 1 - accuracy directly
python run.py --learn_acts          # PSO also chooses the hidden activation
```
See all options with `python run.py --help`.

Run the experiments from the report:
```
python sweep.py list                # show every experiment and its configurations
python sweep.py budget              # one experiment (10 runs per configuration)
python sweep.py all --runs 10       # every experiment
python plot.py budget               # figures for one experiment (or: python plot.py all)
python compare.py budget swarm10x50it swarm100x5it        # t-test on test accuracy
python compare.py loss mse bce f1                         # t-test on another metric
```

| Experiment | What it varies |
|---|---|
| `arch` | Hidden layers / neurons × activation |
| `budget` | Swarm size × iterations at a fixed 500 evaluations |
| `coef` | β/γ balance and δ |
| `informants` | Number of informants (1, 2, 3, 5, 10) |
| `inf_strategy` | Random vs nearest-neighbour informants |
| `boundary` | none / clip / reflect |
| `loss` | MSE / BCE / 0–1 error as the PSO fitness |
| `learned_acts` | PSO-chosen activation vs fixed activations |

## Output

`sweep.py` writes to `results/`:
- `<exp>_summary.csv`: mean and SD of train/test accuracy, training loss, precision, recall and F1 for each configuration
- `<exp>_hist.json`: best training loss per iteration for every run (used for the convergence plots)

`plot.py` writes `results/<exp>_bars.png` and `results/<exp>_convergence.png`.

## Reproducibility

Run *r* uses random seed *r* (0–9 for 10 runs), and the train/test split uses seed 0
(change it with `--split_seed`). Re-running a command gives the same numbers.
