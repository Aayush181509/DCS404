# DCS 404 — Artificial Intelligence and Machine Learning

Lecture notebooks and course website for **DCS 404: Artificial Intelligence & Machine Learning**, an
8-week undergraduate course. Each module is a runnable Jupyter notebook that explains a topic from first
principles, builds the core algorithm from scratch, and checks it against a standard library.

**Course website:** <https://aayush181509.github.io/DCS404/>

---

## What's in each module

Every lecture notebook follows the same structure:

1. **How to work through this** and **Learning objectives**
2. **Setup**: imports, a shared plotting style, a fixed random seed
3. **Numbered sections**: concept → from-scratch implementation → comparison with scikit-learn / statsmodels /
   PyTorch → visual results
4. **Your turn**: exercises
5. **If you remember nothing else**: the key takeaways
6. **Further reading and glossary**

Every number quoted in the text comes from the executed code. Run the notebook top to bottom and you'll get
the same results.

---

## Modules

| Notebook | Topic | Main libraries |
|---|---|---|
| [`17_search_and_games`](notebooks/17_search_and_games.ipynb) | Principles of AI: agents, BFS/DFS, heuristics, A\*, Minimax, alpha-beta, Connect Four | `numpy`, `matplotlib` |
| [`01_introduction_to_machine_learning`](notebooks/01_introduction_to_machine_learning.ipynb) | What ML is, types of learning, a first model | `scikit-learn` |
| [`02_ml_workflow`](notebooks/02_ml_workflow.ipynb) | CRISP-DM, end-to-end workflow, model persistence | `scikit-learn`, `joblib` |
| [`03_regression`](notebooks/03_regression.ipynb) | Simple, multiple and polynomial regression | `scikit-learn` |
| [`04_gradient_descent`](notebooks/04_gradient_descent.ipynb) | Cost functions, learning rate, convexity | `numpy` |
| [`05_regularization`](notebooks/05_regularization.ipynb) | Ridge, Lasso, Elastic Net | `scikit-learn` |
| [`06_logistic_regression`](notebooks/06_logistic_regression.ipynb) | Sigmoid, cross-entropy, decision boundaries | `scikit-learn` |
| [`07_decision_trees`](notebooks/07_decision_trees.ipynb) | Gini, entropy, pruning | `scikit-learn` |
| [`08_evaluation_metrics`](notebooks/08_evaluation_metrics.ipynb) | Confusion matrix, precision/recall, ROC/PR, regression metrics | `scikit-learn` |
| [`09_knn`](notebooks/09_knn.ipynb) | K-Nearest Neighbors, distance metrics, scaling | `scikit-learn` |
| [`15_kmeans_clustering`](notebooks/15_kmeans_clustering.ipynb) | K-Means from scratch, K-Means++, elbow and silhouette | `scikit-learn` |
| [`16_recommender_systems`](notebooks/16_recommender_systems.ipynb) | Content-based and collaborative filtering, matrix factorization | `numpy`, `scikit-learn` |
| [`10_time_series_foundations`](notebooks/10_time_series_foundations.ipynb) | Trend, seasonality, stationarity, baselines | `statsmodels` |
| [`11_exponential_smoothing`](notebooks/11_exponential_smoothing.ipynb) | Simple, Holt and Holt-Winters smoothing | `statsmodels` |
| [`12_arima_models`](notebooks/12_arima_models.ipynb) | ARIMA, SARIMA, Box-Jenkins | `statsmodels` |
| [`13_neural_networks`](notebooks/13_neural_networks.ipynb) | A NumPy MLP from scratch: backprop, Adam, normalization, regularization | `numpy`, `scikit-learn` |
| [`14_pytorch_ann`](notebooks/14_pytorch_ann.ipynb) | Tensors, autograd, `nn.Module`, an MNIST classifier | `torch`, `torchvision` |
| [`project/00_final_project`](notebooks/project/00_final_project.ipynb) | Final project brief | — |

File numbers reflect the order the modules were written. The table, and the website navigation, use teaching
order.

### Mapping to the syllabus

The required text is Nagy, Z. (2018). *Artificial Intelligence and Machine Learning Fundamentals*. Packt.

| Week | Textbook reading | Notebooks |
|---|---|---|
| 1–2 | Python basics (DataCamp) | — |
| 3 | Ch 1 Principles of AI · Ch 2 AI with Search Techniques and Games | `17` (plus `01` for ML framing) |
| 4 | Data preparation and feature engineering | `02` |
| 5 | Ch 3 Regression · Ch 4 Classification | `03`, `04`, `05`, `06`, `08`, `09` |
| 6 | Ch 5 Using Trees for Predictive Analysis | `07` |
| 7 | Ch 6 Clustering | `15` |
| 8 | Ch 7 Deep Learning with Neural Networks | `13`, `14` |

`10`–`12` (time series) and `16` (recommender systems) are supplementary modules beyond the textbook.

---

## Getting started

### 1. Clone

```bash
git clone https://github.com/Aayush181509/DCS404.git
cd DCS404
```

### 2. Create an environment and install dependencies

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` and `torchvision` are only needed for `14_pytorch_ann`. On a machine with an NVIDIA GPU, install the
CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) first.

A step-by-step setup guide (Python, Git, VS Code, virtual environments) is in
[`docs/InstallationGuide.md`](docs/InstallationGuide.md).

### 3. Run the notebooks

```bash
jupyter lab notebooks/
```

Run cells in order (`Shift + Enter`). Notebooks load data from `notebooks/data/` and work whether Jupyter is
started from the repo root or from `notebooks/`.

**Google Colab:** open a notebook from GitHub through *File → Open notebook → GitHub*. Notebooks that read a CSV
need the matching file from `notebooks/data/` uploaded alongside them.

---

## Datasets

The small datasets are committed in `notebooks/data/`:

| File | Used in | Source |
|---|---|---|
| `Advertising.csv` | Regression, gradient descent | *An Introduction to Statistical Learning* |
| `diabetes.csv` | Logistic regression, decision trees, metrics, KNN, K-Means | Pima Indians Diabetes (UCI / Kaggle) |
| `student-mat.csv` | ML workflow, regression, gradient descent, regularization | UCI Student Performance (Cortez & Silva, 2008) |
| `air_passengers.csv` | Time series modules | Box & Jenkins airline passengers |

MNIST (`14_pytorch_ann`) is **not** committed. `torchvision` downloads it to `notebooks/data/mnist/` the
first time the notebook runs (about 65 MB).

---

## Course website (MkDocs)

The site at <https://aayush181509.github.io/DCS404/> is built with [MkDocs](https://www.mkdocs.org/) from
Markdown versions of the notebooks in `docs/DCS404/`.

```bash
pip install mkdocs pymdown-extensions

make serve                    # preview locally at http://127.0.0.1:8000
make build                    # build into site/
make deploy                   # publish to the gh-pages branch
```

To add or update a module:

```bash
# 1. Convert the executed notebook to Markdown
make convert NOTEBOOK=notebooks/17_search_and_games.ipynb

# 2. If it uses static images from notebooks/resources/images/, copy them to docs/DCS404/assets/
#    and point the Markdown at ../assets/ instead of resources/images/

# 3. Add the page under `nav:` in mkdocs.yml
```

`make convert-all` converts every notebook in `notebooks/`. The final project brief is converted separately.

---

## Repository layout

```
DCS404/
├── notebooks/
│   ├── 01_…17_*.ipynb           # lecture modules
│   ├── project/                  # final project brief
│   ├── data/                     # datasets (MNIST downloaded on demand)
│   └── resources/
│       ├── images/               # static diagrams used by the notebooks
│       └── references/           # source/reference notebooks and readings by topic
├── docs/
│   ├── index.md                  # website home page
│   ├── InstallationGuide.md
│   ├── javascripts/mathjax.js    # LaTeX rendering on the site
│   └── DCS404/                   # converted lecture pages, figures and assets
├── mkdocs.yml                    # site config and navigation
├── Makefile                      # convert / serve / build / deploy shortcuts
├── requirements.txt
└── LICENSE                       # Apache 2.0
```

---

## A note on the textbook

The Week 3 module re-implements the textbook's Chapter 1–2 code and found several errors, including a counting
bug that inflates the number of Tic-Tac-Toe games, swapped alpha-beta initial bounds, and library calls that no
longer run on current NumPy and easyAI. Section 13 of
[`17_search_and_games`](notebooks/17_search_and_games.ipynb) lists each one with its correction.

---

## License

Released under the [Apache License 2.0](LICENSE).

Facilitator: **Aayush Raj Regmi** · [LinkedIn](https://www.linkedin.com/in/aayush-regmi-6631ab1b5/)
