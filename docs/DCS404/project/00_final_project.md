# DCS 404 Machine Learning and Artificial Intelligence — Course Project

## Building an End-to-End Machine Learning Application

---

### Objective

Students will apply the machine learning workflow to a real-world problem in a domain of their interest
(e.g. Healthcare, Finance, Sports, Education, Media, Entertainment, Agriculture). The project focuses on:

- Framing a real problem as a well-defined ML problem.
- Collecting, cleaning and exploring data.
- Building, comparing and evaluating models against a baseline.
- Structuring the work as a proper project rather than a single notebook.
- Delivering a simple working application with a live demo.

**Keep it small and finished.** A logistic regression behind a working Streamlit page scores higher than
an ambitious model with no application around it.

---

### Project Tracks

Pick **one track**. Both are equally acceptable and marked to the same total. Your track decides what
your weekly deliverables look like — see the timeline below.

These are the two families covered in depth in this course. Other project types (time series
forecasting, reinforcement learning) are only allowed **with your teacher's prior approval**.

| Track | Example Projects |
|---|---|
| **A. Supervised Learning** | Student dropout risk, loan default prediction, house price estimation, disease risk, match outcome prediction |
| **B. Unsupervised / Recommender** | Customer segmentation, movie or music recommender, anomaly detection in transactions |

Each track has a different notion of "data", "baseline" and "good result":

| Track | Your data is… | Your baseline is… | You are judged on… |
|---|---|---|---|
| **A. Supervised** | A labelled table | Majority class / mean prediction | Metric on a held-out test set (accuracy, F1, RMSE…) |
| **B. Unsupervised** | An unlabelled table or user–item matrix | Random or most-popular recommendations | Interpretable clusters, or offline ranking metrics (precision@10, hit-rate) |

### Sample Project Ideas by Domain

| Domain | Project Idea |
|---|---|
| Healthcare | Predict hospital readmission or appointment no-shows |
| Finance | Loan default prediction or credit risk scoring |
| Education | Predict student performance or dropout risk from attendance data |
| Sports | Predict match outcomes or player performance |
| Media / Entertainment | Movie recommendation system or review sentiment classifier |
| Agriculture | Crop yield prediction from weather data |
| Transport | Predict trip demand, or cluster travel patterns into rider segments |
| Retail / E-Commerce | Customer segmentation, or a product recommendation system |

---

### Where to Find Data

#### General repositories

| Source | Notes |
|---|---|
| [Kaggle Datasets](https://www.kaggle.com/datasets) | The widest catalogue, with notebooks showing what others did. Always check the **Licence** tab |
| [UCI Machine Learning Repository](https://archive.ics.uci.edu/) | Small, clean, classic tabular datasets — ideal for a first supervised project |
| [Google Dataset Search](https://datasetsearch.research.google.com/) | Search engine across thousands of data portals |
| [Hugging Face Datasets](https://huggingface.co/datasets) | Text, image, audio and a growing set of tabular data |
| [data.world](https://data.world/datasets/open-data) | Community-curated open datasets across many domains |
| [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets) | A large, well-organized GitHub list, grouped by domain |
| [scikit-learn built-in datasets](https://scikit-learn.org/stable/datasets.html) | Useful for practice, but **too small and over-used for a project on its own** |

#### Domain-specific

| Domain | Source |
|---|---|
| Health | [WHO Global Health Observatory](https://www.who.int/data/gho), [CDC Data](https://data.cdc.gov/), [MIMIC (clinical, requires registration)](https://physionet.org/) |
| Finance & Economics | [World Bank Open Data](https://data.worldbank.org/), [IMF Data](https://www.imf.org/en/Data), [Yahoo Finance via `yfinance`](https://pypi.org/project/yfinance/) |
| Recommenders | [MovieLens](https://grouplens.org/datasets/movielens/) (the standard choice), [Amazon Reviews](https://amazon-reviews-2023.github.io/), [Goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) |
| Sports | [football-data.co.uk](https://www.football-data.co.uk/data.php), [Kaggle sports datasets](https://www.kaggle.com/datasets?tags=13205-Sports), [ESPN cricinfo Statsguru](https://stats.espncricinfo.com/) |
| Education | [PISA international assessments](https://www.oecd.org/pisa/data/), [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) |
| Climate & Agriculture | [NASA Earthdata](https://www.earthdata.nasa.gov/), [FAOSTAT](https://www.fao.org/faostat/en/#data), [Open-Meteo API](https://open-meteo.com/) |
| Text / NLP | [Hugging Face](https://huggingface.co/datasets), [Kaggle NLP datasets](https://www.kaggle.com/datasets?tags=13204-NLP) |

#### Nepal and regional data

| Source | Notes |
|---|---|
| [Open Data Nepal](https://opendatanepal.com/) | Datasets on education, health, demographics and local government |
| [National Statistics Office (CBS)](https://censusnepal.cbs.gov.np/) | Census and national survey data |
| [Nepal Rastra Bank](https://www.nrb.org.np/) | Macroeconomic and banking statistics |
| [Humanitarian Data Exchange](https://data.humdata.org/group/npl) | Nepal-specific humanitarian, health and infrastructure data |

#### Collecting your own data

Perfectly acceptable, and often more interesting than a Kaggle download — a Google Forms survey, a
scraper, or a public API (sports scores, weather, currency rates). If you scrape, respect the site's
`robots.txt` and terms of service, and be gentle with request rates.

#### Three rules

1. **Check the licence** and cite the source in your README. If the licence forbids redistribution, link
   to it instead of committing the file.
2. **Freeze your data.** Download once and work from that copy. A dataset that changes underneath you
   makes every number in your report irreproducible. If you pull from an API, cache the response to CSV.
3. **Keep the repository small.** Commit files under ~50 MB; for anything larger, add the download
   instructions to `data/README.md` and git-ignore the raw folder.

---

### Core Requirements

| Requirement | Details |
|---|---|
| **Team Size** | <2 students |
| **Language** | Python (scikit-learn is sufficient — deep learning is optional and earns no extra marks by itself) |
| **Dataset** | Public dataset, API or scraped data. Source and licence must be documented |
| **Workflow** | Follow the ML workflow / CRISP-DM stages covered in Module 2 |
| **Code Structure** | Organized repository (see below), not one long notebook. Cookiecutter Data Science is a good reference |
| **Application** | A simple app — Streamlit, Gradio, Flask or FastAPI — that loads the trained model and makes predictions |
| **Containerization** | `Dockerfile` + `docker-compose.yml` (recommended, carries bonus marks) |
| **Version Control** | Git repository with regular commits throughout the project |

**Not accepted:** a notebook that only prints an accuracy score, or a copied tutorial (Titanic, iris,
MNIST) with variable names changed. Well-known datasets are fine *if you ask a new question of them*.

---

### Project Structure

```text
your-project/
├── README.md                 # problem, data source, results, how to run
├── requirements.txt
├── Dockerfile                # optional but recommended
├── docker-compose.yml        # optional but recommended
├── data/
│   ├── raw/                  # original downloaded data
│   └── processed/            # cleaned data used for training
├── notebooks/
│   ├── 01_eda.ipynb          # exploration and visualizations
│   └── 02_modeling.ipynb     # model comparison and evaluation
├── src/
│   ├── data.py               # loading and cleaning
│   ├── features.py           # preprocessing pipeline
│   ├── train.py              # trains and saves the model
│   └── predict.py            # loads the model and predicts
├── models/
│   └── model.joblib          # the trained model your app loads
└── app/
    └── app.py                # Streamlit / FastAPI entry point
```

**One rule that matters:** notebooks are for exploring and showing; `src/` holds anything the app
depends on. Never copy preprocessing code from a notebook into your app — import it from `src/`, or the
two will drift apart and your app will predict nonsense.

Save the **entire pipeline** (preprocessing + model) with `joblib`, not just the model. The scaler and
encoder are part of the model.

#### Adjust the structure to your track

| Track | Changes to the layout above |
|---|---|
| **A. Supervised** | Use it as-is |
| **B. Unsupervised / Recommender** | `src/train.py` → `src/cluster.py` or `src/recommend.py`; save cluster centroids or the item–item similarity matrix in `models/` |

#### Alternative structures you may explore

You are free to use a standard template instead of the layout above — say which one you used in your
README.

| Template | Notes |
|---|---|
| [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) | The most widely used DS project template. Install with `pip install cookiecutter-data-science`, then run `ccds`. More folders than you need, but industry standard |
| [Kedro](https://kedro.org/) | Framework built around reproducible data pipelines and config separation. Powerful, steeper learning curve |
| [PyScaffold + dsproject](https://github.com/pyscaffold/pyscaffoldext-dsproject) | Generates a packaged, `pip install`-able project with testing and CI set up |
| Your own | Perfectly acceptable if it separates data, exploration, source code, models and the app — and you explain it |

---

### Timeline and Deliverables (Weeks 4–8)

The project runs for five weeks. Weeks 4 and 8 are the same for both tracks; Weeks 5–7 are **split by
track** — find your track's row in each week's table.

---

#### **Week 4: Group Formation, Topic and Problem Formulation**

Common to both tracks:

- Form groups (2–4 students) and choose your track and domain.
- Identify who would use your system and what decision or action it supports.
- Secure your dataset now — do not start Week 5 still looking for data.

Your problem statement must answer these, according to your track:

| Track | Your problem statement defines… |
|---|---|
| **A. Supervised** | Unit of observation (what is one row?), target variable, features available *at prediction time*, and the evaluation metric with a justification |
| **B. Unsupervised** | What you are grouping or recommending, the features or the user–item interaction data, and how you will judge whether the result is any good |

**Deliverables (both tracks):**
- Idea presentation (3–5 slides): problem, track, data source, intended application.
- Problem statement document (1 page).

---

#### **Week 5: Data Understanding and Preparation**

| Track | What you do | Deliverable |
|---|---|---|
| **A. Supervised** | Descriptive statistics, distributions, correlations, class balance; handle missing values, outliers and duplicates; encoding and scaling inside a pipeline | EDA notebook with visualizations and written insights, plus a cleaned dataset |
| **B. Unsupervised** | Same cleaning work; check feature scales (critical for distance-based methods); for recommenders, build the user–item matrix and report sparsity and long-tail statistics | EDA notebook with visualizations, plus the prepared feature matrix or user–item matrix |

---

#### **Week 6: Baseline and Modeling**

Everyone establishes a baseline first. **You must beat it, and you must report it.**

| Track | What you do | Deliverable |
|---|---|---|
| **A. Supervised** | Baseline (majority class or mean), then train at least two model families; cross-validation for tuning | Results table: baseline vs. models, on your chosen metric |
| **B. Unsupervised** | Clustering: sweep *k*, use elbow/silhouette to choose it. Recommender: build content-based or item-item collaborative filtering against a popularity baseline | Results table (silhouette scores or precision@10 / hit-rate vs. baseline) |

---

#### **Week 7: Evaluation, Code Structuring and Application**

Common to both tracks:

- Refactor notebook code into the `src/` structure — `python src/train.py` should run end to end and
  save your model artifact to `models/`.
- Build a simple application that loads that artifact.
- Containerize with Docker and Docker Compose (recommended, bonus marks).

| Track | Evaluation focus | What the app does |
|---|---|---|
| **A. Supervised** | Confusion matrix or residual plots, per-class performance, error analysis: where does it fail? | Takes feature inputs, returns a prediction **with its probability** |
| **B. Unsupervised** | Interpret and *name* your clusters, or report offline ranking metrics on held-out interactions | Shows a user's segment with its profile, or returns a top-10 ranked list with reasons |

**Deliverables (both tracks):** Organized repository, reproducible training script, saved model
artifact, locally running application, evaluation results, and a report draft.

---

#### **Week 8: Final Presentation and Submission**

- 10-minute presentation: problem, data, modeling, **live demo**, limitations.
- Peer evaluation of at least two other projects.

**Deliverables (both tracks):** Final report (4–6 pages), presentation slides, GitHub repository link

---

### Final Report Structure (4–6 pages)

1. **Problem Definition** — the problem, who it is for, your track, and the success metric with justification.
2. **Data** — source, licence, size and key characteristics.
3. **EDA and Preprocessing** — key insights and the cleaning decisions you made, with reasons.
4. **Modeling** — baseline, models compared, tuning approach.
5. **Results and Evaluation** — metrics table, error analysis, what your model gets wrong.
6. **Application** — screenshots, how it works, how to run it.
7. **Limitations and Reflection** — what you would do differently, ethical considerations.

---

### Evaluation Criteria

| Component | Weight | Key Points |
|---|---:|---|
| **Problem Definition and Dataset** | 10% | Clear problem statement, appropriate target and metric, documented data source |
| **EDA and Preprocessing** | 20% | Meaningful insights, visualizations, correct handling of missing values, outliers and encoding |
| **Modeling and Evaluation** | 30% | Baseline established and beaten, ≥2 models or configurations compared, honest validation, appropriate metrics, error analysis |
| **Code Structure and Reproducibility** | 10% | Organized repository, working `train.py`, clear commit history |
| **Application and Demo** | 20% | Working app that loads the model and makes predictions; clear demonstration |
| **Report and Presentation** | 10% | Complete report, clear 10-minute presentation, honest discussion of limitations |
| **Total** | **100%** | |

**Bonus (up to 5%):** Docker and Docker Compose setup that runs from a clean clone, a deployed public
URL, model explainability in the app, or unit tests.

---

### Recommended Tools

| Purpose | Recommended |
|---|---|
| Data handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Modeling | scikit-learn |
| Model saving | joblib |
| Application | Streamlit (easiest), Gradio, FastAPI, Flask |
| Containerization | Docker, Docker Compose |
| Version control | Git, GitHub |
| Datasets | See the **Where to Find Data** section above |

### Getting Started

```bash
# Create your repository and environment
git init my-project && cd my-project
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib

# After training and saving your model
streamlit run app/app.py            # open http://localhost:8501

# If containerized
docker compose up --build
```

### Academic Integrity

You may use libraries, documentation and AI tools as references, but you must be able to explain every
line of your code and every modeling decision. Cite your data sources, and note in your report if you
used an AI assistant and how.
