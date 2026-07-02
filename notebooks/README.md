# DCS 404: Data Science and Machine Learning — Notebooks

A complete set of 18 teaching notebooks covering the full course syllabus.
Each notebook contains conceptual explanations, runnable code, visualisations, exercises, and reflection questions.

---

## Notebook Order and Topics

| # | File | Topic | Key Libraries |
|---|------|-------|---------------|
| 1 | `01_ai_fundamentals.ipynb` | AI overview, BFS, DFS, A\*, Minimax, Alpha-Beta | `collections`, `heapq` |
| 2 | `02_numpy_essentials.ipynb` | Arrays, broadcasting, linear algebra | `numpy` |
| 3 | `03_pandas_essentials.ipynb` | Series, DataFrames, groupby, merge | `pandas` |
| 4 | `04_data_visualization.ipynb` | Line, scatter, bar, histogram, heatmap | `matplotlib` |
| 5 | `05_data_preparation.ipynb` | Missing data, outliers, encoding, scaling, pipelines | `sklearn`, `pandas` |
| 6 | `06_linear_regression.ipynb` | Simple/multiple LR, gradient descent, Ridge, Lasso | `sklearn`, `numpy` |
| 7 | `07_logistic_regression.ipynb` | Sigmoid, binary/multiclass, ROC-AUC | `sklearn` |
| 8 | `08_svm.ipynb` | Maximum margin, kernel trick, C and gamma tuning | `sklearn` |
| 9 | `09_knn_naive_bayes.ipynb` | KNN, curse of dimensionality, Gaussian/Multinomial NB | `sklearn` |
| 10 | `10_decision_trees.ipynb` | Gini, information gain, tree visualisation | `sklearn` |
| 11 | `11_ensemble_methods.ipynb` | Random Forest, AdaBoost, Gradient Boosting | `sklearn` |
| 12 | `12_text_classification.ipynb` | Bag-of-Words, TF-IDF, 20 Newsgroups | `sklearn` |
| 13 | `13_model_evaluation.ipynb` | CV, confusion matrix, ROC-AUC, PR curve, GridSearchCV | `sklearn` |
| 14 | `14_clustering.ipynb` | K-Means, hierarchical, DBSCAN, silhouette | `sklearn`, `scipy` |
| 15 | `15_deep_learning_foundations.ipynb` | Perceptron, activation functions, backpropagation | `numpy` |
| 16 | `16_pytorch_basics.ipynb` | Tensors, autograd, nn.Module, training loop | `torch` |
| 17 | `17_cnn_rnn_lstm.ipynb` | CNN on MNIST, RNN, LSTM for sequences | `torch`, `torchvision` |
| 18 | `18_stock_prediction_lstm.ipynb` | Time series, LSTM forecasting, evaluation | `torch`, `sklearn` |

---

## Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn torch torchvision scipy networkx
```

> For GPU acceleration install the CUDA-enabled PyTorch wheel from https://pytorch.org

---

## Running the Notebooks

### Local (Jupyter)
```bash
pip install jupyterlab
jupyter lab notebooks/
```

### Google Colab
Upload individual `.ipynb` files to [colab.research.google.com](https://colab.research.google.com) or open them from GitHub.

---

## Course Structure

```
Module 1  — AI Foundations          Notebooks 01
Module 2  — Python for Data Science Notebooks 02–04
Module 3  — Data Preparation        Notebook  05
Module 4  — Regression              Notebook  06
Module 5  — Classification          Notebooks 07–11
Module 6  — Text Classification     Notebook  12
Module 7  — Model Evaluation        Notebook  13
Module 8  — Clustering              Notebook  14
Module 9  — Deep Learning           Notebooks 15–18
```

---

## Learning Outcomes

After completing all notebooks, students will be able to:

- Implement and compare classical search algorithms (BFS, DFS, A\*, Minimax)
- Wrangle tabular data with Pandas and visualise findings with Matplotlib
- Build end-to-end ML pipelines including preprocessing, modelling, and evaluation
- Apply and tune seven classification algorithms (logistic regression, SVM, KNN, Naive Bayes, decision trees, random forest, gradient boosting)
- Classify text documents using TF-IDF features
- Diagnose and fix underfitting / overfitting using learning curves and cross-validation
- Cluster unlabelled data and evaluate cluster quality
- Build and train deep neural networks (MLP, CNN, LSTM) in PyTorch
- Apply an LSTM to time series forecasting and critically assess its limitations

---

## Notebook Design Principles

Each notebook follows this structure:

1. **Learning Objectives** — clear list of what you will be able to do
2. **Conceptual explanation** — intuition before formulas
3. **Mathematical intuition** — key equations explained step by step
4. **Runnable code** — complete, self-contained examples
5. **Visualisations** — plots to build geometric intuition
6. **Mini exercises** — hands-on tasks for deeper understanding
7. **Reflection questions** — encourage critical thinking beyond the code

---

*DCS 404 — Data Science and Machine Learning*
