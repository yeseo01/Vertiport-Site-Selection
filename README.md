# Vertiport Site Selection with Weighted K-Means

A course team project that applies weighted K-Means clustering to a Regional Air Mobility (RAM) vertiport siting scenario in South Korea.

This repository focuses on the executable clustering pipeline for selecting **17 representative vertiport sites** from a set of candidate locations. The value `K = 17` was specified by the project assignment rather than selected by the clustering algorithm.

## Project Scope

The original assignment included two related tasks:

1. Select 17 vertiport locations from the provided candidate sites.
2. Investigate an appropriate number of vertiports under an unconstrained-budget scenario.

This repository primarily reproduces the first task. The original team also used Elbow and Silhouette analyses as diagnostics for the second task and supplemented them with demand- and capacity-based reasoning. That demand/capacity analysis is not fully reproduced in the current codebase.

## My Contribution

This was a team project.

My primary contribution was implementing the core K-Means algorithm from scratch with NumPy and extending it to support weighted centroid updates for the vertiport analysis.

I also contributed to the weighting strategy, including the use of traffic congestion and inter-regional commuter demand as input signals. Broader preprocessing and site-selection decisions were developed collaboratively by the team.

## Method

### Weighted K-Means

The custom weighted K-Means implementation is in `src/kmeans.py`.

For each data point, cluster assignment uses squared Euclidean distance to select the nearest centroid.

The centroid of each cluster is then updated using a weighted mean:

$$
\mu_k =
\frac{\sum_{i \in C_k} w_i x_i}
{\sum_{i \in C_k} w_i}
$$

The clustering weight used by the current implementation is:

$$
w_i =
\text{NormComm}_i +
\text{NormTrf}_i +
\text{Penalty}_i
$$

where the processed dataset contains the normalized project-specific input signals.

The implementation also includes:

- deterministic initialization through `random_state=10`
- a maximum of 100 iterations by default
- convergence based on centroid movement
- explicit handling of empty clusters
- final reassignment of points using the converged centroids

### Representative Vertiport Selection

K-Means centroids are not necessarily valid candidate sites.

After clustering, each centroid is mapped to the nearest available candidate point **within its assigned cluster**. These candidate points are reported as the representative vertiport sites.

### Diagnostics

The repository includes:

- weighted inertia for Elbow analysis
- standard geometric Silhouette scores
- Silhouette diagrams
- a separate unweighted scikit-learn K-Means baseline for comparison

The custom K-Means algorithm itself does **not** use scikit-learn. Scikit-learn is used only for Silhouette metrics and the comparison baseline.

## Data

The assignment provided 7,774 candidate locations.

The archived processed dataset used by the clustering pipeline contains 766 candidate locations after the original team preprocessing. The processed coordinates correspond to the provided candidate locations up to stored coordinate precision and contain no duplicate coordinates.

```text
data/
├── raw/
│   ├── Project_1_data_South_Korea_territory.csv
│   └── Project_1_data_Vertiport_candidates.csv
└── processed/
    └── VertiportProcessed_Final.csv
```

`Project_1_data_South_Korea_territory.csv` is used for map visualization.

`Project_1_data_Vertiport_candidates.csv` contains the original candidate coordinates supplied for the assignment.

`VertiportProcessed_Final.csv` is the archived processed input used by the weighted clustering analysis.

Some legacy text metadata in the processed CSV has encoding damage. The executable pipeline therefore reads only the numeric columns required for clustering.

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── kmeans.py
│   ├── analysis.py
│   ├── main.py
│   └── reference_sklearn.py
├── Pipfile
├── Pipfile.lock
└── README.md
```

- `src/kmeans.py` — custom weighted K-Means implementation and representative-site selection
- `src/analysis.py` — Elbow, Silhouette-score, and Silhouette-diagram diagnostics
- `src/main.py` — data loading and pipeline orchestration
- `src/reference_sklearn.py` — unweighted scikit-learn K-Means comparison baseline

## How to Run

The project uses Python 3.13 and Pipenv.

Install the locked dependencies:

```bash
pipenv sync
```

Run the main weighted clustering pipeline:

```bash
pipenv run python -m src.main
```

Run the scikit-learn comparison baseline:

```bash
pipenv run python src/reference_sklearn.py
```

The main script prints the 17 selected representative candidate coordinates and displays the clustering and diagnostic plots.

## Reproducibility

The custom K-Means model uses an explicit `random_state` rather than relying on NumPy's global random state.

With `random_state=10`, the cleaned implementation reproduces the representative sites generated by the original project implementation for `K = 17`.

## Limitations

The project preserves the methodology used in the original course work.

In particular:

- longitude and latitude are treated directly as Euclidean coordinates rather than projected geographic coordinates
- the archived processed dataset is included, but the complete original preprocessing pipeline is not reconstructed in this repository
- the current code reproduces the clustering-based site-selection portion of the project, not the complete demand/capacity analysis used in the team's unconstrained-budget discussion

These limitations are retained rather than retroactively changing the historical project methodology.
