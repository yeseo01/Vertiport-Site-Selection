from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


def main():
    """Run an unweighted scikit-learn K-Means baseline for comparison."""
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "processed" / "VertiportProcessed_Final.csv"

    data = pd.read_csv(
        data_path,
        usecols=["Longitude", "Latitude"],
    )
    data_points = data[["Longitude", "Latitude"]]

    model = KMeans(
        n_clusters=17,
        random_state=10,
        n_init=10,
    )
    labels = model.fit_predict(data_points)
    centroids = model.cluster_centers_

    plt.scatter(
        data_points["Longitude"],
        data_points["Latitude"],
        c=labels,
        cmap="viridis",
    )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        s=200,
        alpha=0.75,
    )

    plt.title("Unweighted scikit-learn K-Means Baseline")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
