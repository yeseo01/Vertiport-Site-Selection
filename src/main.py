"""Run the vertiport site-selection pipeline."""

from pathlib import Path

import pandas as pd

from .analysis import (
    elbow_analysis,
    silhouette_analysis,
    silhouette_diagram,
)
from .kmeans import KMeans


def main():
    project_root = Path(__file__).resolve().parents[1]

    data_path = project_root / "data" / "processed" / "VertiportProcessed_Final.csv"
    boundary_path = (
        project_root
        / "data"
        / "raw"
        / "Project_1_data_South_Korea_territory.csv"
    )

    data = pd.read_csv(
        data_path,
        usecols=[
            "Longitude",
            "Latitude",
            "NormComm",
            "NormTrf",
            "Penalty",
        ],
    )
    data_points = data[
        ["Longitude", "Latitude"]
    ].values.tolist()

    weights = (
        data["NormComm"]
        + data["NormTrf"]
        + data["Penalty"]
    )

    n_clusters = 17
    max_iter = 100
    k_max = 20

    model = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
    )
    model.fit(data_points, weights)

    model.plot(boundary_path)

    elbow_analysis(
        data_points=data_points,
        k_max=k_max,
        max_iter=max_iter,
        weights=weights,
    )

    silhouette_analysis(
        data_points=data_points,
        k_max=k_max,
        max_iter=max_iter,
        weights=weights,
    )

    silhouette_diagram(
        data_points=data_points,
        n_clusters=n_clusters,
        max_iter=max_iter,
        weights=weights,
    )

if __name__ == "__main__":
    main()
