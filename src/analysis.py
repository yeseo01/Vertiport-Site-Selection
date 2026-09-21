"""Analysis utilities for weighted K-Means clustering."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    silhouette_samples,
    silhouette_score,
)

from .kmeans import KMeans


def elbow_analysis(data_points, k_max, max_iter, weights):
    weighted_inertia = []

    for n_clusters in range(1, k_max + 1):
        model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
        )
        model.fit(data_points, weights)

        inertia = 0.0

        for cluster_index, (cluster, cluster_weights) in enumerate(
            zip(model.cluster_list, model.cluster_weights)
        ):
            centroid = model.centroids[cluster_index]

            for point, point_weight in zip(cluster, cluster_weights):
                distance_squared = (
                    (point[0] - centroid[0]) ** 2
                    + (point[1] - centroid[1]) ** 2
                )
                inertia += point_weight * distance_squared

        weighted_inertia.append(inertia)

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, k_max + 1),
        weighted_inertia,
        marker="o",
    )
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Weighted Inertia")
    plt.title("Elbow Method for Weighted K-Means")
    plt.grid(True)
    plt.show()


def silhouette_analysis(data_points, k_max, max_iter, weights):
    silhouette_scores = []

    for n_clusters in range(2, k_max + 1):
        model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
        )
        model.fit(data_points, weights)

        labels = model.labels_
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            score = -1.0
        else:
            score = silhouette_score(
                np.array(data_points),
                labels,
            )

        silhouette_scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(2, k_max + 1),
        silhouette_scores,
        marker="s",
    )
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")
    plt.grid(True)
    plt.show()


def silhouette_diagram(
    data_points,
    n_clusters,
    max_iter,
    weights,
):
    model = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
    )
    model.fit(data_points, weights)

    labels = model.labels_
    silhouette_values = silhouette_samples(
        np.array(data_points),
        labels,
    )

    fig, ax = plt.subplots(figsize=(7, 9))
    y_lower = 10

    for cluster_index in range(n_clusters):
        cluster_values = silhouette_values[labels == cluster_index]
        cluster_values.sort()

        cluster_size = len(cluster_values)
        y_upper = y_lower + cluster_size

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_values,
        )

        ax.text(
            -0.05,
            y_lower + 0.5 * cluster_size,
            str(cluster_index),
        )

        y_lower = y_upper + 10

    ax.set_title(f"Silhouette Diagram (K = {n_clusters})")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster Label")

    ax.axvline(
        np.mean(silhouette_values),
        linestyle="--",
        label="Average silhouette score",
    )

    ax.legend()
    ax.grid(True)
    plt.show()
