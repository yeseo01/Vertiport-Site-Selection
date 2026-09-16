from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples

class KMeans:
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = 10,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _assign_clusters(self, data_points, centroids, weights):
        cluster_list = [[] for _ in range(self.n_clusters)]
        cluster_weights = [[] for _ in range(self.n_clusters)]
        labels = np.empty(len(data_points), dtype=int)

        for i, point in enumerate(data_points):
            distances = []

            for centroid in centroids:
                x_diff = centroid[0] - point[0]
                y_diff = centroid[1] - point[1]
                distance_squared = x_diff**2 + y_diff**2
                distances.append(distance_squared)

            nearest_cluster = distances.index(min(distances))

            cluster_list[nearest_cluster].append(point)
            cluster_weights[nearest_cluster].append(weights[i])
            labels[i] = nearest_cluster

        return cluster_list, cluster_weights, labels

    def _update_centroids(
        self,
        cluster_list,
        cluster_weights,
        previous_centroids,
    ):
        centroids = []

        for i, cluster in enumerate(cluster_list):
            if len(cluster) == 0:
                centroids.append(previous_centroids[i])
                continue

            x_coords = np.array([point[0] for point in cluster])
            y_coords = np.array([point[1] for point in cluster])
            weights = np.array(cluster_weights[i])

            weighted_x = np.sum(weights * x_coords) / np.sum(weights)
            weighted_y = np.sum(weights * y_coords) / np.sum(weights)

            centroids.append((weighted_x, weighted_y))

        return centroids

    def fit(self, data_points, weights):
        # Randomly initialize centroids from the input data points.
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(
            len(data_points),
            self.n_clusters,
            replace=False,
        )
        self.centroids = [data_points[i] for i in indices]

        for _ in range(self.max_iter):
            cluster_list, cluster_weights, _ = self._assign_clusters(
                data_points,
                self.centroids,
                weights,
            )

            new_centroids = self._update_centroids(
                cluster_list,
                cluster_weights,
                self.centroids,
            )

            movement = np.linalg.norm(
                np.array(self.centroids) - np.array(new_centroids)
            )

            # Always keep the newly computed centroids.
            self.centroids = new_centroids

            if movement < self.tol:
                break

        # Reassign points once using the final centroids so that
        # stored clusters and centroids represent the same final state.
        (
            self.cluster_list,
            self.cluster_weights,
            self.labels_,
        ) = self._assign_clusters(
            data_points,
            self.centroids,
            weights,
        )

    def _select_representative_sites(self):
        """Return the candidate point nearest to each centroid within its cluster."""
        representative_sites = []

        for cluster, centroid in zip(self.cluster_list, self.centroids):
            if not cluster:
                representative_sites.append(None)
                continue

            nearest_point = min(
                cluster,
                key=lambda point: (
                    (point[0] - centroid[0]) ** 2
                    + (point[1] - centroid[1]) ** 2
                ),
            )
            representative_sites.append(nearest_point)

        return representative_sites

    def plot(self, boundary_path):
        boundary_df = pd.read_csv(boundary_path)

        boundary_longitudes = boundary_df["Longitude (deg)"]
        boundary_latitudes = boundary_df["Latitude (deg)"]

        plt.figure(figsize=(5, 6))
        color_map = plt.get_cmap("tab20")

        plt.plot(
            boundary_longitudes,
            boundary_latitudes,
            "k-",
            linewidth=1,
        )

        for cluster_index, cluster in enumerate(self.cluster_list):
            longitudes = [point[0] for point in cluster]
            latitudes = [point[1] for point in cluster]

            plt.scatter(
                longitudes,
                latitudes,
                c=[color_map(cluster_index)],
                marker=".",
                s=80,
            )

        representative_sites = self._select_representative_sites()

        for cluster_index, site in enumerate(representative_sites):
            if site is None:
                continue

            plt.scatter(
                site[0],
                site[1],
                c="k",
                marker="o",
                s=100,
                edgecolors="white",
            )

            print(
                f"Cluster {cluster_index + 1} - "
                f"Representative site: "
                f"({site[0]:.4f}, {site[1]:.4f})"
            )

        plt.title("Weighted K-Means Clustering")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.axis("equal")
        plt.grid(True)
        plt.show()


# Elbow Method
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


# Silhouette Score
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


# Silhouette Diagram
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
        cluster_values = silhouette_values[
            labels == cluster_index
        ]
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

    ax.set_title(
        f"Silhouette Diagram (K = {n_clusters})"
    )
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


if __name__ == '__main__':
    main()
