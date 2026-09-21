"""Weighted K-Means implementation for vertiport site selection."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
