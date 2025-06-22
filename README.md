# Vertiport Site Selection

A clustering-based approach to selecting optimal vertiport locations across South Korea, supporting the national roadmap for Regional Air Mobility (RAM).

> 🚀 **Note:** This project implements the K-Means algorithm from scratch using only NumPy. No external machine learning libraries were used, to ensure full control and understanding of the algorithm.

---

## 📌 Project Overview

As part of the government’s effort to introduce Regional Air Mobility (RAM), this project aims to:
1. Select 17 optimal vertiport locations from candidate sites using K-Means clustering
   
2. Explore how many vertiports would be appropriate if there were no budget constraints

---

## 📁 Dataset

- **File**: `Vertiport_candidates.csv`
  
- **Description**: Contains geographic coordinates (latitude, longitude) of candidate vertiport sites in South Korea

---

## 🧠 Implementation Details

- **Algorithm**: K-Means (manual implementation in NumPy)
  
- **Distance Metric**: Euclidean
  
- **Initialization**: Random or custom seed
  
- **Convergence Criteria**: Stable cluster assignments

---

## 📊 Evaluation

- Visualization of cluster centroids on the map of South Korea
  
- Elbow method to estimate optimal number of vertiports
  
- Analysis of trade-offs between cost (number of sites) and coverage efficiency

---

## 🚀 How to Run

```bash
python src/main.py
