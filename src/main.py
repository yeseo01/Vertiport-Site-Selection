import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples

np.random.seed(10)  # 실행할 때마다 같은 결과를 얻기 위한 시드 고정

# KMeans클래스
class KMeans:
    def __init__(self, K, iteration):
        self.K = K
        self.iteration = iteration
        self.epsilon = 1e-4 # 클러스터 중심이 더이상 움직이지 않음을 판단하기 위한 상수

    def Cluster(self, data_point, Centroid, weight):
        # K개의 클러스터를 담을 리스트 생성
        self.cluster_list = [[] for _ in range(self.K)]
        # 각 클러스터에 속한 가중치 저장
        self.cluster_weight = [[] for _ in range(self.K)]  

        for i, point in enumerate(data_point):
            distance_list = []

            # 각 중심에 대한 거리 계산
            for centroid in Centroid:
                x_d = centroid[0] - point[0]
                y_d = centroid[1] - point[1]
                d = x_d**2 + y_d**2 # 거리제곱

                # 가중치를 반영한 거리
                d_weighted = d * weight[i]
                distance_list.append(d_weighted)
            
            # 최소거리를 갖는 클러스터에 분류  
            min_d_index = distance_list.index(min(distance_list))
            self.cluster_list[min_d_index].append(point)
            self.cluster_weight[min_d_index].append(weight[i])

        return self.cluster_list, self.cluster_weight

    def find_centroid(self, cluster_list, cluster_weight, prev_centroid):
        Centroid = []
    
        for i, cluster in enumerate(cluster_list):
            # 빈 클러스터가 있는 경우 이전 중심을 사용
            if len(cluster) == 0:
                Centroid.append(prev_centroid[i])
                continue

            # 클러스터 점들의 좌표와 가중치 추출    
            x_coords = np.array([p[0] for p in cluster])
            y_coords = np.array([p[1] for p in cluster])
            weights = np.array(cluster_weight[i])

            # 가중평균으로 좌표 평균 계산
            avg_x = np.sum(weights * x_coords) / np.sum(weights)
            avg_y = np.sum(weights * y_coords) / np.sum(weights)
            Centroid.append((avg_x, avg_y))

        return Centroid

    def fit(self, data_point, weight):
        # 중심점 랜덤 초기화: Point 중에서 무작위로 K개 선택
        indices = np.random.choice(len(data_point), self.K, replace=False)
        self.Centroid = [data_point[i] for i in indices]
        
        for n in range(self.iteration):
            # 클러스터링
            cluster_list, cluster_weight = self.Cluster(data_point, self.Centroid, weight)
            # 새로운 중심 찾기
            Centroid_new = self.find_centroid(cluster_list, cluster_weight, self.Centroid)
            
            # 수렴 조건: 중심 변화량이 epsilon보다 작으면 종료
            movement = np.linalg.norm(np.array(self.Centroid) - np.array(Centroid_new))
            if movement < self.epsilon:
                break

            self.Centroid = Centroid_new
        self.cluster_list = cluster_list

    def plot(self):
        # 대한민국 경계선 데이터 로드
        boundary_df = pd.read_csv('/Users/yeseo/Desktop/항공우주AI기초/Project_1/project1_3/Project_1_data_South_Korea_territory.csv')

        lonB = boundary_df['Longitude (deg)']
        latB = boundary_df['Latitude (deg)']

        fig = plt.figure(figsize=(5, 6))
        color_map = plt.get_cmap('tab20')

        # 대한민국 경계선 그리기
        plt.plot(lonB, latB, 'k-', linewidth=1)

        # 데이터 포인트 플랏
        for index, cluster in enumerate(self.cluster_list):
            x = [coord[0] for coord in cluster]
            y = [coord[1] for coord in cluster]
            plt.scatter(x, y, c=[color_map(index)], marker='.', s=80)

        # 각 클러스터의 중심에서 가장 가까운 실제 점 찾기
        nearest_points = []
        for i, centroid in enumerate(self.Centroid):
            if len(self.cluster_list[i]) > 0:  # 클러스터가 비어있지 않은 경우에만
                # 클러스터 내의 모든 점과 중심점 사이의 거리 계산
                distances = []
                for point in self.cluster_list[i]:
                    dx = point[0] - centroid[0]
                    dy = point[1] - centroid[1]
                    distance = dx**2 + dy**2
                    distances.append((distance, point))
                
                # 가장 가까운 점 찾기
                nearest_point = min(distances, key=lambda x: x[0])[1]
                nearest_points.append(nearest_point)
                
                # 가장 가까운 점 플롯
                plt.scatter(nearest_point[0], nearest_point[1], c='k', marker='o', s=100, edgecolors='white')
                
                # 터미널에 출력
                print(f"Cluster {i+1} - Nearest point to centroid: ({nearest_point[0]:.4f}, {nearest_point[1]:.4f})")

        plt.title("Weighted KMeans Clustering")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.axis('equal')
        plt.grid(True)
        plt.show()

# Elbow Method
def elbow_analysis(data_point, K_max, iteration, weight):
    SSE_list = []
    
    for K in range(1, K_max + 1):
        # KMeans 실행
        model = KMeans(K, iteration)
        model.fit(data_point, weight)

        # SSE 계산
        sse = 0
        for i in range(K):
            for p in model.cluster_list[i]:
                dx = p[0] - model.Centroid[i][0]
                dy = p[1] - model.Centroid[i][1]
                sse += dx**2 + dy**2
        SSE_list.append(sse)

    # 그래프 출력
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, K_max + 1), SSE_list, marker='o')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal K")
    plt.grid(True)
    plt.show()

# Silhouette Score
def silhouette_analysis(data_point, K_max, iteration, weight):
    # 각 K에 대한 실루엣 점수를 저장할 리스트
    silhouette_scores = []

    # K는 2부터 시작 (K=1일 때는 실루엣 점수를 정의할 수 없음)
    for K in range(2, K_max + 1):
        # KMeans 실행
        model = KMeans(K, iteration)
        model.fit(data_point, weight)

        # 실루엣 점수 계산을 위한 클러스터 라벨 생성
        labels = np.zeros(len(data_point), dtype=int)
        for cluster_idx, cluster in enumerate(model.cluster_list):
            for p in cluster:
                idx = data_point.index(p)
                labels[idx] = cluster_idx
        try:
            # 실루엣 점수 계산
            score = silhouette_score(np.array(data_point), labels)
        except:
            # 클러스터 중 하나라도 비어있으면 예외 발생 → 점수 -1 처리
            score = -1
        # 해당 K에 대한 실루엣 점수 저장
        silhouette_scores.append(score)

    # 결과 그래프 출력
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, K_max + 1), silhouette_scores, marker='s')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis for Optimal K")
    plt.grid(True)
    plt.show()

# Silhouette Diagram
def silhouette_diagram(data_point, K, iteration, weight):
    # KMeans 실행
    model = KMeans(K, iteration)
    model.fit(data_point, weight)

    # 실루엣 점수 계산을 위한 클러스터 라벨 생성
    labels = np.zeros(len(data_point), dtype=int)
    for cluster_idx, cluster in enumerate(model.cluster_list):
        for p in cluster:
            idx = data_point.index(p)
            labels[idx] = cluster_idx
    
    # 실루엣 점수 계산
    silhouette_vals = silhouette_samples(np.array(data_point), labels)

    # Silhouette 다이어그램 출력
    fig, ax = plt.subplots(figsize=(7, 9))
    y_lower = 10
    for i in range(K):
        ith_vals = silhouette_vals[labels == i]
        ith_vals.sort()
       
        size_i = len(ith_vals)
        y_upper = y_lower + size_i

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals)
        ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10

    ax.set_title(f"Silhouette Diagram (K = {K})")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster Label")
    ax.axvline(np.mean(silhouette_vals), color="red", linestyle="--", label="Avg silhouette score")
    ax.legend()
    ax.grid(True)
    plt.show()

# main 함수
def main():
    # 데이터 로드
    data = pd.read_csv('/Users/yeseo/Desktop/항공우주AI기초/Project_1/project1_3/VertiportProcessed_Final.csv')
    data_point = data[['Longitude', 'Latitude']].values.tolist()
    weight_raw = data['weight'].replace(0, np.finfo(float).eps) # 0으로 나누는 것을 방지
    weight = 1 / (weight_raw/3)

    # 하이퍼파라미터
    K = 17
    iteration = 100
    K_max = 20

    # KMeans 실행
    model = KMeans(K, iteration)
    model.fit(data_point, weight)

    # 클러스터링 결과 출력
    model.plot()

    # Elbow 분석 실행
    elbow_analysis(data_point, K_max, iteration, weight)
    # Silhoutte_score
    silhouette_analysis(data_point, K_max, iteration, weight)
    # SIlhoutte_diagram
    silhouette_diagram(data_point, K, iteration, weight)


# 실행
if __name__ == '__main__':
    main()
