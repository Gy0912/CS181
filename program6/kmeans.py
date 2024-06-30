import numpy as np
from typing import List, Tuple

import copy, random
random.seed(1024) # for reproducibility. Do not modify this

def _read_datas(path: str = "./iris/iris.data"):
    with open(path, "r") as f:
        lines = f.readlines()

        datas = np.zeros((150, 4))
        labels = []

        for idx, line in enumerate(lines):
            line_list = line.split(",")
            if len(line_list) == 1: continue # get rid of \n line

            datas[idx] = line_list[:4]
            labels.append(line_list[4])
    print(f"datas has a shape of {datas.shape}")
    return datas, labels

def kmeans_step(
        data: np.ndarray,
        current_center: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, float]:
    """
    The K-Means step function.
    Here you need to implement both E step and M step according to what we learned in class.

    Input:
    - data: a np.ndarray with a shape of [150, 4] containing all data samples in the iris dataset;
    - current_center: a [k] array containg all centers.
    - assign_to: a [150] array, each contains a `int` value range from 0 to `k-1`.
        data[i] is currently assigned to cluster assign_to[i].
    - k: a int value represents the number of clusters.

    Output:
    - new_center: a np.ndarray with a shape of [k] containing all updated centers.
    - sse: a float value represents the sse for current states
    """
    
    new_center = copy.deepcopy(current_center)
    assign_to = np.empty(150) # data[i] is currently assigned to cluster assign_to[i].

    # remove this line before you start
    "*** YOUR CODE HERE ***"
    for i in range(data.shape[0]):
        distances = np.linalg.norm(data[i] - current_center, axis=1)
        assign_to[i] = np.argmin(distances)
    for j in range(k):
        cluster_points = data[assign_to == j]
        if len(cluster_points) > 0:
            new_center[j] = np.mean(cluster_points, axis=0)
    sse = 0.0
    for j in range(k):
        cluster_points = data[assign_to == j]
        sse += np.sum((cluster_points - new_center[j])**2)

    return new_center, sse


def kmeans(data: np.ndarray, k: int) -> float:
    """
    The K-Means main function.
    Input:
    - data: a np.ndarray containing all data samples in the iris dataset;
    - k: a int value represents the number of clusters.
    Output:
    - sse: the sum of squared errors in current state
    """
    # randomly choose k data points as center
    rand_idxs = random.sample(range(data.shape[0]), k=k)
    centers = data[rand_idxs]

    while True:
        new_centers, sse = kmeans_step(data, centers, k)
        if np.sum((centers - new_centers)**2) > 0.001: # not converge
            centers = new_centers
        else:
            print(f"converge. k = {N}, SSE: {sse}")
            return sse


def plot(all_sse: List[float]):
    """
    use matplotlib to plot the SSE-#clusters figure.
    input: all_sse: A list contains all SSE for different number of clusters.
        where `all_sse[i]` represents the sse value for doing K-Means on `i+1` clusters.
    """
    from matplotlib import pyplot as plt
    plt.plot(range(1, len(all_sse) + 1), all_sse, "-go")
    plt.show()

if __name__ == "__main__":
    """ Entrance of the code. """
    
    data, _ = _read_datas() # We don't need to use lables in K-Means.
    all_sse = [] # A list to store all SSE for different numebr of clusters.

    for N in range(1, 9):
        converge = False
        all_sse.append( kmeans(data, N) )
    
    plot(all_sse)
