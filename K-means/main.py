import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import random

# Load dataset
dataset_path = 'champions_info.csv'
df = pd.read_csv(dataset_path)

# 1. Preprocessing
# a. Dataset description
# The dataset contains attributes related to champions in a league game, including their performance metrics
# and characteristics like champion name, league, game result, kills, deaths, assists, side, damage to champions,
# difficulty, items for spike, attack type, and positions played.

# b. Data cleaning
df = df.dropna()

# c. Mean and variance for numerical attributes
numerical_attributes = ['result', 'kills', 'deaths', 'assists']
means = df[numerical_attributes].mean()
variances = df[numerical_attributes].var()

print('Mean for numerical attributes: \n', means)
print('Variance for numerical attributes: \n', variances)

# d. Remove the target attribute in your dataset ???
dataset_final = df.drop(columns=['result', 'damagetochampions'])
dataset_numerical = df[numerical_attributes]
# 2. Distances
# a. Convert the discrete attributes that are not numeric (such as strings or boolean values) into
# numerical. If this doesn't apply to your dataset, provide a short explanation on how you would proceed
non_numerical_attribues = ['champion', 'league', 'side', 'Difficulty', 'Items_For_Spike', 'Attack_Type', 'side',
                           'top', 'mid', 'jng', 'sup', 'bot', 'first_blood_kill']

label_encoder = LabelEncoder()

# Apply label encoder to each non_numerical attribute
for col in non_numerical_attribues:
    dataset_final[col] = label_encoder.fit_transform(dataset_final[col])

print('Dataset after encoding: \n', dataset_final)
# print columns
print('Columns: \n', dataset_final.columns)


# b. Write a function distance_points that calculates the distance between two points. The function should take three
# parameters: the two points and p , where p indicates the order of the Minkowski distance (remember that p=1 is the
# equivalent for the Manhattan distance, and p=2 for the Euclidean one).

def distance_points(point1, point2, p):
    # Ensure the inputs are array-like
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)

    # Calculate the distance
    dist = np.sum(np.abs(point1 - point2) ** p) ** (1 / p)
    return np.round(dist, 4)


# Example usage
point_a = [1, 2]
point_b = [4, 5]
distance_1 = distance_points(point_a, point_b, 1)  # Manhattan distance
distance_2 = distance_points(point_a, point_b, 2)  # Euclidean distance

print("Manhattan distance:", distance_1)
print("Euclidean distance:", distance_2)


# c. Write a function generate_random_points that generates n d -dimensional points using the uniform distribution.
# The values should be greater than left_range and lower than right_range .

def generate_random_points(n, d, left_range, right_range):
    points = np.random.uniform(left_range, right_range, (n, d))
    return np.round(points, 4)


# Example usage
n_points = 5  # Number of points
dimensions = 2  # Number of dimensions per point
left = 0  # Left range for uniform distribution
right = 10  # Right range for uniform distribution

random_points = generate_random_points(n_points, dimensions, left, right)
print("Generated random points:\n", random_points)


# d. Write a function distance_to_df that calculates the distance between a point X and a dataframe df . The function
# should return a vector with n values that contains the distance between X and each instance belonging to df ( n
# represents the number of instances of the dataframe). Hint: Check the norm calculation function from the numpy module

def distance_to_df(X, df, p):
    distances = {}

    for index, row in df.iterrows():
        # print(row)
        # check the norm calculation function from the numpy module
        distances[index] = np.round(np.linalg.norm(X - row, p), 4)
    return distances


for p in range(0, 5):
    print("Distance to point", p, ":", distance_to_df(p, dataset_final, 2))


# 3. kMeans a. Write a function distance_to_centroids that calculates the distance between the points from a dataset
# and a list of centroids. The function will take as parameters the dataframe df and the list of centroids "centroids"
# and will return a n x m matrix, where n is the number of points from df as m the number of centroids

def distance_to_centroids(df, centroids, p):
    # Empty matrix
    distances_matrix = np.zeros((len(df), len(centroids)))

    # Calc distances
    for i, centroid in enumerate(centroids):
        for j, point in enumerate(df.values):
            distances_matrix[j, i] = np.round(np.linalg.norm(point - centroid, p), 4)

    return distances_matrix


# # Example usage
centroids = generate_random_points(3, 15, 0, 10)
distances_matrix = distance_to_centroids(dataset_final, centroids, 2)
print("Distances matrix:\n", distances_matrix)


# c. Write a function get_clusters that uses the closest centroid list to create the list of clusters. The function
# will return a dictionary { index_centroid_1 : [index point for which centroid 1 is the closest]}

def closest_centroid(distances_matrix):
    closest_indices = np.argmin(distances_matrix, axis=1)
    return closest_indices


closest_centroids = closest_centroid(distances_matrix)
print("Closest centroids:\n", closest_centroids)


# d. Using the list of clusters and the dataframe, write a function update_centroids that will recalculate the
# centroids as the arithmetic mean of the points from each cluster. The function should return a list with the new
# coordinates of the centroid. Notes: 1. Treat the case when a cluster is empty, i.e. there is a centroid that is not
# considered the closest for any of the points from the dataframe. 2. Keep the order of the old indices, meaning the
# new centroid of the cluster 2 should be the third in the list (assuming the indexing starts with 0).

def update_centroids(df, cluster_assignments, n_clusters):
    new_centroids = np.zeros((n_clusters, df.shape[1]))
    for i in range(n_clusters):
        # Extract points assigned to this cluster
        points_in_cluster = df[cluster_assignments == i]

        # Check if the cluster is empty
        if len(points_in_cluster) == 0:
            # Handle empty cluster, reinitialize randomly within the range of the dataset
            new_centroid = np.random.uniform(df.min(), df.max(), df.shape[1])
        else:
            # Calculate the mean of the points in the cluster
            new_centroid = points_in_cluster.mean().values

        new_centroids[i] = np.round(new_centroid, 4)

    return new_centroids


n_clusters = 2
new_centroids = update_centroids(dataset_final, closest_centroids, n_clusters)
print("New centroids:\n", new_centroids)


# e. Write a function that performs the kMeans++ initialisation. The function should take as parameters the dataframe
# df , the desired number of clusters nclusters and the random seed (for reproducibility) and should return the list
# of centroids.

def kmeans_plusplus_init(df, n_clusters, random_seed):
    random.seed(random_seed)

    # Initialize centroids list and add the first centroid
    centroids = []
    centroids.append(df.sample().values[0])

    # Calculate the distance to the closest centroid for each point
    distances = distance_to_centroids(df, centroids, 2)
    min_distances = np.min(distances, axis=1)

    # Repeat until we have n_clusters centroids
    while len(centroids) < n_clusters:
        # Choose a random point with a probability proportional to the distance to the closest centroid
        new_centroid = random.choices(df.values, weights=min_distances)[0]
        centroids.append(new_centroid)

        # Calculate the distance to the closest centroid for each point
        distances = distance_to_centroids(df, centroids, 2)
        min_distances = np.min(distances, axis=1)

    return centroids


n_clusters = 2
random_seed = 0
centroids = kmeans_plusplus_init(dataset_final, n_clusters, random_seed)


# f. Write the implementation of the kMeans algorithm. The function should have the following parameters: the dataframe df , the desired number of
# clusters, the number of iterations, the initalisation type (random or kmeans++) and the random seed. The function should return a dictionary with the
# following fields:
# clusters : the membership vector (for each point, the index of the cluster it belongs to)
# centroids : the coordinates of the centroids

def kmeans(df, n_clusters, n_iterations, init_type, random_seed):
    # Intialize centroids
    if init_type == 'random':
        centroids = generate_random_points(n_clusters, df.shape[1], df.min(), df.max())
    elif init_type == 'kmeans++':
        centroids = kmeans_plusplus_init(df, n_clusters, random_seed)
    else:
        raise ValueError("Invalid init_type")

    for _ in range(n_iterations):
        # Calculate distances to centroids and assign clusters
        distances = distance_to_centroids(df, centroids, 2)
        cluster_assignments = closest_centroid(distances)

        # Update centroids
        centroids = update_centroids(df, cluster_assignments, n_clusters)

    return {'clusters': cluster_assignments, 'centroids': centroids}


# Example usage
n_clusters = 3
n_iterations = 10
random_seed = 42
kmeans_result = kmeans(dataset_final, n_clusters, n_iterations, init_type='random', random_seed=random_seed)

# Printing the final kMeans result with formatted centroids
print("Cluster assignments:", kmeans_result['clusters'])
print("Centroids:")
for centroid in kmeans_result['centroids']:
    formatted_centroid = ['{:.4f}'.format(val) for val in centroid]
    print(formatted_centroid)

# g. Write a function that, given a dataframe df , a membership vector mb and the list of centroids , calculates the J score.

def calculate_J(df, cluster_assignments,centroids):
    j_score = 0
    for i, centroid in enumerate(centroids):
        # Points in current cluster
        points_in_cluster = df[cluster_assignments == i]
        # Sum of squared distances of points to their centroid
        j_score += np.sum(np.linalg.norm(points_in_cluster - centroid) ** 2)

    return j_score

# Example usage
j_score = calculate_J(dataset_final, kmeans_result['clusters'], kmeans_result['centroids'])
print("J score:", j_score)

# h. Write a function that enables multiple initialisations. Besides the parameters specified at f , you will add the number of initialisations.
# By multiple initialisation we understand running kmeans multiple times with different random seeds.
# The function will return the clustering with the best J score.
# The output will be a dictionary with the fields clusters , centroids and J .

def kmeans_multiple_initializations(df, n_clusters, n_iterations, n_initializations, init_type, random_seed=None):
    best_j_score = np.inf
    best_result = None

    for i in range(n_initializations):
        # Set a new random seed for each initialization
        current_seet = random_seed + i if random_seed is not None else None

        # Run kmeans
        result = kmeans(df, n_clusters, n_iterations, init_type, current_seet)

        # Calculate J score
        j_score = calculate_J(df, result['clusters'], result['centroids'])

        #Update bet result if current J score is lower
        if j_score < best_j_score:
            best_j_score = j_score
            best_result = result
            best_result['J_score'] = j_score

    return best_result

# i. Run the kmeans implementation from h on your dataset with the following parameters: ninit = 100, niter = 30,
# init = "kmeans++" and nclusters varying from 2 to 30. Plot the evolution of the J score as the number of clusters
# increases. What is the natural number of clusters in your case? Justify your reasoning

import matplotlib.pyplot as plt
# Parameters
n_iterations = 10
n_initializations = 5
max_clusters = 30
random_seed = 42

# Record the best J scores for different numbers of clusters
j_scores = []

for n_clusters in range(2, max_clusters + 1):
    best_kmeans = kmeans_multiple_initializations(dataset_final, n_clusters, n_iterations, n_initializations, 'kmeans++', random_seed)
    j_scores.append(best_kmeans['J_score'])
    print(f"Number of Clusters: {n_clusters}, Best J score: {best_kmeans['J_score']}")

# Plotting the J scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_clusters + 1), j_scores, marker='o')
plt.title("J Score Evolution with Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("J Score")
plt.grid(True)
plt.show()