from random import random

import pandas as pd
import math
from pprint import pprint

from sklearn.preprocessing import LabelEncoder
from sklearn.svm._libsvm import predict
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

## 1. Preprocessing

# Read in data
data = pd.read_csv('champions_info.csv')

# b. Drop unnecessary columns
data = data.dropna()

# Describe the dataset
print(data.describe())

# a. Identify attributes and target attribute
attributes = data.columns[:-1]  # all columns except the last one
target_attribute = 'first_blood_kill'

# a. Identify discrete and continuous attributes
discrete_attributes = ['league', 'result', 'side', 'Difficulty', 'Items_For_Spike', 'Attack_Type', 'top',
                       'mid', 'jng', 'sup', 'bot']
continuous_attributes = ['kills', 'deaths', 'assists', 'damagetochampions']

# c. Calculate the mean and variance for each numerical attribute
mean_variances = {}
for attribute in continuous_attributes:
    mean_variances[attribute] = [data[attribute].mean(), data[attribute].var()]

print(mean_variances)


## 3. Probabilities and Information Theory

# a. Calculate the probability mass function of discrete attributes.

def compute_probabilities(data, attribute):
    return data[attribute].value_counts(normalize=True).to_dict()


# e.g. compute_probabilities(data, 'league')
probabilites = compute_probabilities(data, 'league')
print("Probabilities : ", probabilites)


# b. Calculate the entropy for discrete attributes.

def calculate_entropy(data, attribute):
    probabilities = compute_probabilities(data, attribute)
    entropy = -sum(p * math.log2(p) for p in probabilities.values())
    return entropy


# e.g. calculate_entropy(data, 'league')
entropy_league = calculate_entropy(data, 'league')
print("Entropy : ", entropy_league)


# c. Calculate conditional entropy for target attribute and a discrete attribute.

def calculate_conditional_entropy(data, attribute, target_attribute):
    conditional_entropy = 0
    for attribute_values in data[attribute].unique():
        subset = data[data[attribute] == attribute_values]
        probability_attribute_values = len(subset) / len(data)
        entropy_attribute_value = calculate_entropy(subset, target_attribute)
        conditional_entropy += probability_attribute_values * entropy_attribute_value
    return conditional_entropy


# e.g. calculate_conditional_entropy(data, 'league', 'first_blood_kill')
conditional_entropy = calculate_conditional_entropy(data, 'league', 'first_blood_kill')
print(f"Condititonal Entropy of league and first_blood_kill : ", conditional_entropy)


# d. Calculate the infroamtion gain for discrete attributes.

def calculate_information_gain(data, attribute, target_attribute):
    target_entropy = calculate_entropy(data, target_attribute)
    conditional_entropy = calculate_conditional_entropy(data, attribute, target_attribute)
    information_gain = target_entropy - conditional_entropy
    return information_gain


# e.g. calculate_information_gain(data, 'league', 'first_blood_kill')
information_gain = calculate_information_gain(data, 'league', 'first_blood_kill')
print("Information Gain of league and first_blood_kill: ", information_gain)


## 3. ID3

# a. Find the root node of the decision tree.

def find_root_node(data, attributes, target_attribute):
    information_gains = [(attribute, calculate_information_gain(data, attribute, target_attribute)) for attribute in
                         attributes]
    best_attribute, best_information_gain = max(information_gains, key=lambda x: x[1])
    return best_attribute, best_information_gain


# e.g. find_root_node(data, attributes, target_attribute)
root_node, root_ig = find_root_node(data, discrete_attributes, 'first_blood_kill')
print("Root Node:", root_node, "Information Gain:", root_ig)


# b. . Write a function id3_discrete that implements the ID3 algorithm for the discrete attributes. The function should return a dictionary following this
# structure

def id3_discrete(data, discrete_attributes, target_attribute):
    # if all target attributes have the same value, return that value
    if len(data[target_attribute].unique()) == 1:
        return data[target_attribute].unique()[0]

    # if data is empty, return the most common value of the target attribute
    if len(data) == 0:
        return data[target_attribute].value_counts().idxmax()

    # if there are no attributes left, return the most common target attribute value
    if len(discrete_attributes) <= 1:
        return data[target_attribute].value_counts().idxmax()

    # Choose the attribute with the highest information gain
    root_attribute, root_information_gain = find_root_node(data, discrete_attributes, target_attribute)
    tree = {
        "node_attribute": root_attribute,
        "observations": dict(data[target_attribute].value_counts()),
        "information_gain": calculate_information_gain(data, root_attribute, target_attribute),
        "values": {}
    }

    # For each value of the root attribute, create a new subtree
    for value in data[root_attribute].unique():
        # Create a new subtree for the current value
        subtree = data[data[root_attribute] == value]

        # Remove the root attribute from the list of attributes
        new_attributes = [attr for attr in discrete_attributes if attr != root_attribute]

        # Recursively call the ID3 algorithm
        subtree_result = id3_discrete(subtree, new_attributes, target_attribute)

        # Add the new subtree to the existing tree
        tree["values"][value] = subtree_result

    return tree


print(discrete_attributes)
pprint(id3_discrete(data, discrete_attributes, target_attribute))

# c. Run id3_discrete on the dataset containing only discrete attributes.
# Compare the results with the ones from sklearn . (make your comparison as
# thorough as possible)

# Assuming 'data' is your preprocessed dataset with discrete attributes
X = data[discrete_attributes]
y = data[target_attribute]

# One-hot encode the discrete attributes
X = pd.get_dummies(X)

# Use sklearn's DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion='entropy')
dt_classifier.fit(X, y)

tree_rules = export_text(dt_classifier, feature_names=list(X.columns))
print(tree_rules)


# putem observa ca este acelasi nod radacina, items_for_spike

# d. Write a function get_splits which, given a continuous attribute and the labels,
# will identify the splits that could be used to discretization of the
# variable. Test your function on an example.

def get_splits(labels, attribute):
    labels = sorted(data[attribute].unique())
    splits = []
    for i in range(len(labels) - 1):
        splits.append((labels[i] + labels[i + 1]) / 2)
    return splits


labels = []
# example from my dataset
print(get_splits(labels, 'kills'))


# e. Write a function id3 that implements ID3 on the entire dataset,
# both continuous and discrete attributes. The function should return a dictionary
# similar with the one above. Compare the results with the ones from sklearn .

# First we need to add new functions to calculate the information gain for continuous attributes

# Calculate information gain for continuous attributes

def continous_information_gain(data, attribute, target_attribute, split_point):
    subset1 = data[data[attribute] <= split_point]
    subset2 = data[data[attribute] > split_point]

    p1 = len(subset1) / len(data)
    p2 = len(subset2) / len(data)

    entropy1 = calculate_entropy(subset1, target_attribute)
    entropy2 = calculate_entropy(subset2, target_attribute)

    conditional_entropy = p1 * entropy1 + p2 * entropy2
    return conditional_entropy


# Find the best split point for a continuous attribute
def find_best_split(data, attribute, target_attribute):
    splits = get_splits(data[attribute], attribute)
    information_gains = [(split_point, continous_information_gain(data, attribute, target_attribute, split_point)) for
                         split_point in splits]
    best_split, best_information_gain = max(information_gains, key=lambda x: x[1])
    return best_split


# find root note that handles both discrete and continuous attributes
def find_root_node_discrete_and_continuous(data, attributes, target_attribute):
    information_gains = [(attribute, calculate_information_gain(data, attribute, target_attribute)) for attribute in
                         attributes]
    for attribute in continuous_attributes:
        best_split = find_best_split(data, attribute, target_attribute)
        information_gains.append((attribute, continous_information_gain(data, attribute, target_attribute, best_split)))
    best_attribute, best_information_gain = max(information_gains, key=lambda x: x[1])
    return best_attribute, best_information_gain


def id3(data, attributes, target_attribute):
    # if all target attributes have the same value, return that value
    if len(data[target_attribute].unique()) == 1:
        return data[target_attribute].unique()[0]

    # if data is empty, return the most common value of the target attribute
    if len(data) == 0:
        return data[target_attribute].value_counts().idxmax() if not data.empty else None

    # if there are no attributes left, return the most common target attribute value
    if len(attributes) <= 1:
        return data[target_attribute].value_counts().idxmax() if not data.empty else None

    # Choose the attribute with the highest information gain
    root_attribute, root_information_gain = find_root_node(data, attributes, target_attribute)
    if root_attribute in continuous_attributes:
        best_split = find_best_split(data, root_attribute, target_attribute)
        best_information_gain = continous_information_gain(data, root_attribute, target_attribute, best_split)
        tree = {
            "node_attribute": root_attribute,
            "observations": dict(data[target_attribute].value_counts()),
            "information_gain": best_information_gain,
            "split_point": best_split,
            "values": {}
        }

        # Create two subtrees based on the split
        for branch in [True, False]:
            subset = data[data[root_attribute] <= best_split] if branch else data[data[root_attribute] > best_split]
            subtree_result = id3(subset, attributes, target_attribute)
            tree["values"][f"branch_{branch}"] = subtree_result

    # Daca e discret
    else:
        tree = {
            "node_attribute": root_attribute,
            "observations": dict(data[target_attribute].value_counts()),
            "information_gain": root_information_gain,
            "values": {}
        }
        # For each value of the root attribute, create a new subtree
        for value in data[root_attribute].unique():
            subtree = data[data[root_attribute] == value]
            new_attributes = [attr for attr in discrete_attributes if attr != root_attribute]
            subtree_result = id3_discrete(subtree, new_attributes, target_attribute)
            tree["values"][value] = subtree_result

    return tree


# pprint(id3(data, discrete_attributes + continuous_attributes, target_attribute))

# def id3(data, attributes, target_attribute, current_depth=0, max_depth=2):
#     # if all target attributes have the same value, return that value
#     if len(data[target_attribute].unique()) == 1:
#         return data[target_attribute].unique()[0]
#
#         # if data is empty, return the most common value of the target attribute
#     if len(data) == 0:
#         return data[target_attribute].value_counts().idxmax() if not data.empty else None
#
#         # if there are no attributes left or reached max depth, return the most common target attribute value
#     if len(attributes) <= 1 or current_depth == max_depth:
#         return data[target_attribute].value_counts().idxmax() if not data.empty else None
#
#     # Choose the attribute with the highest information gain
#     root_attribute, root_information_gain = find_root_node(data, attributes, target_attribute)
#     if root_attribute in continuous_attributes:
#         best_split = find_best_split(data, root_attribute, target_attribute)
#         best_information_gain = continous_information_gain(data, root_attribute, target_attribute, best_split)
#         tree = {
#             "node_attribute": root_attribute,
#             "observations": dict(data[target_attribute].value_counts()),
#             "information_gain": best_information_gain,
#             "split_point": best_split,
#             "values": {}
#         }
#         left_subset = data[data[root_attribute] <= best_split]
#         right_subset = data[data[root_attribute] > best_split]
#
#         # Create two subtrees based on the split
#         for branch, subset in zip([True, False], [left_subset, right_subset]):
#             subtree_result = id3(subset, attributes, target_attribute, current_depth=current_depth + 1,
#                                  max_depth=max_depth)
#             tree["values"][f"branch_{branch}"] = subtree_result
#
#     # Daca e discret
#     else:
#         tree = {
#             "node_attribute": root_attribute,
#             "observations": dict(data[target_attribute].value_counts()),
#             "information_gain": root_information_gain,
#             "values": {}
#         }
#         # For each value of the root attribute, create a new subtree
#         for value in data[root_attribute].unique():
#             subset = data[data[root_attribute] == value]
#             new_attributes = [attr for attr in discrete_attributes if attr != root_attribute]
#             subtree_result = id3_discrete(subset, new_attributes, target_attribute, current_depth=current_depth + 1,
#                                           max_depth=max_depth)
#             tree["values"][value] = subtree_result
#
#     return tree
#
#
# # Call the modified id3 function
# result_tree = id3(data, discrete_attributes + continuous_attributes, target_attribute, max_depth=2)
#
# # Pretty print the resulting tree
# pprint(result_tree)

# make tree with sklearn
X = data[discrete_attributes + continuous_attributes]
y = data[target_attribute]

# One-hot encode the discrete attributes
X = pd.get_dummies(X)

# Use sklearn's DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion='entropy')
dt_classifier.fit(X, y)

tree_rules = export_text(dt_classifier, feature_names=list(X.columns))
print(tree_rules)


# f. Modify the two ID3 functions such that they will allow pruning.
# Use TWO methods of pruning, one of which should be based on the depth of the tree.

# Prune by depth

def id3_discrete_depth_pruning(data, discrete_attributes, target_attribute, max_depth=None, current_depth=0):
    # if all target attributes have the same value, return that value
    if len(data[target_attribute].unique()) == 1:
        return data[target_attribute].unique()[0]

    # if data is empty, return the most common value of the target attribute
    if len(data) == 0:
        return data[target_attribute].value_counts().idxmax() if not data.empty else None

    # if there are no attributes left or reached max depth, return the most common target attribute value
    if len(discrete_attributes) <= 1 or current_depth == max_depth:
        return data[target_attribute].value_counts().idxmax() if not data.empty else None

    # Choose the attribute with the highest information gain
    root_attribute, root_information_gain = find_root_node(data, discrete_attributes, target_attribute)
    tree = {
        "node_attribute": root_attribute,
        "observations": dict(data[target_attribute].value_counts()),
        "information_gain": calculate_information_gain(data, root_attribute, target_attribute),
        "values": {}
    }

    # For each value of the root attribute, create a new subtree
    for value in data[root_attribute].unique():
        # Create a new subtree for the current value
        subtree = data[data[root_attribute] == value]

        # Remove the root attribute from the list of attributes
        new_attributes = [attr for attr in discrete_attributes if attr != root_attribute]

        # Recursively call the ID3 algorithm
        subtree_result = id3_discrete_depth_pruning(subtree, new_attributes, target_attribute, max_depth=max_depth,
                                                    current_depth=current_depth + 1)

        # Add the new subtree to the existing tree
        tree["values"][value] = subtree_result

    return tree


# Call the modified id3 function
result_tree = id3_discrete_depth_pruning(data, discrete_attributes, target_attribute, max_depth=2)
pprint(result_tree)


# Prune by number of observations
def id3_discrete_observation_pruning(data, discrete_attributes, target_attribute, min_obs):
    # if all target attributes have the same value, return that value
    if len(data[target_attribute].unique()) == 1:
        return data[target_attribute].unique()[0]

    # if data is empty or observations are less than min_obs, return the most common value of the target attribute
    if len(data) == 0 or len(data) < min_obs:
        return data[target_attribute].value_counts().idxmax() if not data.empty else None

    # if there are no attributes left, return the most common target attribute value
    if len(discrete_attributes) <= 1:
        return data[target_attribute].value_counts().idxmax() if not data.empty else None

    # Choose the attribute with the highest information gain
    root_attribute, root_information_gain = find_root_node(data, discrete_attributes, target_attribute)
    tree = {
        "node_attribute": root_attribute,
        "observations": dict(data[target_attribute].value_counts()),
        "information_gain": calculate_information_gain(data, root_attribute, target_attribute),
        "values": {}
    }

    # For each value of the root attribute, create a new subtree
    for value in data[root_attribute].unique():
        # Create a new subtree for the current value
        subtree = data[data[root_attribute] == value]

        # Remove the root attribute from the list of attributes
        new_attributes = [attr for attr in discrete_attributes if attr != root_attribute]

        # Recursively call the ID3 algorithm
        subtree_result = id3_discrete_observation_pruning(subtree, new_attributes, target_attribute, min_obs)

        # Add the new subtree to the existing tree
        tree["values"][value] = subtree_result

    return tree


# Call the modified id3 function
result_tree = id3_discrete_observation_pruning(data, discrete_attributes, target_attribute, min_obs=3)
pprint(result_tree)

