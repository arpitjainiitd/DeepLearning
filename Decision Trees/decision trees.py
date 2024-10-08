import numpy as np
import matplotlib.pyplot as plt
from public_tests import *
from utils import *

%matplotlib inline
X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])
print("First few elements of X_train:\n", X_train[:5])
print("Type of X_train:",type(X_train))
print("First few elements of y_train:", y_train[:5])
print("Type of y_train:",type(y_train))
print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))

# UNQ_C1
# GRADED FUNCTION: compute_entropy

def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    # You need to return the following variables correctly
    entropy = 0.
    
    ### START CODE HERE ###
    if(len(y)!=0):
        p1=len(y[y == 1]) / len(y)
        if p1!=0 and p1!=1:
            entropy=-p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
        else:
            entropy=0
            
    ### END CODE HERE ###        
    
    return entropy
# Compute entropy at the root node (i.e. with all examples)
# Since we have 5 edible and 5 non-edible mushrooms, the entropy should be 1"

print("Entropy at root node: ", compute_entropy(y_train)) 

# UNIT TESTS
compute_entropy_test(compute_entropy)

# UNQ_C2
# GRADED FUNCTION: split_dataset

def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split
    
    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """    
    left_indices = []
    right_indices = []
    
    ### START CODE HERE ###
    for i in node_indices:
        if X[i,feature]==1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    ### END CODE HERE ###
    
    return left_indices, right_indices

node_indices = np.arange(len(X_train)).tolist()
feature = 0
left_indices, right_indices = split_dataset(X_train, node_indices, feature)

print('X_train:', X_train)
print('left_indices:', left_indices)
print('right_indices:', right_indices)

# UNIT TESTS
split_dataset_test(split_dataset)

# UNQ_C3
# GRADED FUNCTION: compute_information_gain

def compute_information_gain(X, y, node_indices, feature):
    """
    Computes the information gain
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        y (array like):          list or ndarray with n_samples containing the target variable
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split
    
    Returns:
        info_gain (float):       Information gain
    """    
    # You need to return the following variables correctly
    info_gain = 0
    
    ### START CODE HERE ###
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    entropy_before=compute_entropy(y[node_indices])
    entropy_left=compute_entropy(y[left_indices])
    entropy_right=compute_entropy(y[right_indices])
    
    w_left=len(left_indices)/len(node_indices)
    w_right=len(right_indices)/len(node_indices)
    info_gain=entropy_before-(w_left*entropy_left+w_right*entropy_right)
    ### END CODE HERE ###
    
    return info_gain

feature = 0
node_indices = np.arange(len(X_train)).tolist()
print("Information gain:", compute_information_gain(X_train, y_train, node_indices, feature))

# UNIT TESTS
compute_information_gain_test(compute_information_gain)

# UNQ_C4
# GRADED FUNCTION: get_best_split

def get_best_split(X, y, node_indices):
    """
    Returns the best feature to split on
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        y (array like):          list or ndarray with n_samples containing the target variable
        node_indices (list):     List containing the active indices. I.e, the samples being considered in this step.
        
    Returns:
        best_feature (int):      Index of the best feature to split
    """
    
    # You need to return the following variables correctly
    best_feature = -1
    
    ### START CODE HERE ###
    max_info_gain = -1
    n_features = X.shape[1]
    for feature in range(n_features):
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
    ### END CODE HERE ###
    
    return best_feature

node_indices = np.arange(len(X_train)).tolist()
best_feature = get_best_split(X_train, y_train, node_indices)
print('Best feature:', best_feature)

# UNIT TESTS
get_best_split_test(get_best_split)

# Not graded
tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
   
    """ 

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)

build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
generate_tree_viz(root_indices, y_train, tree)
