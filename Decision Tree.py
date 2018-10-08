import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import random
from pprint import pprint

dataset = pd.read_csv("Iris.csv")
dataset = dataset.rename(columns={"sepal.length":"sepal_length","sepal.width":"sepal_width","petal.length":"petal_length","petal.width":"petal_width", "variety":"label"})
dataset.head()


'''Hepler Function Code'''

def train_test_split(dataset,test_size):
    if isinstance(test_size,float):
        test_size = round(test_size*len(dataset))
    
    indices = dataset.index.tolist()
    test_indices = random.sample(population=indices,k=test_size)
    test_dataset = dataset.loc[test_indices]
    train_dataset = dataset.drop(test_indices)
    
    return train_dataset,test_dataset


train_dataset,test_dataset = train_test_split(dataset,0.2)
train_dataset.head()
test_dataset.head() 

data = train_dataset.values                          #numpy array
data[:5]


def check_purity(data):
    label_column = data[:,-1]
    unique_classes = np.unique(label_column)
    
    if len(unique_classes)==1:
        return True
    else:
        return False
    
check_purity(train_dataset[train_dataset.petal_width < 0.8].values)


def classify_data(data):
    label_column = data[:,-1]
    unique_classes,count_unique_classes = np.unique(label_column,return_counts=True)
    
    index = count_unique_classes.argmax()
    classification = unique_classes[index]
    return classification

classify_data(train_dataset[train_dataset.petal_width < 0.8].values)


def get_potential_splits(data):
    potential_splits = {}
    _,n_columns = data.shape
    for column_index in range(n_columns-1):
        potential_splits[column_index] = []
        values = data[:,column_index]
        unique_values = np.unique(values)
        
        for index in range(len(unique_values)):
            if index!=0:
                current_value = unique_values[index]
                previous_value = unique_values[index-1]
                potential_split = (current_value + previous_value)/2
                potential_splits[column_index].append(potential_split)
        
    return potential_splits

potential_splits = get_potential_splits(data)

sns.lmplot(data=train_dataset, x="petal_width" , y="petal_length", hue="label",fit_reg=False,size=6,aspect=1.5)


def split_data(data,split_column,split_value):
    split_column_values = data[:,split_column]
    data_below = data[split_column_values<=split_value]
    data_above = data[split_column_values>split_value]
    
    return data_below, data_above

split_column=3
split_value=0.8
data_below,data_above = split_data(data,split_column,split_value)
plotting_dataset = pd.DataFrame(data, columns=dataset.columns)
sns.lmplot(data= plotting_dataset,x="petal_width", y="petal_length", fit_reg=False,size=6,aspect=1.5)
plt.vlines(x=split_value,ymin=1,ymax=7)


def calculate_entropy(data):
    label_column = data[:,-1]
    _,counts = np.unique(label_column,return_counts=True)
    probabilities = counts/counts.sum()
    entropy = sum(probabilities*-np.log2(probabilities))
    
    return entropy

def calculate_overall_entropy(data_below,data_above):
    n_data_points = len(data_below)+len(data_above)
    p_data_below = len(data_below)/n_data_points
    p_data_above = len(data_above)/n_data_points
    overall_entropy = (p_data_below*calculate_entropy(data_below) + p_data_above*calculate_entropy(data_above))
    
    return overall_entropy


def determine_best_split(data,potential_splits):
    
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data,column_index,value)
            current_overall_entropy = calculate_overall_entropy(data_below,data_above)
            
            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column,best_split_value

determine_best_split(data,potential_splits)


'''Decision Tree Main Algorithm Code'''

def decision_tree_algorithm(dataset,counter=0,min_samples=5,max_depth=5):
    
    #data preparations
    if counter==0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = dataset.columns
        data = dataset.values                  #convert dataframe to numpy n-d array
    else:
        data = dataset
    
    #base case
    if (check_purity(data)) or (len(dataset) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        return classification
    
    #recurise call
    else:
        counter+=1
        
        #helper function calls
        potential_split = get_potential_splits(data)
        split_column, split_value = determine_best_split(data,potential_split)
        data_below, data_above = split_data(data,split_column,split_value)
        
        #instantiate sub_tree
        split_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(split_name,split_value)
        subtree = {question:[]}
        
        #recursion
        yes_answer = decision_tree_algorithm(data_below,counter,min_samples,max_depth)
        no_answer = decision_tree_algorithm(data_above,counter,min_samples,max_depth)
       
        if yes_answer == no_answer:
            subtree = yes_answer
        else:
            subtree[question].append(yes_answer)
            subtree[question].append(no_answer)
        
        return subtree
    


'''Classificy Examples'''
def classify_example(example,tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split()
    
    if example[feature_name] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]
    
    #base case
    if not isinstance(answer,dict):
        return answer
    
    #recursive call
    else:
        sub_tree = answer
        return classify_example(example,sub_tree)
    
def calculate_accuracy(dataset,tree):
    dataset["classification"] = dataset.apply(classify_example,axis=1,args=(tree,))
    dataset["classification_correct"] = dataset.classification == dataset.label
    
    accuracy = dataset.classification_correct.mean()
    
    return accuracy


train_dataset,test_dataset = train_test_split(dataset,0.2)
train_dataset.head()
test_dataset.head() 

data = train_dataset.values                          #numpy array
data[:5]

tree = decision_tree_algorithm(train_dataset,0,0,3)
pprint(tree)
calculate_accuracy(test_dataset,tree)
    
    
   
        
        
        
    
    






