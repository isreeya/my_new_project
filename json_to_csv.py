#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
my_file.json is generated from this script
"""
# from sklearn import datasets
# import json 

# iris = datasets.load_iris()

# # convert array to list
# for e in ['data', 'target', 'target_names']:
#     iris[e] = iris[e].tolist()


# with open('data/my_file.json', 'w') as file:
#     json_string = json.dumps(iris)
#     file.write(json_string)
#     file.close()


# In[11]:


"""
1. read my_file.json as dict
2. process the dict into preferred format (table)
3. export csv
"""
import pandas as pd
import json
from datetime import datetime

# read iris dict
with open('data/my_file.json') as file:
    iris_dict = json.load(file)
    
# create dataframe based on data array
df = pd.DataFrame(iris_dict['data'], columns=iris_dict['feature_names'])
df['target'] = iris_dict['target']

def map_target_name(numerical_target, num_to_name):
    """map numerical target to target name
    
    Args:
        numerical_target (int): 0,1,2
    
    Returns:
        target_name (str): 'setosa', 'versicolor', 'virginica'
    """
    
    target_name = num_to_name[numerical_target]
    
    return target_name

num_to_name = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
}

df['target_names'] = df['target'].apply(lambda x: map_target_name(x, num_to_name))

# drop original target axis
df = df.drop('target', axis=1)

# to csv

# output_filename = "final_iris_table_{}.csv".format(datetime.now())
output_filename = "final_iris_table.csv"

df.to_csv("data/{}".format(output_filename), index=False)


# In[ ]:
