
# coding: utf-8

# In[89]:


"""
Date: 13 May, 2018
Time: 17:38
Author: Vageesh Saxena
Ver: 1.0
Description: Predicting the max temperature in SEATTLE,WA using 6 years of past weather data.
Machine Learning Technique: Random Forest Regression
Dataset: https://www.ncdc.noaa.gov/cdo-web/
Prerequisites: The Dataset should be named as weather.csv and should be present in the working directory
"""


# In[55]:


"""Installing the necessary libraries"""
get_ipython().system(' pip install sklearn')
get_ipython().system(' pip install pydot')
# Install graphviz in sudo mode if already not installed
#! sudo apt-get install graphviz
get_ipython().system(' pip install matplotlib')
get_ipython().system(' pip install pandas')
get_ipython().system(' pip install numpy')


# In[61]:


"""Importing the libraries"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Getting the data from csv file
df = pd.read_csv("weather.csv")


# In[26]:


# Finding the shape and descriptive statistics of dataframe
# print("Shape of df : ",df.shape)
# print("Description of df : ",df.describe())


# In[25]:


# Changing the single column of weekdays into seven columns(One-hot encode) of binary data
df = pd.get_dummies(df)


# In[24]:


# Getting the features and labels
# Labels are the values we want to predict
labels = np.array(df['actual'])
# Remove the labels from the features
# axis 1 refers to the columns
features= df.drop('actual', axis = 1)


# In[82]:


# Saving feature names for later use
feature_list = list(features.columns)
# Convert features to numpy array
features = np.array(features)


# In[23]:


# Split the data into training and testing sets to 0.25
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[27]:


# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('average')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))


# In[32]:


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);


# In[33]:


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)


# In[38]:


# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
"""The average estimate thus found is off by 3.83 degrees. That is more than a 1 degree average improvement over 
the baseline. Although this might not seem significant, it is nearly 25% better than the baseline."""


# In[39]:


"""Determination of the Performance Metrics"""
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[50]:


"""Virtualizing a single decision tree"""
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


# In[51]:


"""Reducing the size of the tree annotated with labels."""
# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(train_features, train_labels)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');


# In[53]:


"""Feature Relevance"""
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
"""In future implementations of the model, one can remove those variables that have no importance and the 
performance will not suffer. Additionally, if we are using a different model, say a support vector machine, we 
could use the random forest feature importances as a kind of feature selection method."""


# In[60]:


"""Visualization of Feature Importance"""
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Features'); plt.title('Feature Importances');


# In[84]:


"""Plotting the entire dataset with predictions. """
# Dates of training values
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]
# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]       
# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels}) 


# In[85]:


# Dates of predictions
months = test_features[:, feature_list.index('month')]
days = test_features[:, feature_list.index('day')]
years = test_features[:, feature_list.index('year')]
# Column of dates
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})


# In[86]:


# Plot the actual values
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()
# Graph labels
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');


# In[87]:


# Make the data accessible for plotting
true_data['temp_1'] = features[:, feature_list.index('temp_1')]
true_data['average'] = features[:, feature_list.index('average')]
true_data['friend'] = features[:, feature_list.index('friend')]
# Plot all the data as lines
plt.plot(true_data['date'], true_data['actual'], 'b-', label  = 'actual', alpha = 1.0)
plt.plot(true_data['date'], true_data['temp_1'], 'y-', label  = 'temp_1', alpha = 1.0)
plt.plot(true_data['date'], true_data['average'], 'k-', label = 'average', alpha = 0.8)
plt.plot(true_data['date'], true_data['friend'], 'r-', label = 'friend', alpha = 0.3)
# Formatting plot
plt.legend(); plt.xticks(rotation = '60');
# Lables and title
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual Max Temp and Variables');


# In[ ]:


"""It is a little hard to make out all the lines, but we can see why the max temperature one day prior and the 
historical max temperature are useful for predicting max temperature."""

