#!/usr/bin/env python
# coding: utf-8

# # Microsoft Malware Prediction competition on Kaggle

# ### Goal of the competition:
# ### The general idea is to predict which devices are most likely to be infected by malware based on device characteristics.

# Link to the competition website: https://www.kaggle.com/c/microsoft-malware-prediction

# __General steps to take:__
# 1. Import the dataset and get a feeling for the data.
# 2. EDA (exploratory data analysis) to get first insights and get familiar with the data set.
# 3. Clean the data, processing of data into format that can be used by ML algorithms.
# 4. Build the model and train the model.
# 5. Evaluate the model.
# 6. Change hyper-parameters to improve model performance.
# 7. (Deploy model to be actually usable but skipped here.)

# <img src="1_PAqzvCxPjpDN8RC9HQw45w.png" style="width: 500px;height: 200px"/>

# ## Import all libraries and the dataset, then get a 'feeling' for the data:

# In[1]:


#!/usr/bin/python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)

# import standard ML libraries
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.utils.multiclass import unique_labels

# import specific classifiers
from sklearn.ensemble import RandomForestClassifier

print("Lib import check positive ++.")


# **Import the dataset (or a sample of the entire file) and get a feeling for the data:**

# In[2]:


# import the data and get a feeling for which columns and data types it contains:
df = pd.read_csv("train.csv", nrows=50000, encoding="utf8", engine="python")
print("=" * 15 + "Dataframe column overview: " + "=" * 15)
print(df.describe())
print("=" * 15 + "Dataframe dimensions: " + "=" * 15)
print(df.shape)
print("=" * 15 + "All column names: " + "=" * 15)
print(df.columns.values)


# In[3]:


# df


# In[4]:


# check each column for how many unique values it contains:
unique_values = df.nunique().sort_values()
print(unique_values)


# __Seems like the dataframe has 83 columns of mixed types (binary and non-binary). The 'MachineIdentifier' column seems to be the only column with all unique columns.__

# ## EDA (Exploratory Data Analysis):

# First ideas and hypotheses:
# 1. Touch-enabed devices are more likely to be mobile devices without anti-virus software so they have more malware.
# 2. Same goes for pen-cabable devices, those are probably not PCs/ Laptops with better anti-malware software.
# 3. "Census_IsAlwaysOnAlwaysConnectedCapable" could indicate IoT devices that probably have worse anti-malware software so they have more Malware.

# **1. Touch enabled devices:**

# In[5]:


# select touch enabled devices and see if they have a higher chance of being affected by malware:
df_touchenabled = df[["Census_IsTouchEnabled", "HasDetections"]]
df_touchenabled[["Census_IsTouchEnabled"]] = df_touchenabled[
    ["Census_IsTouchEnabled"]
].astype(bool)
df_touchenabled_pivot_avg = pd.pivot_table(
    df_touchenabled, values="HasDetections", index=["Census_IsTouchEnabled"]
)
df_touchenabled_pivot_count = pd.pivot_table(
    df_touchenabled,
    values="HasDetections",
    index=["Census_IsTouchEnabled"],
    aggfunc="count",
)
print("Dataframe dimensions: ", df_touchenabled.shape)
print(df_touchenabled_pivot_avg)
print(df_touchenabled_pivot_count)


# In[6]:


df_touchenabled_pivot_avg.plot(kind="bar")
df_touchenabled_pivot_count.plot(kind="bar")


#  --> Interestingly, touch-enabled devices actually seem to have a lower probability to have malware than non-touch devices. But also keeping in mind that around 87,5 % of all devices in the sample are not touch-enabled.

# **2. Pen enabled devices:**

# In[7]:


# select touch enabled devices and see if they have a higher chance of being affected by malware:
df_pen_enabled = df[["Census_IsPenCapable", "HasDetections"]]
df_pen_enabled[["Census_IsPenCapable"]] = df_pen_enabled[
    ["Census_IsPenCapable"]
].astype(bool)
df_pen_enabled_pivot_avg = pd.pivot_table(
    df_pen_enabled, values="HasDetections", index=["Census_IsPenCapable"]
)
df_pen_enabled_pivot_count = pd.pivot_table(
    df_pen_enabled,
    values="HasDetections",
    index=["Census_IsPenCapable"],
    aggfunc="count",
)
print("Dataframe dimensions: ", df_pen_enabled.shape)
print(df_pen_enabled_pivot_avg)
print(df_pen_enabled_pivot_count)


#  --> Only around 3.7% are pen-capable, but pen-enabled devices seem to have a marginally lower probability to be affected by malware, although the difference is smaller than with the touch-enabled feature.

# **3. IoT devices:**

# In[ ]:


# select touch enabled devices and see if they have a higher chance of being affected by malware:
df_alwaysConnected = df[["Census_IsAlwaysOnAlwaysConnectedCapable", "HasDetections"]]
df_alwaysConnected[["Census_IsAlwaysOnAlwaysConnectedCapable"]] = df_alwaysConnected[
    ["Census_IsAlwaysOnAlwaysConnectedCapable"]
].astype(bool)
df_alwaysConnected_pivot_avg = pd.pivot_table(
    df_alwaysConnected,
    values="HasDetections",
    index=["Census_IsAlwaysOnAlwaysConnectedCapable"],
)
df_alwaysConnected_pivot_count = pd.pivot_table(
    df_alwaysConnected,
    values="HasDetections",
    index=["Census_IsAlwaysOnAlwaysConnectedCapable"],
    aggfunc="count",
)
print("Dataframe dimensions: ", df_alwaysConnected.shape)
print(df_alwaysConnected_pivot_avg)
print(df_alwaysConnected_pivot_count)


#  --> Only around 6.5% are suspected IoT devices, but they seem to have a lower rate of infection with Malware.

# ## Machine Learning:

# **Preprocessing: Get the columns with only 2 or 3 unique values as a starting point to separate into individual columns, otherwise the number of columns will get huge:**

# In[8]:


# fill all NaN values with a 0.0
# (this assumption may be challenged later on, does not have to be true.)
df = df.fillna(0.0)

# split the data into actual feature data and the target column (in this case "HasDetection")
Y = df[["HasDetections"]]
X = df.drop(["HasDetections"], axis=1)


# In[9]:


# make a list with all the columns that have only 2 unique values in them:
print("====Only 2 unique values.")
df_unique_values_two = unique_values[unique_values == 2]
list_two_unique_value_columns = list(df_unique_values_two.index)
print(list_two_unique_value_columns)

# make a list with all the columns that have only 3 unique values in them:
print("====Only 3 unique values.")
df_unique_values_three = unique_values[unique_values == 3]
list_three_unique_value_columns = list(df_unique_values_three.index)
print(list_three_unique_value_columns)

# concatenate to get a list of all column names with 2 or 3 unique values
list_two_three_unique_value_columns = (
    list_two_unique_value_columns + list_three_unique_value_columns
)


# In[10]:


# make dataframe of the original with only the columns that have two unique values:
df_two_three_uniques = df[list_two_three_unique_value_columns]
print(
    "Number of columns in the dataframe with 2/3 unique values in each column: ",
    df_two_three_uniques.shape[1],
)
# df_two_three_uniques


# In[11]:


# split the data into actual feature data and the target column (in this case "HasDetection")
X_two_three = df_two_three_uniques.drop(["HasDetections"], axis=1)
Y_two_three = df_two_three_uniques[["HasDetections"]]
X_two_three_cols = list(X_two_three.columns)


# In[12]:


# create dummies, split rows with differing values into separate columns:
X_two_three = pd.get_dummies(data=X_two_three, columns=X_two_three_cols)
# X_two_three


# In[13]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_two_three, Y_two_three, test_size=0.3
)  # 70% training and 30% test


# In[14]:


# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y
clf.fit(X_train, y_train)


# **Evaluate how good the model is and check which features contribute most to model:**

# In[15]:


# test the classifier's accuracy:
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))


# In[17]:


cm = confusion_matrix(y_test, y_pred)
labels = [0, 1]
print(cm)


# In[18]:


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=fontsize
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion matrix for classifier:")
    # fig.colorbar(shrink=0.8)
    return fig


fig = print_confusion_matrix(cm, labels)


# In[19]:


# check which features have the highest predictive value
feature_imp = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(
    ascending=False
)
feature_imp


# In[20]:


# get_ipython().run_line_magic("matplotlib", "inline")
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Visualizing Important Features")
plt.legend()
plt.tick_params(axis="both", which="major", labelsize=10, size=10)
plt.show()


# **Take all columns for the ML model:**

# In[21]:


# split the data into actual feature data and the target column (in this case "HasDetection")
Y_all_cols = df[["HasDetections"]]
X_all_cols = df.drop(["HasDetections"], axis=1)
X_all_cols_list = list(X_all_cols.columns.values)
X_all_cols_list


# In[ ]:


# attempt to take out the columns with low variance:
# from sklearn.feature_selection import VarianceThreshold
# sel = VarianceThreshold(threshold=0.5)
# sel.fit_transform(X_all_cols)


# In[22]:


# drop the columns with high variety:
cols_to_drop = [
    "AvSigVersion",
    "Census_FirmwareVersionIdentifier",
    "CityIdentifier",
    "Census_OEMModelIdentifier",
    "Census_SystemVolumeTotalCapacity",
    "MachineIdentifier",
]
X_without_large_cols = X.drop(columns=cols_to_drop, axis=1)
# create dummies, split rows with differing values into separate columns:
X_proc = pd.get_dummies(
    data=X_without_large_cols, columns=list(X_without_large_cols.columns.values)
)
print(X_proc)


# In[23]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_proc, Y, test_size=0.2
)  # 80% training and 20% test


# In[24]:


# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y
clf.fit(X_train, y_train.values.ravel())
# test the classifier's accuracy:
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
labels = [0, 1]
fig = print_confusion_matrix(cm, labels)


# **--> With 58% accuracy slightly better than the previous model but still pretty bad.**

# **Let's try with gridsearch CV to get better hyper parameters for the model:**

# In[ ]:


"""
rfc=RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train.values.ravel())
"""
