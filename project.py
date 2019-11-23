# BASIC
# -----------------------------------------------
# 1- refactor import code to use sklearn
# -----------------------------------------------
#
# SEE README.MD for analysis
#
# -----------------------------------------------
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import f1_score
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# performance print results
def performance(lr, X_test, y_test):
    # Predicting the results for our test dataset
    predicted_values = lr.predict(X_test)

    print('------------------------------------------------------')

    # Printing accuracy score(mean accuracy) from 0 - 1
    print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

    # Printing the classification report
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    print('Classification Report')
    print(classification_report(y_test, predicted_values))

    # Printing the classification confusion matrix (diagonal is true)
    print('Confusion Matrix')
    print(confusion_matrix(y_test, predicted_values))

    print('Overall f1-score')
    print(f1_score(y_test, predicted_values, average="macro"))

    from sklearn import metrics
    print(f"Printing MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
    print(f"Printing MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
    print(f"Printing RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")

    print('------------------------------------------------------')

def removeColumn(df, column):
    print(f"--------------------- Dropping : {column}  -----------------------")

    df.drop(column, axis=1, inplace=True)
    print(f"Columns : {df.columns}")

    if df.size > 0:
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.35)

        print('Classification Report for SVC')
        from sklearn.svm import SVC
        lr = SVC(kernel='linear', C = 1.0)
        lr.fit(X_train, y_train)

        performance(lr, X_test, y_test)
        visualize(df)

def visualize(df):
    # Just for fun visualizing structure of dataset in 2D
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    if df.count().size > 2:
        pca = PCA(n_components=3)
        proj = pca.fit_transform(df)
        plt.scatter(proj[:, 0], proj[:, 1], c=data.target, edgecolors='black')
        plt.colorbar()
        plt.show()
    plt.close()

data = load_wine()
df = pd.DataFrame(data=data['data'],
                  columns=data['feature_names'])


# -----------------------------------------------
# 1- visualization of the wine dataset correlation
# -----------------------------------------------
# sea.set()
#
# fig, ax = plt.subplots(figsize=(20,20))
# sea.heatmap(df.corr(), annot=True, cmap='summer')
# ax.set_xticklabels(df.columns)
# ax.set_yticklabels(df.columns)
# plt.savefig('plots//wine_correlation_heatmap.png')
#
# plt.clf()


# -----------------------------------------------
# 2- cleanup of the wine dataset, extracting data
# -----------------------------------------------

correlation_list = []
positions = []
i = 0
j = 0

for x in df.corr().values:
    for y in x:
        if y < 0.1 and y > -0.1:
            if y not in correlation_list:
                correlation_list.append(y)
                position = (i, j)
                positions.append(position)
        x = x + 1
        j = j + 1
    j = 0
    i = i + 1
print("------------------------------------------------------------------")
print(correlation_list) #values
print(positions)        #tuples

corr = df.corr()

# get matches
print("------------------------------------------------------------------")
for pos in positions:
    print(corr.columns.values[pos[0]], " and ", corr.index.values[pos[1]])

# iterate over list of tuples to identify the most prevalent feature
count = [0] * corr.columns.values.size
for feature in positions:
    # count feature
    count[feature[0]] += 1
    count[feature[1]] += 1

# print the results
print("------------------------------------------------------------------")
matches = []
column_position = 0
for entry in count:
    print(corr.columns.values[column_position], "has count of", entry)
    match = (entry, corr.columns.values[column_position])
    matches.append(match)
    column_position += 1

#order by count
matches.sort(key=lambda tup: tup[0], reverse=True)

# we have a sorted list of tuples with count and column name for the less correlated entries
print(matches)

# -----------------------------------------------
# 3- classification regression on the wine dataset
# -----------------------------------------------

# for column in df.columns:
#     removeColumn(df, column)