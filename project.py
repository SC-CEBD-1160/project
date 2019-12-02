# BASIC
# -----------------------------------------------
# 1- refactor import code to use sklearn
# -----------------------------------------------
#
# SEE README.MD for analysis
#
# -----------------------------------------------
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import f1_score
import seaborn as sea
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sample_count = 50 #50
range_count = 20 #20
correlation_coeficient = 0.05

lr = SVC(kernel='linear', C=1)
# lr = lr = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=4000)
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


def removeColumn(df, column, index):
    # drop feature
    df.drop(column, axis=1, inplace=True)
    # print(f"Columns : {df.columns}")
    for j in range(0, range_count):
       average = 0.0
       if df.size > 0:
           for i in range(0, sample_count):
                y = data.target
                X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25)
                lr.fit(X_train, y_train)

                predicted_values = lr.predict(X_test)
                average += f1_score(y_test, predicted_values, average="macro")

           print("Average Overall f1-score = ", average / sample_count)
           x[j][index] = (average / sample_count)

def visualize(df):
    # Visualizing structure of dataset in 2D
    if df.count().size > 2:
        pca = PCA(n_components=3)
        proj = pca.fit_transform(df)
        plt.scatter(proj[:, 0], proj[:, 1], c=data.target, edgecolors='black')
        plt.colorbar()
        plt.show()
    plt.close()

# -----------------------------------------------
# 1- Load wine dataset
# -----------------------------------------------

print("------------------------------------------------------------------")
print(" Load wine dataset")
print("------------------------------------------------------------------")

raw_data = load_wine()
df = pd.DataFrame(data=raw_data['data'],
                  columns=raw_data['feature_names'])

data = df

# -----------------------------------------------
# 2- visualization of the wine dataset correlation
# -----------------------------------------------

print("------------------------------------------------------------------")
print(" correlation heatmap")
print("------------------------------------------------------------------")

sea.set()

fig, ax = plt.subplots(figsize=(20,20))
sea.heatmap(df.corr(), annot=True, cmap='autumn')
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.savefig('plots//wine_correlation_heatmap.png')

plt.clf()
plt.close()

# We'll use Seaborn's .kdeplot() method so we can cleanly distinguish each class per feature
print("------------------------------------------------------------------")
print(" kdeplot to show class distribution per feature")
print("------------------------------------------------------------------")

features = pd.DataFrame(data=raw_data['data'],columns=raw_data['feature_names'])
data = features
data['target']=raw_data['target']
data['class']=data['target'].map(lambda ind: raw_data['target_names'][ind])
data.head()

os.makedirs('plots/class_feature/', exist_ok=True)

for feature in raw_data['feature_names']:
    print(feature)
    fig, ax = plt.subplots(figsize=(5, 5))
    gs1 = gridspec.GridSpec(3,1)
    ax1 = plt.subplot(gs1[:-1])
    ax2 = plt.subplot(gs1[-1])
    sea.boxplot(x=feature, y='class', data=data, ax=ax2)
    sea.kdeplot(data[feature][data.target==0],ax=ax1,label='0')
    sea.kdeplot(data[feature][data.target==1],ax=ax1,label='1')
    sea.kdeplot(data[feature][data.target==2],ax=ax1,label='2')
    ax2.yaxis.label.set_visible(False)
    ax1.xaxis.set_visible(False)

    if feature == 'od280/od315_of_diluted_wines':
        plt.savefig(f'plots/class_feature/od280_class.png')
    else:
        plt.savefig(f'plots/class_feature/{feature}_class.png')

    plt.close(fig)

# -----------------------------------------------
# 3- cleanup of the wine dataset, extracting data
# -----------------------------------------------

print("------------------------------------------------------------------")
print(" Extracting features that have low correlation")
print("------------------------------------------------------------------")

correlation_list = []
positions = []
i = 0
j = 0

for x in df.corr().values:
    for y in x:
        if y < correlation_coeficient and y > -correlation_coeficient:
            if y not in correlation_list:
                correlation_list.append(y)
                position = (i, j)
                positions.append(position)
        x = x + 1
        j = j + 1
    j = 0
    i = i + 1

print("------------------------------------------------------------------")
print(correlation_list)  # values
print(positions)  # tuples

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

# order by count
matches.sort(key=lambda tup: tup[0], reverse=True)

# we have a sorted list of tuples with count and column name for the less correlated entries
print(matches)

# -----------------------------------------------
# 3- classification regression on the wine dataset
# -----------------------------------------------

# Splitting features and target datasets into: train and test
X = raw_data.data
y = raw_data.target

data = raw_data

# Splitting features and target datasets into: train and test
# initial overall-f1 with no dropped feature
print("------------------------------------------------------------------")
print(" Initial overall calculation features (baseline all features)")
print("------------------------------------------------------------------")
all = []
average = 0.0
for i in range(0, range_count):
    average = 0.0
    for j in range(0, sample_count):
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25)
        lr.fit(X_train, y_train)
        predicted_values = lr.predict(X_test)
        average += f1_score(y_test, predicted_values, average="macro")

    print("Average Overall f1-score = ", average / sample_count)
    all.append(average/sample_count)

print(all)
count = 0.0
for value in all:
    count = count + value
print("Total Average Overall f1-score = ", count / range_count)

    # performance(lr, X_test, y_test)
    # visualize(df)

# single removal of features
all_features = []
x = np.zeros((range_count, 13))
print("------------------------------------------------------------------")
print(" Single removal of features")
print("------------------------------------------------------------------")
count = 0
for column in matches:
   df = pd.DataFrame(data=data['data'],
                  columns=data['feature_names'])
   print("--------------------- Dropping feature : " + column[1])
   removeColumn(df, column[1], count)
   all_features.append(f'{column[1]}')
   count += 1

single_removal_results = pd.DataFrame(data=x,
                       columns=all_features)

print(single_removal_results)

single_removal_results.insert(0,'All', all)

tsint = single_removal_results.interpolate(method='linear')

print("------------------------------------------------------------------")
print(" Generating violinplot for single removal")
print("------------------------------------------------------------------")
fig, ax = plt.subplots(figsize=(20,10))
pal = sea.cubehelix_palette(8, rot=-.10, dark=.3)
ax.set_xticklabels(df.columns, rotation=45)
ax.set_title('Single feature removal')
ax.set_xlabel('Feature')
ax.set_ylabel('f1-score')
# Show each distribution with both violins and points
sea.violinplot(data=tsint, palette=pal, inner="points")
plt.tight_layout()
os.makedirs('plots/Violinplots/', exist_ok=True)
plt.savefig(f'plots/Violinplots/Single_removal_features.png', dpi=300)

# for feature in matches incremental removal of features
print("------------------------------------------------------------------")
print(" Incremental removal of features")
print("------------------------------------------------------------------")
features = []
x = np.zeros((range_count, 7))
count = 0
df = pd.DataFrame(data=data['data'],
                  columns=data['feature_names'])
droppingFeatures = ""
for column in matches[0:7]:
    droppingFeatures = droppingFeatures + ", " + column[1]
    print("--------------------- Dropping features : " + droppingFeatures)
    removeColumn(df, column[1], count)
    count += 1
    features.append(f'{column[1]}')

incremental_removal_results = pd.DataFrame(data=x,
                       columns=features)

incremental_removal_results.insert(0,'All', all)

print(incremental_removal_results)

tsint = incremental_removal_results.interpolate(method='linear')

print("------------------------------------------------------------------")
print(" Generating violinplot for incremental removal of features")
print("------------------------------------------------------------------")
# analysis of the incremental removal of features
# Use violin plot to get a custom sequential palette
fig, ax = plt.subplots(figsize=(10,10))
pal = sea.cubehelix_palette(8, rot=-.7, dark=.3)
ax.set_xticklabels(df.columns, rotation=45)
ax.set_title('Incremantal feature removal')
ax.set_xlabel('Feature')
ax.set_ylabel('f1-score')
# Show each distribution with both violins and points
sea.violinplot(data=tsint, palette=pal, inner="points")
plt.tight_layout()
os.makedirs('plots/Violinplots/', exist_ok=True)
plt.savefig(f'plots/Violinplots/Incremental_removal_features.png', dpi=300)
plt.close()

# bar plot to show the mean value of each features
mean = tsint.mean()
print(mean)

print("------------------------------------------------------------------")
print("generating histogram plot from mean values")
print("------------------------------------------------------------------")

fig, axes = plt.subplots(1, 1, figsize=(5, 5))
plt.ylim(0.91, 0.97)
axes.set_xticklabels(tsint.columns, rotation=45)
axes.set_title('Mean value of incremantal feature removal')
axes.set_xlabel('Feature')
axes.set_ylabel('f1-score')
plt.tight_layout()
mean.plot.bar()

plt.savefig(f'plots/Histogram_mean_values.png', dpi=300)
plt.close()

print("------------------------------------------------------------------")
print("generating histogram plot for percentage variation")
print("------------------------------------------------------------------")
# getting percent difference between initial and subsequent features
results = []
for iter in range(0, 8):
    results.append(((mean[iter] - mean[0])/2) * 100)

print(results)

rows = ["ALl", "Ash", "Proanthocyanins", "color_intensity", "alcalinity_of_ash", "od315_of_diluted_wines", "alcohol", "malic_acid"]
df=pd.DataFrame(data=results)
df.index = rows
df.columns = ["Percent variation"]
print(df)

fig, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.set_xticklabels(df.index, rotation=45)
df.plot.bar()
plt.title('Percentage variation of incremental feature removal')
plt.xlabel('Feature')
plt.ylabel('Percent variation')
plt.tight_layout()

plt.savefig(f'plots/Histogram_percentage_variation_values.png', dpi=300)

