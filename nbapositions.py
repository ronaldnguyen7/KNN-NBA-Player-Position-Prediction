import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Loading and reducing dataset to heavily played players (MPG > 30 Minutes)
df = pd.read_csv('2023 24 NBA Player Statistics.csv')
df = df.drop(df[df['MPG'] <= 24.0].index)

# Dataset contains players who play two positions, (eg. F-G) but order matters so we will take the most played
df['POS'] = df['POS'].str.split('-').str[0]
# print(df['POS'].value_counts())

# Converting the Pandas data frame to a Numpy array to use Sci-kit library
X = df[['MPG', 'USG%', 'FT%', '3PA', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'ORtg', 'DRtg']].values
y = df['POS'].values

# Gives the data zero mean and unit variance
X = preprocessing.StandardScaler().fit(X).transform(X)

def evaluate_knn_model(X, y, k, num_tests = 100):
    train_tests = []
    test_tests = [] 

    for i in range(num_tests):
        '''
        Splitting data set into Training and Testing sets
        80% -> Training
        20% -> Testing
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        # Train Model and Predict  
        neigh = KNeighborsClassifier(n_neighbors = k)
        neigh.fit(X_train,y_train)

        # predicting POS
        yhat = neigh.predict(X_test)

        # Measuring Accuracy Score
        train_tests.append(metrics.accuracy_score(y_train, neigh.predict(X_train)))
        test_tests.append(metrics.accuracy_score(y_test, yhat))

    return np.mean(train_tests), np.mean(test_tests)

# Tests different values of k
ks = range(1,20)
train_means = []
test_means = []
for k in ks:
    train_mean, test_mean = evaluate_knn_model(X, y, k)
    train_means.append(train_mean)
    test_means.append(test_mean)

# Plot results
plt.plot(ks, train_means, label='Train Accuracy')
plt.plot(ks, test_means, label='Test Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. Number of Neighbors')
plt.legend()
plt.show()

# Print final accuracy for best k
best_k = np.argmax(test_means) + 1
print(f'Best k: {best_k} with Test Accuracy: {test_means[best_k - 1]}')