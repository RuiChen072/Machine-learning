import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import seaborn as sns

# import data from 'heart.csv'
data = pd.read_csv('heart.csv', keep_default_na=False, na_values=[""])

# create feature and target variable
X = data.drop(['Heart_Disease', 'Last_Checkup'], axis=1)
y = data['Heart_Disease']

# transfer negative to positive
X['Age'] = X['Age'].apply(lambda x: -x if x < 0 else x)

# make these codings consistent and categorical encoding
mapping_G = {'Male': 0, 'M': 0, 'Female': 1, 'F': 1, 'Unknown': 2}
X['Gender'] = X['Gender'].map(mapping_G)
mapping_S = {'No': 0, 'N': 0, 'Yes': 1, 'Y': 1, 'nan': 2}
X['Smoker'] = X['Smoker'].map(mapping_S)

# split Blood_Pressure
X[['Systolic', 'Diastolic']] = X['Blood_Pressure'].str.split('/', expand=True)

del (X['Blood_Pressure'])

# split into train, test, test_size= =0.3,random_state=2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# imputr value in Age: drop nan, calculate mediean
X_train_without_nan = X_train.dropna(subset=['Age'])
median_age_by_gender = X_train_without_nan.groupby('Gender')['Age'].median()
for index, row in X_test.iterrows():
    if pd.isna(row['Age']):
        gender = row['Gender']
        X_test.at[index, 'Age'] = median_age_by_gender[gender]
# print(X_test[:20])

# Scale Age, Heigh_feet, Weigh_kg, Cholesterol, systolic, diastolic use class MinMaxScaler
Scale_column = ['Age', 'Height_feet', 'Weight_kg', 'Cholesterol', 'Systolic', 'Diastolic']
# normalization
scaler = MinMaxScaler()
X_train[Scale_column] = scaler.fit_transform(X_train[Scale_column])
X_test[Scale_column] = scaler.transform(X_test[Scale_column])
# print(X_train)
# print(X_test)

# plot
plt.hist(y_train)
plt.show()
plt.savefig("Histogram")

# set threshold=0.1
threshold = 0.1
y_train_quantized = (y_train>=threshold).astype(int)
y_test_quantized = (y_test>=threshold).astype(int)
print("original target value(first 20):\n", y_train[:20])
print("original target value(first 20):\n", y_train_quantized[:20])

# Q2(b)
# create C grid
C_value = np.logspace(-4, 4, 100)
# save training and test log-loss
train_losses = []
test_losses = []

# For each C fit into model
for C in C_value:
    # initate use l2 and lbfgs
    model = LogisticRegression(C=C, penalty="l2", solver="lbfgs")
    model.fit(X_train, y_train_quantized)

    # predict probability
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    # compute log-loss
    train_loss = log_loss(y_train_quantized, train_probs)
    test_loss = log_loss(y_test_quantized, test_probs)

    # append into train_losses, test_losses
    train_losses.append(train_loss)
    test_losses.append(test_loss)

# draw curve
plt.figure(figsize=(10, 6))
plt.plot(C_value, train_losses, label='Train Log-loss')
plt.plot(C_value, test_losses, label='Test Log-loss')
plt.xscale('log')
plt.xlabel('C (Regularization Strength)')
plt.ylabel('Log-loss')
plt.title('Train vs. Test Log-loss for different C value(L2 Regularization)')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("Train&Test log-loss")

# select the optimalC
optimal_C = C_value[np.argmin(test_losses)]
print("Optimal C based on test log-loss:", optimal_C)

# Q2(c)
# create C grid
C_value = np.logspace(-4, 4, 100)
# set fold
N = len(X_train)
fold_size = N // 5
fold_X = []
fold_y = []
for i in range(5):
    start = i * fold_size
    end = (i + 1) * fold_size if i < 4 else N
    fold_X.append(X_train[start:end])
    fold_y.append(y_train_quantized[start:end])

# save log-loss of every C
CV_log_loss = []

# perform 5-fold cross validation
for C in C_value:
    losses = []
    for fold in range(5):
        # validation, train set
        val_X = fold_X[fold]
        val_y = fold_y[fold]
        # Concatenate the values in all folds except the current fold
        train_X = np.concatenate([f for idx, f in enumerate(fold_X) if idx != fold])
        train_y = np.concatenate([f for idx, f in enumerate(fold_y) if idx != fold])
        train_X = pd.DataFrame(train_X, columns=X.columns)

        # train the model(l2,lbfgs)
        model = LogisticRegression(C=C, penalty='l2', solver='lbfgs')
        model.fit(train_X, train_y)

        # predict probability, compute log-loss
        val_probas = model.predict_proba(val_X)
        loss = log_loss(val_y, val_probas)
        losses.append(loss)
    CV_log_loss.append(losses)

# box-plot

plt.figure(figsize=(15, 8))
sns.boxplot(data=CV_log_loss)

plt.xscale('log')
plt.xlabel('X(log scale)')
plt.ylabel('Log Loss')
plt.title('5-Fold Cross Validation Log Loss for Different C Value')
plt.tight_layout()
plt.show()
plt.savefig("Boxplot")

# select optimal
mean_loss = [np.mean(losses) for losses in CV_log_loss]
best_C = C_value[np.argmin(mean_loss)]

# re-train model
final_model = LogisticRegression(C=best_C, penalty='l2', solver='lbfgs')
final_model.fit(X_train, y_train_quantized)

# computer accuracy
train_accuracy = final_model.score(X_train, y_train_quantized)
test_accuracy = final_model.score(X_test, y_test_quantized)

print("BestC:", best_C)
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Q2(d)
# use the 5-fold cross-validation
N = len(X_train)
fold_size = N // 5

# set index
test_fold = []
for i in range(5):
    test_fold += [i] * fold_size
test_fold += [4] * (N - 5 * fold_size)
custom_cv = PredefinedSplit(test_fold)

# set grid
Cs = np.logspace(-4, 4, 100)
param_grid = {'C': Cs}
grid_lr = GridSearchCV(estimator=LogisticRegression(penalty='l2', solver='lbfgs'), cv=custom_cv, scoring='neg_log_loss',
                       param_grid=param_grid)
grid_lr.fit(X_train, y_train_quantized)
print("Best C(GridSearchCV):", grid_lr.best_params_['C'])
