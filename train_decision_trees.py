import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def train_decision_tree_with_split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = DecisionTreeRegressor(min_samples_leaf=3, min_samples_split=8)
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    print(np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
    print(model.score(x_train, y_train))

    print(model.score(x_test, y_test))
    return model


def train_random_forest_with_split(x, y):
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, y_train = x, y

    model = RandomForestRegressor(n_estimators=1000, max_depth=200, min_samples_split=10, min_samples_leaf=3,
                                  warm_start=True, max_samples=0.9, n_jobs=4)
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    print(np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
    print(model.score(x_train, y_train))

    # print(model.score(x_test, y_test))
    return model
