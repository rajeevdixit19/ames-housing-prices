import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils.random import sample_without_replacement
from data_prep import clean_features


def train_model_with_split(x, y, model_name, alpha, max_iter, l1_ratio):
    """
    Returns model trained on 80% train:test split data
    :param x: features df
    :param y: target df
    :param model_name: linreg / ridge / lasso / elastic
    :param alpha: normalization factor
    :param max_iter: Max iterations
    :param l1_ratio: for elasticnet - l1 normalization ratio
    :return: trained sklearn.linear_model object
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    if model_name == "linreg":
        model: LinearRegression = LinearRegression().fit(x_train, y_train)

    elif model_name == "ridge":
        model: Ridge = Ridge(alpha=alpha, normalize=True, max_iter=max_iter).fit(x_train, y_train)

    elif model_name == "lasso":
        model: Lasso = Lasso(alpha=alpha, normalize=True, max_iter=max_iter).fit(x_train, y_train)

    elif model_name == "elastic":
        model: ElasticNet = ElasticNet(alpha=alpha, normalize=True, max_iter=max_iter, l1_ratio=l1_ratio).fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    print(np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

    y_test_pred = model.predict(x_test)
    print(np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

    return model


def train_model(x, y, model_name, alpha, max_iter, l1_ratio):
    """
    Trains linear model
    :param x: features df
    :param y: target df
    :param model_name: linreg / ridge / lasso / elastic
    :param alpha: normalization factor
    :param max_iter: Max iterations
    :param l1_ratio: for elasticnet - l1 normalization ratio
    :return: trained sklearn.linear_model object
    """
    if model_name == "linreg":
        model: LinearRegression = LinearRegression().fit(x, y)

    elif model_name == "ridge":
        model: Ridge = Ridge(alpha=alpha, normalize=True, max_iter=max_iter).fit(x, y)

    elif model_name == "lasso":
        model: Lasso = Lasso(alpha=alpha, normalize=True, max_iter=max_iter).fit(x, y)

    elif model_name == "elastic":
        model: ElasticNet = ElasticNet(alpha=alpha, normalize=True, max_iter=max_iter, l1_ratio=l1_ratio).fit(x, y)

    y_pred = model.predict(x)
    print(np.sqrt(metrics.mean_squared_error(y, y_pred)))
    return model


def train_linear_ensemble(x, y, alpha, max_iter, n_ensembles):
    """
    Train ensemble of linear models
    :param x: features df
    :param y: target df
    :param alpha: normalization factor
    :param max_iter: Max iterations
    :param n_ensembles: Number of ensembles
    :return: List of sklearn Ridge regression models
    """
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, y_train = x, y
    ensemble_models = []
    for i in range(n_ensembles):
        samples = sample_without_replacement(n_population=x_train.shape[0], n_samples=(x_train.shape[0]/5))
        x_seg_train = pd.DataFrame()
        y_seg_train = pd.Series()
        for sample in samples:
            x_seg_train = pd.concat([x_seg_train, x_train.iloc[[sample]]])
            y_seg_train = pd.concat([y_seg_train, y_train.iloc[[sample]]])

        model: Ridge = Ridge(alpha=alpha, normalize=True, max_iter=max_iter).fit(x_seg_train, y_seg_train)
        print(model.score(x_seg_train, y_seg_train))
        # print(model.score(x_test, y_test))
        ensemble_models.append(model)

    return ensemble_models


def make_ensemble_prediction(trn_features, models, test_df):
    """
    Predicts output by averaging output of individual models
    :param trn_features: train features, to get all values of Enum fields in one-hot coding
    :param models: List of sklearn models
    :param test_df: test dataframe
    """
    test_id: pd.Series = test_df.iloc[:, 0]
    test_features: pd.DataFrame = test_df.iloc[:, 1:]

    test_features = pd.concat([trn_features, test_features])

    clean_test_features, test_log = clean_features(test_features)
    clean_test_features = clean_test_features.iloc[trn_features.shape[0]:, :]

    predictions = np.empty([0, trn_features.shape[0] - 1])
    for model in models:
        prediction = np.array([model.predict(clean_test_features)])
        predictions = np.vstack([predictions, prediction])

    prediction = np.mean(predictions, axis=0)

    # print(predictions, prediction)
    result_df = pd.DataFrame(columns=["Id", "SalePrice"])

    for i in range(test_id.shape[0]):
        result_df = result_df.append(pd.DataFrame([[int(test_id.iloc[i]), prediction[i]]], columns=["Id", "SalePrice"]))

    print(result_df)
    result_df.to_csv('../data/result.csv', index=False)
