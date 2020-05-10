import matplotlib.pyplot as plt
import pandas as pd
import train_linear_model as tlm
import train_decision_trees as tdc

csv_train_data_path = '../data/train.csv'
csv_test_data_path = '../data/test.csv'


def plot_basic(x, y, feature_label):
    """
    Simple scatter plot of single feature vs target
    :param feature_label: Name of feature
    :param x: Feature
    :param y: Target - Price in usd
    """

    plt.scatter(x, y, 1)
    plt.xlabel(feature_label)
    plt.ylabel('Price $')
    plt.title(feature_label + " vs Price")
    plt.show()


def observation_plots(x, y):
    """
    Basic plots
    :param x: Features
    :param y: Target
    """
    plot_basic(x['GrLivArea'], y, "Living area")
    plot_basic(x['LotArea'], y, "Lot area")

    df_by_neighborhood = df.groupby(['Neighborhood']).mean()
    plot_basic(df_by_neighborhood.iloc[:, 0], df_by_neighborhood['SalePrice'], "Neighborhood")


def clean_features(raw_features):
    """
    Cleans feature df by one-hot encoding of non-numeric features
    :param raw_features: Original features
    :return: updated features with no NaN and text fields to one-hot coding
    """
    updated_features = pd.DataFrame()
    dim_log = [[]]
    for feature in raw_features:
        feat_series: pd.DataFrame = raw_features[feature]
        if not pd.api.types.is_numeric_dtype(feat_series):
            one_hot = pd.get_dummies(feat_series, prefix=feat_series.name)
            dim_log.append({str(one_hot.shape)})
            updated_features = pd.concat([updated_features, one_hot], axis=1)

        else:
            updated_features = pd.concat([updated_features, feat_series], axis=1)

    updated_features = updated_features.fillna(updated_features.mean())
    return updated_features, dim_log


def make_prediction(trn_features, model, df):
    """
    Make prediction on given model for training data
    :param trn_features: train features, to get all values of Enum fields in one-hot coding
    :param model: sklearn model
    :param df: test dataframe
    """
    test_id: pd.Series = df.iloc[:, 0]
    test_features: pd.DataFrame = df.iloc[:, 1:]

    test_features = pd.concat([trn_features, test_features])

    clean_test_features, test_log = clean_features(test_features)
    clean_test_features = clean_test_features.iloc[trn_features.shape[0]:, :]

    prediction = model.predict(clean_test_features)

    result_df = pd.DataFrame(columns=["Id", "SalePrice"])

    for i in range(test_id.shape[0]):
        result_df = result_df.append(pd.DataFrame([[int(test_id.iloc[i]), prediction[i]]], columns=["Id", "SalePrice"]))

    print(result_df)
    result_df.to_csv('../data/result.csv', index=False)


if __name__ == '__main__':
    df = pd.read_csv(csv_train_data_path)
    # df.info()

    train_features: pd.DataFrame = df.iloc[:, 1:-1]
    train_target: pd.Series = df.iloc[:, -1]

    clean_train_features, train_log = clean_features(train_features)

    alp = 0.5
    l1_rat = 0.9
    iters = 1000

    test_df = pd.read_csv(csv_test_data_path)

    """    
    trained_model = tlm.train_model_with_split(clean_train_features, train_target, "elastic", alp, iters, l1_rat)
    
    trained_model = tlm.train_model(clean_train_features, train_target, "elastic", alp, iters, l1_rat)

    make_prediction(train_features, trained_model, test_df)
    """

    """
    trained_ensemble = tlm.train_linear_ensemble(clean_train_features, train_target, alp, iters, 5)

    make_ensemble_prediction(train_features, trained_ensemble, test_df)
    """

    trained_model = tdc.train_random_forest_with_split(clean_train_features, train_target)

    make_prediction(train_features, trained_model, test_df)
