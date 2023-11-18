import pandas as pd
from pycaret.classification import ClassificationExperiment

def load_data(filepath='/home/daboa/Homework/Week_5/prepped_churn_data.csv'):
    df = pd.read_csv('/home/daboa/Homework/Week_5/prepped_churn_data.csv')
    return df


def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    classifier = ClassificationExperiment()
    model = classifier.load_model('/home/daboa/Homework/Week_5/pycaret_model')
    predictions = classifier.predict_model(model, data=df)

    print(predictions.columns)

    predictions.rename({'Label': 'Churn_prediction'}, axis=1, inplace=True)

    return predictions['prediction_score']


if __name__ == "__main__":
    df = load_data('/home/daboa/Homework/Week_5/new_churn_data.csv')
#    df = df.drop(['Unnamed: 0', 'TotalCharges_to_Tenure'], axis=1)
    print(df.columns)
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)