import argparse
from sklearn.model_selection import train_test_split
from data_processor import DataProcessor
from model_estimator import LGBM, LassoEst


def parse_input_arguments():
    """Parses input arguments for model training.

    Returns:
        A tuple containing the filename with data for model training,
        the name of the target variable to model and length for test data.

    Raises:
        ValueError: If the inputs are not provided.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--n_ahead',
        dest='n_ahead',
        action='store',
        type=int,
        default=None,
        help='Number of hours to predict ahead.')

    parser.add_argument(
        '--dayhour',
        dest='dayhour',
        action='store',
        type=int,
        default=None,
        help='Hour of day to predict.')
    parser.add_argument(
        '--model',
        dest='model_type',
        action='store',
        type=str,
        default=None,
        help='Model type name. Only \'LGBM\' and \'Lasso\' are available.')
    args = parser.parse_args()

    if args.n_ahead is None or args.dayhour is None or args.model_type is None:
        raise ValueError('Missing input argument!'
            'Make sure to include --n_ahead and --dayhour --model')

    return args.n_ahead, args.dayhour, args.model_type


if __name__=='__main__':
    # Parsing input arguments
    n_ahead, dayhour, model_type = parse_input_arguments()
    # Data Load
    data_processor = DataProcessor.from_csv()
    X, y = data_processor.dayhour_data(n_ahead, dayhour)
    # Data split
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, train_size=0.9, shuffle=False)
    # Fitting model
    if model_type=='LGBM':
        model = LGBM().fit(X_train, y_train)
    elif model_type=='Lasso':
        model = LassoEst().fit(X_train, y_train)
    predicted = model.predict(X_test)
    # Results
    model.fitted_model_plot(y_test)
    model.evaluate(y_test, y_test.shift(1).bfill())
