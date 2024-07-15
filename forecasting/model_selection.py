# Functions for train/test splits & cross validation
# We can output summary tables for key forecasting metrics (i.e., MAPE/MAE/...)

def cross_val(data, n_splits, test_size, model=None):
    
    """
    Cross validation (k-fold)
    Inputs:
        data (pandas.DataFrame): Univariate time series dataset.
        n_splits (int): Number of folds.
        test_size (int): Forecast horizon during each fold.
        model: Time series model to evaluate.
    Outputs:
        Cross validation summary (pandas.DataFrame)
    """
    
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    cross_val = tscv.split(data)
    
    cv_summary={}
    for fold, (train, test) in enumerate(cross_val):

        cv_output = data.copy().iloc[test].reset_index(drop=True)
        # Fake forecast
        cv_output['forecast'] = 5
        cv_summary[fold] = cv_output
    
    return pd.concat(cv_summary)