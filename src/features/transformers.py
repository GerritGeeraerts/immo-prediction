class OutlierRemover(BaseEstimator ,TransformerMixin):
    def __init__(self ,factor=1.5):
        self.factor = factor

    def outlier_removal(self ,X ,y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (self.factor * iqr)
        upper_bound = q3 + (self.factor * iqr)
        X.loc[((X < lower_bound) | (X > upper_bound))] = np.nan
        return pd.Series(X)

    def fit(self ,X ,y=None):
        return self

    def transform(self ,X ,y=None):
        return X.apply(self.outlier_removal)

