import numpy as np


def get_HARSVJ_baseline_fit(lags, baseline_fit, ts_norm, idx_range_train):
    harsvj = HARSVJ(lags, baseline_fit)
    harsvj.fit(ts_norm.values[idx_range_train[0] -
               harsvj.max_lag:idx_range_train[1], :])
    return harsvj.coeffs


class HARSVJ(HAR):

    def __init__(self, lags, fit_method='OLS', clip_value=0.0):
        super(HARSVJ, self).__init__(lags, fit_method, clip_value)

    def get_linear_system_fit(self, ts):
        y = ts[self.max_lag:, 0]
        X = self.get_X(ts[:-1, :])
        return X, y

    def get_X(self, ts):
        # X = np.zeros([len(ts[:, 0]) - self.max_lag + 1, np.shape(ts)[1] + len(self.lags)] + 1)
        ts_avgs = []
        for k in range(np.shape(ts)[1] - 1):
            ts_avgs.append(
                np.array([get_avg_ts(ts[:, k], lag) for lag in self.lags]))
        X = np.transpose(np.concatenate(ts_avgs, 0))
        X = np.column_stack((X, ts[:, -1]))
        X = np.c_[np.ones(X.shape[0]), X]
        return X[self.max_lag - 1:, :]
    
class HAR(object):
    def __init__(self, lags, fit_method='OLS', clip_value=0.0):
        self.lags = lags
        self.fit_method = fit_method
        self.clip_value = clip_value

    def get_X(self, ts):
        X = np.zeros([len(ts) - self.max_lag + 1, len(self.lags) + 1])
        ts_avgs = np.array([get_avg_ts(ts, lag) for lag in self.lags])

        for k in range(np.shape(X)[0]):
            X[k, 0] = 1
            X[k, 1:] = ts_avgs[:, k + self.max_lag - 1]
        return X

    def get_linear_system_fit(self, ts):
        y = ts[self.max_lag:, 0]
        X = self.get_X(ts[:-1, 0])
        return X, y

    def fit(self, ts):
        X, y = self.get_linear_system_fit(ts)
        self.lm = LinearRegression(fit_intercept=False).fit(X, y)
        if self.fit_method == 'WLS':
            weights = 1 / self.lm.predict(X)
        elif self.fit_method == 'OLS':
            weights = 1.0
        else:
            logging.warning(f"Baseline fit {self.baseline_fit} unknown. Using weights = 1.0.")
            weights = 1.0
        self.lm = LinearRegression(fit_intercept=False).fit(X, y, sample_weight=weights)

    def predict(self, ts):
        X = self.get_X(ts)
        return np.clip(self.lm.predict(X), a_min=self.clip_value, a_max=None)

    @property
    def max_lag(self):
        return np.max(self.lags)

    @property
    def coeffs(self):
        return self.lm.coef_

    @property
    def is_tf_model(self):
        return False

    def save(self, path):
        with open(path + "/lm.joblib", "w"):
            joblib.dump(self.lm, path + "/lm.joblib")
        np.save(path + "/clipping_value.npy", self.clip_value)

    def restore(self, path):
        with open(path + "/lm.joblib", "r"):
            self.lm = joblib.load(path + "/lm.joblib")
        self.clip_value = np.load(path + "/clipping_value.npy")
