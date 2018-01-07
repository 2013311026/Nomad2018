import logging
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor


import support_functions as sf
import global_flags_constanst as gf

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(gf.LOGGING_LEVEL)


class BaseModel:

    def __init__(self,
                 model_name=None,
                 n_features=None):

        assert isinstance(n_features, int), "n_features must be an integer."


        self.model_name = model_name
        self.n_features = n_features

    def fit(self, x, y):
        """
        A simple linear regression will be our
        base model which in turn will implement
        the over all interface.
        The base model is:
        y = beta*x + epsilon
        :param x:
        :param y:
        :return:
        """

        x = x.T
        n, m = x.shape

        # n + 1 due to the error term epsilon
        self.beta = np.zeros((n + 1, m))

        epsilon = np.ones((1, m))
        xe = np.concatenate((x, epsilon))

        w = np.dot(xe, xe.T)
        w_inv = np.linalg.inv(w)

        self.beta = np.dot(np.dot(w_inv, xe), y.T)

    def predict(self, x):

        x = x.T
        _, m = x.shape
        epsilon = np.ones((1, m))
        xe = np.concatenate((x, epsilon))

        y_pred = np.dot(self.beta.T, xe)

        y_pred = y_pred.reshape(-1, 1)
        #print("y_pred.shape; " + str(y_pred.shape))
        #print(y_pred)
        return y_pred


    def evaluate(self, x, y_true):

        y_pred = self.predict(x)
        y_true = y_true.reshape((-1, 1))

        rmsle = sf.root_mean_squared_logarithmic_error(y_true, y_pred)
        return rmsle

class GBRModel(BaseModel):

    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=20,
                 random_state=1,
                 verbose=0,
                 n_features=None,
                 max_features=None):
        BaseModel.__init__(self, "GradientBoostingClassifierModel", n_features=n_features)
        self.model = GradientBoostingRegressor(n_estimators=n_estimators,
                                               learning_rate=learning_rate,
                                               max_depth=max_depth,
                                               random_state=random_state,
                                               max_features=max_features,
                                               verbose=verbose)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        y_pred = self.model.predict(x)
        y_pred = y_pred.reshape(-1, 1)
        return y_pred


class XGBRegressorModel(BaseModel):

    def __init__(self,
                 n_features=None,
                 max_depth=3,
                 learning_rate=0.1,
                 n_estimators=100,
                 silent=True,
                 objective='reg:linear',
                 booster='gbtree',
                 n_jobs=1,
                 nthread=None,
                 gamma=0,
                 min_child_weight=1,
                 max_delta_step=0,
                 subsample=1,
                 colsample_bytree=1,
                 colsample_bylevel=1,
                 reg_alpha=0,
                 reg_lambda=1,
                 scale_pos_weight=1,
                 base_score=0.5,
                 random_state=0,
                 seed=None,
                 missing=None):

        BaseModel.__init__(self, "XGBRegressor", n_features=n_features)
        self.model = xgb.XGBRegressor(max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      n_estimators=n_estimators,
                                      silent=silent,
                                      objective=objective,
                                      booster=booster,
                                      n_jobs=n_jobs,
                                      nthread=nthread,
                                      gamma=gamma,
                                      min_child_weight=min_child_weight,
                                      max_delta_step=max_delta_step,
                                      subsample=subsample,
                                      colsample_bytree=colsample_bytree,
                                      colsample_bylevel=colsample_bylevel,
                                      reg_alpha=reg_alpha,
                                      reg_lambda=reg_lambda,
                                      scale_pos_weight=scale_pos_weight,
                                      base_score=base_score,
                                      random_state=random_state,
                                      seed=seed,
                                      missing=missing)

    def fit(self, x, y):
        self.model.fit(x, y)

        fi = self.model.feature_importances_
        # logger.info("Number of features in feature_importances_: {0}".format(len(fi)))
        # for i in range(len(fi)):
        #     logger.info("feature_id: {0}; importance {1:.9f}".format(i, fi[i]))
        #xgb.plot_importance(self.model)

    def predict(self, x):
        y_pred = self.model.predict(x)
        y_pred = y_pred.reshape(-1, 1)
        return y_pred