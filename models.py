import logging
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor

from keras.layers import Input, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import optimizers
from keras import callbacks


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
                 n_features=None,
                 validation_data=None):

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

        n, m = y_true.shape

        if m == 1:
            y_pred = self.predict(x)
            y_true = y_true.reshape((-1, 1))

            logger.info("Example predictions:")

            if n == 1:
                # number of example to print
                noetp = 1
            elif n > 5:
                noetp = 5
            else:
                noetp = 0

            for i in range(noetp):
                logger.info("y_pred: {0}; y_true: {1}".format(y_pred[i], y_true[i]))

            rmsle = sf.root_mean_squared_logarithmic_error(y_true, y_pred)
        elif m == 2:
            y_pred = self.predict(x)
            y_true = y_true.reshape((-1, m))

            y_pred_0 = y_pred[:, 0].reshape((-1, 1))
            y_pred_1 = y_pred[:, 1].reshape((-1, 1))

            y_true_0 = y_true[:, 0].reshape((-1, 1))
            y_true_1 = y_true[:, 1].reshape((-1, 1))

            rmsle_0 = sf.root_mean_squared_logarithmic_error(y_true_0, y_pred_0)
            rmsle_1 = sf.root_mean_squared_logarithmic_error(y_true_1, y_pred_1)

            rmsle = (rmsle_0 + rmsle_1)/2.0
        else:
            pass

        return rmsle

class GBRModel(BaseModel):

    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=20,
                 random_state=1,
                 verbose=0,
                 n_features=None,
                 max_features=None,
                 validation_data=None):
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
                 missing=None,
                 validation_data=None):

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


class FeedForwardNeuralNetworkModel(BaseModel):

    def __init__(self,
                 n_features=10,
                 n_hidden_layers=2,
                 n_output=2,
                 layer_dim=10,
                 dropout_rate=0.9,
                 alpha=0.1,
                 learning_rate=0.1,
                 loss="mean_squared_error",
                 validation_data=None):

        BaseModel.__init__(self, "FeedForwardNeuralNetwork", n_features=n_features)

        self.n_features = n_features
        self.n_hidden_layers = n_hidden_layers
        self.n_output = n_output
        self.layer_dim = layer_dim
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.learning_rate=learning_rate
        self.loss = loss
        self.validation_data = validation_data

        self.input_features = Input(shape=(self.n_features,), name="autoencoder_input")

        layer = Dense(layer_dim, activation="linear", name="encoder_first_layer")(self.input_features)
        layer = LeakyReLU(alpha=alpha)(layer)
        layer = Dropout(dropout_rate, name="encoder_first_layer_dropout")(layer)

        for i in range(n_hidden_layers):
            layer = Dense(layer_dim, activation="linear", name="encoder_hidden_layer_{0}".format(i))(layer)
            layer = LeakyReLU(alpha=alpha)(layer)
            layer = Dropout(dropout_rate, name="encoder_layer_dropout_{0}".format(i))(layer)

        self.output = Dense(self.n_output, activation='softmax')(layer)
        self.model = Model(self.input_features, self.output)

        opt = optimizers.Adam(lr=self.learning_rate)
        self.model.compile(optimizer=opt,
                           loss=loss)

        self.model.summary()

    def fit(self, x, y):

        if self.validation_data == None:
            self.model.fit(x, y,
                           epochs=100,
                           batch_size=128,
                           shuffle=True,
                           verbose=1)
        else:
            self.model.fit(x, y,
                           epochs=100,
                           batch_size=256,
                           shuffle=True,
                           verbose=1,
                           validation_data=self.validation_data)

    def predict(self, x):

        y_pred = self.model.predict(x)
        return y_pred