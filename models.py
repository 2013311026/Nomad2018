import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

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

        #print("y_pred.shape; " + str(y_pred.shape))
        #print(y_pred)
        return y_pred


    def evaluate(self, x, y_true):

        y_pred = self.predict(x)

        y_true = y_true.reshape((1, -1))
        y_pred = y_pred.reshape((1, -1))

        lpi = np.log(y_pred + 1.0)
        lai = np.log(y_true + 1.0)
        s2 = (lpi - lai)*(lpi - lai)

        # Root mean squared logaritmic error
        rmsle = np.sqrt((1.0/self.n_features)*np.sum(s2))

        return rmsle

class GBRModel(BaseModel):

    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=10,
                 random_state=1,
                 verbose=0,
                 n_features=None):
        BaseModel.__init__(self, "GradientBoostingClassifierModel", n_features=n_features)
        self.model = GradientBoostingRegressor(n_estimators=n_estimators,
                                               learning_rate=learning_rate,
                                               max_depth=max_depth,
                                               random_state=random_state,
                                               verbose=verbose)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred