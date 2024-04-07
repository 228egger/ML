import numpy as np
from scipy.special import expit
import time


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=100,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        w = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        
        
        def lr_shedule(t, alpha, beta):
            '''
            function for updating learning rate
            t: int
            alpha: float
            beta: float
            '''
            return alpha / (t ** beta)

        
        trace_ = dict()
        trace_time = []
        trace_val = []
        trace_train = []
        np.random.seed(self.random_seed)
        if X_val is None:
            X_val = X.copy()
        if y_val is None:
            y_val = y.copy()
        if w_0 is None:
            w_0 = np.random.normal(size = X.shape[1])
            w_0 = np.hstack((1, w_0))
        w = w_0.copy()
        begin = time.time()
        for i in range(1, self.max_iter + 1):
            lr = lr_shedule(i, self.step_alpha, self.step_beta)
            loss_prev = self.loss_function.func(X = X, y = y, w = w)
            if self.batch_size >= X.shape[0]:
                w -=  self.loss_function.grad(X, y, w) * lr 
            else:
                indeces = np.arange(X.shape[0])
                np.random.shuffle(indeces)
                cur_pos = 0
                for j in range(X.shape[0] // self.batch_size):
                    mask = indeces[cur_pos: cur_pos + min(self.batch_size, X.shape[0] - cur_pos)]
                    X_sample = X[mask]
                    y_sample = y[mask]
                    w -=  lr * self.loss_function.grad(X_sample, y_sample, w)
                    cur_pos += self.batch_size
            new_loss = self.loss_function.func(X, y, w)
            if abs(new_loss - loss_prev) < self.tolerance:
                print(f'Early stop on {i} epoch')
                break
            loss_prev = new_loss
            end = time.time()
            if trace:
                trace_time.append(end - begin)
                trace_val.append(self.loss_function.func(X_val, y_val, w))
                trace_train.append(new_loss)
        self.w = w.copy()
        if trace:
            trace_['time'] = trace_time
            trace_['val'] = trace_val
            trace_['train'] = trace_train
            return trace_
        

    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        proba_ = expit(X @ self.w.T)
        pred = np.ones(X.shape[0])
        pred[proba_ > threshold] = -1
        return pred
        

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.w[1:]

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        return self.loss_function.loss_without_reg(X, y, self.w)

    def get_bias(self):
        """
        Get model bias

        Returns
        -------
        : float
            model bias
        """
        return self.w[0]
