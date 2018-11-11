import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm


class ISimpleAlsModel(BaseEstimator):

    def describe(self):
        res = 'my_simple_als_f-{0}_ep-{1}_alpha-{2}_eps-{3}_lr-{4}_cfunc-{5}_test_impl-{6}_smean-{7}'
        res = res.format(self.n_factors,
                         self.n_epochs,
                         self.alpha,
                         self.eps,
                         self.lr,
                         self.c_function,
                         self.use_test_as_implicit,
                         self.substract_mean)
        return res

    def __init__(self,
                 n_factors = 4,
                 n_epochs = 20,
                 alpha = 5,
                 eps = 0.01,
                 init_mean = 0,
                 init_std_dev= 0.1,
                 lr = 0.07,
                 random_state = None,
                 verbose = False,
                 c_function = 'linear',
                 use_test_as_implicit = True,
                 substract_mean = True,
                 mean_decrease = 0.85):

        self.n_factors =  n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr = lr
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = alpha
        self.eps = eps
        self.use_test_as_implicit = use_test_as_implicit
        self.substract_mean = substract_mean
        self.c_function = c_function
        self.mean_decrease = mean_decrease

        if c_function is 'linear':
            self.confidense_func = self._confidense_linear
        if c_function is 'log':
            self.confidense_func = self._confidense_log


        pass

    def get_confidense_func(self):
        if self.c_function is 'linear':
            return  self._confidense_linear
        if self.c_function is 'log':
            return  self._confidense_log
        return self._confidense_linear

    @staticmethod
    def _confidense_linear(R, alpha, eps):
        C = np.ones(R.shape) + alpha*R
        return C

    @staticmethod
    def _confidense_log(R, alpha, eps):
        C = np.ones(R.shape) + alpha * np.log(np.ones(R.shape) + R / eps)
        return C

    def _init_X(self, users_count):
        X = np.empty((users_count, self.n_factors))
        X = self.init_std_dev * np.random.randn(users_count, self.n_factors)
        return X

    def _init_Y(self, items_count):
        Y = np.empty((items_count, self.n_factors))
        Y = self.init_std_dev * np.random.randn(items_count, self.n_factors)
        return Y

    def fit(self, train_data, y_train, test_data, shape):
        R_train  = np.zeros(shape)
        R_train[train_data[:, 0].astype(int), train_data[:, 1].astype(int)] = y_train

        R_test = np.zeros(shape)
        R_test[test_data[:, 0].astype(int), test_data[:, 1].astype(int)] = 1

        users_count = R_train.shape[0]
        items_count = R_train.shape[1]

        not_zero_mask = (R_train > 0)
        zero_mask = (R_train == 0)
        implicit_mask = (R_test > 0)

        #R_test[implicit_mask] = self.mean_decrease


        R_full = R_train +  R_test


        full_mean = R_train[not_zero_mask].mean()

        if self.substract_mean:
            P = R_train - not_zero_mask.astype(np.int) * full_mean
        else:
            P = R_train# + zero_mask.astype(int) * full_mean

        #P += zero_mask.astype(int) * self.mean_decrease
        # P[implicit_mask] = self.mean_decrease

        self.confidense_func = self.get_confidense_func()
        if self.use_test_as_implicit:
            C = self.confidense_func(R_full, self.alpha, self.eps)
        else:
            C = self.confidense_func(R_train, self.alpha, self.eps)
        C[not_zero_mask] = 1.0
        C[implicit_mask] = 0.8

        #C[zero_mask] += self.mean_decrease

        X = self._init_X(users_count)
        Y = self._init_Y(items_count)

        lrmat = self.lr*np.eye(self.n_factors, self.n_factors)

        if self.verbose:
            epochs_range = tqdm(range(self.n_epochs))
        else:
            epochs_range = range(self.n_epochs)

        for num_epoch in epochs_range:

            # user step

            YtY = np.matmul(Y.T, Y)
            Cp = C * P
            for u in range(users_count):
                to_inv = YtY + np.matmul(Y.T * (C[u, :] - 1), Y) + lrmat
                inv_mat = np.linalg.inv(to_inv)
                #_a = np.matmul(Y.T * C[u, :], Pgamma[u, :])
                _a = np.matmul(Y.T, Cp[u, :])
                X[u, :] = np.matmul(inv_mat, _a)



            # item step

            XtX = np.matmul(X.T, X)
            Cp = C * P
            for i in range(items_count):
                to_inv = XtX + np.matmul(X.T*(C[:, i] - 1), X) + lrmat
                inv_mat = np.linalg.inv(to_inv)
                #_a = np.matmul(X.T * C[:, i] , Pbeta[:, i])
                _a = np.matmul(X.T, Cp[:, i])
                Y[i, :] = np.matmul(inv_mat, _a)



        self.X = X
        self.Y = Y
        self.full_mean = full_mean

        pass

    def estimate(self, u, i):
        score = 0
        score += self.full_mean * self.substract_mean
        score += np.matmul(self.X[u],self.Y[i])
        return score

    def predict(self, test_data):
        score = 0
        score += self.full_mean * self.substract_mean
        score += (self.X[test_data[:, 0].astype(int)] *
                           self.Y[test_data[:, 1].astype(int)]).sum(1)
        return score

    def score(self, test_data, y_test):
        y_pred = self.predict(test_data)
        return np.sqrt(np.mean((y_test - y_pred)**2))




class IBiasedAlsModel(BaseEstimator):

    def model_name(self):
        return 'my_IBiasedAlsModel'

    def describe(self):
        res = 'my_ials_f-{0}_ep-{1}_alpha-{2}_eps-{3}_lr-{4}_cfunc-{5}_test_impl-{6}_smean-{7}'
        res = res.format(self.n_factors,
                         self.n_epochs,
                         self.alpha,
                         self.eps,
                         self.lr,
                         self.c_function,
                         self.use_test_as_implicit,
                         self.substract_mean)
        return res

    def __init__(self,
                 n_factors = 4,
                 n_epochs = 20,
                 alpha = 5,
                 eps = 0.01,
                 init_mean = 0,
                 init_std_dev= 0.1,
                 lr = 0.07,
                 random_state = None,
                 verbose = False,
                 c_function = 'linear',
                 use_test_as_implicit = True,
                 substract_mean = True,
                 mean_decrease = 0.85):

        self.n_factors =  n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr = lr
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = alpha
        self.eps = eps
        self.use_test_as_implicit = use_test_as_implicit
        self.substract_mean = substract_mean
        self.c_function = c_function
        self.mean_decrease = mean_decrease

        if c_function is 'linear':
            self.confidense_func = self._confidense_linear
        if c_function is 'log':
            self.confidense_func = self._confidense_log


        pass

    def get_confidense_func(self):
        if self.c_function is 'linear':
            return  self._confidense_linear
        if self.c_function is 'log':
            return  self._confidense_log
        return self._confidense_linear

    @staticmethod
    def _confidense_linear(R, alpha, eps):
        C = np.ones(R.shape) + alpha*R
        return C

    @staticmethod
    def _confidense_log(R, alpha, eps):
        C = np.ones(R.shape) + alpha * np.log(np.ones(R.shape) + R / eps)
        return C

    def _init_X(self, users_count):
        X = np.empty((users_count, 1 + self.n_factors))
        X[:, 0] = np.ones(users_count)
        X[:, 1:] = self.init_std_dev * np.random.randn(users_count, self.n_factors)
        return X

    def _init_Y(self, items_count):
        Y = np.empty((items_count, 1 + self.n_factors))
        Y[:, 0] = np.ones(items_count)
        Y[:, 1:] = self.init_std_dev * np.random.randn(items_count, self.n_factors)
        return Y

    def fit(self, train_data, y_train, test_data, shape, y_test = None, verbose = -1):
        R_train  = np.zeros(shape)
        R_train[train_data[:, 0].astype(int), train_data[:, 1].astype(int)] = y_train

        R_test = np.zeros(shape)
        R_test[test_data[:, 0].astype(int), test_data[:, 1].astype(int)] = 1

        users_count = R_train.shape[0]
        items_count = R_train.shape[1]

        not_zero_mask = (R_train > 0)
        zero_mask = (R_train == 0)
        implicit_mask = (R_test > 0)

        #R_test[implicit_mask] = self.mean_decrease


        R_full = R_train + self.mean_decrease * R_test


        full_mean = R_train[not_zero_mask].mean()
        full_std = R_train[not_zero_mask].std()

        # if self.substract_mean:
        #     P = (R_train - not_zero_mask.astype(np.int) * full_mean)
        # else:
        #     P = R_train # + zero_mask.astype(int) * full_mean



        #P += zero_mask.astype(int) * self.mean_decrease

        self.confidense_func = self.get_confidense_func()
        if self.use_test_as_implicit:
            C = self.confidense_func(R_full, self.alpha, self.eps)
        else:
            C = self.confidense_func(R_train, self.alpha, self.eps)

        # C = np.zeros_like(C)
        # C[not_zero_mask] = 1.0
        # C[implicit_mask] = 0.9
        #
        # P[implicit_mask] = 0.0

        #C[zero_mask] += self.mean_decrease

        X = self._init_X(users_count)
        Y = self._init_Y(items_count)

        user_bias = np.repeat(0.0, users_count)
        # for u in range(users_count):
        #     d = R_train[u]
        #     if d[d>0].shape[0] == 0:
        #         user_bias[u] = 0.0
        #     else:
        #         user_bias[u] = d.mean()

        item_bias = np.repeat(0.0, items_count)
        # for i in range(items_count):
        #     d = R_train[:, i]
        #     if d[d>0].shape[0] == 0:
        #         item_bias[i] = 0.0
        #     else:
        #         item_bias[i] = d[d>0].mean()

        # P = R_train + 0.85 * (zero_mask & ~implicit_mask) * ( full_mean + user_bias[:, None] + item_bias[None, :]) + \
        #     implicit_mask * (full_mean + user_bias[:, None] + item_bias[None, :])
        P = R_train - not_zero_mask*full_mean - not_zero_mask*user_bias[:, None] - not_zero_mask*item_bias[None, :]

        lrmat = self.lr*np.eye(self.n_factors + 1, self.n_factors + 1)

        if self.verbose:
            epochs_range = tqdm(range(self.n_epochs))
        else:
            epochs_range = range(self.n_epochs)


        train_scores = []
        test_scores = []

        for num_epoch in epochs_range:

            Pbeta = P - user_bias[:, None]
            Pgamma = P - item_bias[None, :]

            # user step

            Y[:, 0] = np.ones(items_count)
            X[:, 0] = user_bias

            YtY = np.matmul(Y.T, Y)
            Cp = C * Pgamma
            for u in range(users_count):
                to_inv = YtY + np.matmul(Y.T * (C[u, :] - 1), Y) + lrmat
                inv_mat = np.linalg.inv(to_inv)
                _a = np.matmul(Y.T * C[u, :], Pgamma[u, :])
                #_a = np.matmul(Y.T, Cp[u, :])
                X[u, :] = np.matmul(inv_mat, _a)

            # update user bias
            user_bias = np.copy(X[:, 0])


            # item step

            X[:, 0] = np.ones(users_count)
            Y[:, 0] = item_bias

            XtX = np.matmul(X.T, X)
            Cp = C * Pbeta
            for i in range(items_count):
                to_inv = XtX + np.matmul(X.T*(C[:, i] - 1), X) + lrmat
                inv_mat = np.linalg.inv(to_inv)
                _a = np.matmul(X.T * C[:, i] , Pbeta[:, i])
                #_a = np.matmul(X.T, Cp[:, i])
                Y[i, :] = np.matmul(inv_mat, _a)

            #update item bias
            item_bias = np.copy(Y[:, 0])



            self.user_bias = user_bias
            self.item_bias = item_bias
            self.X = X[:, 1:]
            self.Y = Y[:, 1:]
            self.full_mean = full_mean
            self.full_std = full_std

            if y_test is not None:
                test_score = self.score(test_data, y_test)
                train_score = self.score(train_data, y_train)
                train_scores.append(train_score)
                test_scores.append(test_score)

                if verbose > 0 and num_epoch%verbose == 0 or num_epoch == self.n_epochs-1:
                    print('iter: {0} train_score: {1}   test_score: {2}'.format(num_epoch, train_score, test_score))

        return np.array(train_scores), np.array(test_scores)

    def estimate(self, u, i):
        score = 0
        score += self.full_mean
        score += self.user_bias[u] + self.item_bias[i]
        score += np.matmul(self.X[u],self.Y[i])
        return score

    def predict(self, test_data):
        score = 0
        score += self.full_mean
        score += self.user_bias[test_data[:, 0].astype(int)]
        score += self.item_bias[test_data[:, 1].astype(int)]
        score += (self.X[test_data[:, 0].astype(int)] *
                          self.Y[test_data[:, 1].astype(int)]).sum(1)
        return score

    def score(self, test_data, y_test):
        y_pred = self.predict(test_data)
        return np.sqrt(np.mean((y_test - y_pred)**2))


class ExplicitCF(BaseEstimator):

    def model_name(self):
        return 'my_explicitCF'

    def describe(self):
        res = 'my_explicitCF_f-{0}_ep-{1}_alpha-{2}_eps-{3}_lr-{4}_cfunc-{5}_test_impl-{6}_smean-{7}'
        res = res.format(self.n_factors,
                         self.n_epochs,
                         self.alpha,
                         self.eps,
                         self.reg_emb,
                         self.reg_bias,
                         self.c_function,
                         self.use_test_as_implicit,
                         self.substract_mean)
        return res

    def __init__(self,
                 n_factors = 4,
                 n_epochs = 20,
                 alpha = 5,
                 eps = 0.01,
                 init_mean = 0,
                 init_std_dev= 0.1,
                 reg_emb = 0.07,
                 reg_bias = 0.01,
                 random_state = None,
                 verbose = False,
                 c_function = 'linear',
                 use_test_as_implicit = True,
                 substract_mean = True,
                 mean_decrease = 0.85):

        self.n_factors =  n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.reg_emb = reg_emb
        self.reg_bias = reg_bias
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = alpha
        self.eps = eps
        self.use_test_as_implicit = use_test_as_implicit
        self.substract_mean = substract_mean
        self.c_function = c_function
        self.mean_decrease = mean_decrease

        # if c_function is 'linear':
        #     self.confidense_func = self._confidense_linear
        # if c_function is 'log':
        #     self.confidense_func = self._confidense_log


        pass

    # def get_confidense_func(self):
    #     if self.c_function is 'linear':
    #         return  self._confidense_linear
    #     if self.c_function is 'log':
    #         return  self._confidense_log
    #     return self._confidense_linear

    @staticmethod
    def _confidense_linear(R, alpha, eps):
        C = np.ones(R.shape) + alpha*R
        return C

    @staticmethod
    def _confidense_log(R, alpha, eps):
        C =  np.log(np.ones(R.shape) + R / eps)
        return C

    def _init_X(self, users_count):
        X = np.empty((users_count, self.n_factors))
        X = self.init_std_dev * np.random.randn(users_count, self.n_factors)
        return X

    def _init_Y(self, items_count):
        Y = np.empty((items_count, self.n_factors))
        Y = self.init_std_dev * np.random.randn(items_count, self.n_factors)
        return Y

    def fit(self, train_data, y_train, test_data, shape, y_test = None, verbose = -1):
        R_train  = np.zeros(shape)
        R_train[train_data[:, 0].astype(int), train_data[:, 1].astype(int)] = y_train

        R_test = np.zeros(shape)
        R_test[test_data[:, 0].astype(int), test_data[:, 1].astype(int)] = 1

        users_count = R_train.shape[0]
        items_count = R_train.shape[1]



        not_zero_mask = (R_train > 0)
        zero_mask = (R_train == 0)
        implicit_mask = (R_test > 0)



        full_mean = R_train[not_zero_mask].mean()

        X = self._init_X(users_count)
        Y = self._init_Y(items_count)

        #init biases
        # user_bias = np.true_divide(R_train.sum(1),1+(R_train!=0).sum(1))#R_train.mean(1)
        # item_bias = np.true_divide(R_train.sum(0),1+(R_train!=0).sum(0))

        user_bias = np.repeat(0.0, users_count)
        for u in range(users_count):
            d = R_train[u]
            if d[d>0].shape[0] == 0:
                user_bias[u] = 0.0
            else:
                user_bias[u] = d.mean()

        item_bias = np.repeat(0.0, items_count)
        for i in range(items_count):
            d = R_train[:, i]
            if d[d>0].shape[0] == 0:
                item_bias[i] = 0.0
            else:
                item_bias[i] = d[d>0].mean()

        lrmat = self.reg_emb * np.eye(self.n_factors, self.n_factors)


        P = R_train
        #P = R_train - full_mean #+ (implicit_mask)*(user_bias[ :, None] + item_bias[None, :])

        W = not_zero_mask
        W = W.astype(int)
        #
        # C = self._confidense_linear(R_train + R_test, self.alpha, self.eps)

        if self.verbose:
            epochs_range = tqdm(range(self.n_epochs))
        else:
            epochs_range = range(self.n_epochs)


        train_scores = []
        test_scores = []

        for num_epoch in epochs_range:

            # if num_epoch%10== 0 and num_epoch > 0:
            #     self.reg_emb = self.reg_emb * 1.2
            #     self.reg_bias = self.reg_bias * 1.2

            # user step
            YtY = np.matmul(Y.T, Y)
            Puser = W*(P  - full_mean - item_bias[None, :] - user_bias[:, None])
            yt_inv = np.linalg.inv(YtY + lrmat)
            for u in range(users_count):
                _a = np.matmul(Y.T, Puser[u, :])
                X[u, :] = np.matmul(yt_inv, _a)


            # item step
            XtX = np.matmul(X.T, X)
            Pitem = W*(P  - full_mean - user_bias[:, None] - item_bias[None, :])
            xt_inv = np.linalg.inv(XtX  + lrmat)
            for i in range(items_count):
                _a = np.matmul(X.T, Pitem[:,i])
                Y[i, :] = np.matmul(xt_inv, _a)

            #bias step
            # for u in range(users_count):
            #     user_bias[u] = 1/self.lr * (C[u,:] * (P[u, :]  - X[u, :].dot(Y.T))).mean()
            user_bias = 1.0 / (W.sum(1) + self.reg_bias) * (W*(P - item_bias[None, :] - full_mean - np.matmul(X, Y.T))).sum(1)
            item_bias = 1.0 / (W.sum(0) + self.reg_bias) * (W*(P - user_bias[:, None] - full_mean - np.matmul(X, Y.T))).sum(0)

            self.user_bias = user_bias
            self.item_bias = item_bias
            self.X = X
            self.Y = Y
            self.full_mean = full_mean

            if y_test is not None:
                test_score = self.score(test_data, y_test)
                train_score = self.score(train_data, y_train)
                train_scores.append(train_score)
                test_scores.append(test_score)

                if verbose > 0 and num_epoch%verbose == 0 or num_epoch == self.n_epochs-1:
                    print('iter: {0} train_score: {1}   test_score: {2}'.format(num_epoch, train_score, test_score))

        return np.array(train_scores), np.array(test_scores)

    def estimate(self, u, i):
        score = 0
        score += self.full_mean * self.substract_mean
        score += self.user_bias[u] + self.item_bias[i]
        score += np.matmul(self.X[u],self.Y[i])
        if score > 5:
            return 5
        if score < 1:
            return 1

        return score

    def predict(self, test_data):
        score = np.zeros(test_data.shape[0])
        score += self.full_mean * self.substract_mean
        score += self.user_bias[test_data[:, 0].astype(int)]
        score += self.item_bias[test_data[:, 1].astype(int)]
        score += (self.X[test_data[:, 0].astype(int)] *
                          self.Y[test_data[:, 1].astype(int)]).sum(1)

        score[score > 5] = 5
        score[score < 1] = 1
        return score

    def score(self, test_data, y_test):
        y_pred = self.predict(test_data)
        return np.sqrt(np.mean((y_test - y_pred)**2))


class ImplicitCF(BaseEstimator):

    def model_name(self):
        return 'my_implicitCF'

    def describe(self):
        res = 'my_implicitCF_f-{0}_ep-{1}_alpha-{2}_eps-{3}_lr-{4}_cfunc-{5}_test_impl-{6}_smean-{7}'
        res = res.format(self.n_factors,
                         self.n_epochs,
                         self.alpha,
                         self.eps,
                         self.reg_emb,
                         self.reg_bias,
                         self.c_function,
                         self.use_test_as_implicit,
                         self.substract_mean)
        return res

    def __init__(self,
                 n_factors = 4,
                 n_epochs = 20,
                 alpha = 5,
                 eps = 0.01,
                 init_mean = 0,
                 init_std_dev= 0.1,
                 reg_emb = 0.07,
                 reg_bias = 0.01,
                 random_state = None,
                 verbose = False,
                 c_function = 'linear',
                 use_test_as_implicit = True,
                 substract_mean = True,
                 mean_decrease = 0.85):

        self.n_factors =  n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.reg_emb = reg_emb
        self.reg_bias = reg_bias
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = alpha
        self.eps = eps
        self.use_test_as_implicit = use_test_as_implicit
        self.substract_mean = substract_mean
        self.c_function = c_function
        self.mean_decrease = mean_decrease

        if c_function is 'linear':
            self.confidense_func = self._confidense_linear
        if c_function is 'log':
            self.confidense_func = self._confidense_log

        np.random.seed(self.random_state)


        pass

    def get_confidense_func(self):
        if self.c_function is 'linear':
            return  self._confidense_linear
        if self.c_function is 'log':
            return  self._confidense_log
        return self._confidense_linear

    @staticmethod
    def _confidense_linear(R, alpha, eps):
        C = np.ones(R.shape) + alpha*R
        return C

    @staticmethod
    def _confidense_log(R, alpha, eps):
        C =  np.log(np.ones(R.shape) + R / eps)
        return C

    @staticmethod
    def _confidense_log_2(R, alpha, eps):
        C =  np.log(1 +  1.5 *(R>0) * eps)
        return C

    def _init_X(self, users_count):
        X = np.empty((users_count, self.n_factors))
        X = self.init_std_dev * np.random.randn(users_count, self.n_factors)
        return X

    def _init_Y(self, items_count):
        Y = np.empty((items_count, self.n_factors))
        Y = self.init_std_dev * np.random.randn(items_count, self.n_factors)
        return Y

    def fit(self, train_data, y_train, test_data, shape, y_test = None, verbose = -1):
        R_train  = np.zeros(shape)
        R_train[train_data[:, 0].astype(int), train_data[:, 1].astype(int)] = y_train

        R_test = np.zeros(shape)
        R_test[test_data[:, 0].astype(int), test_data[:, 1].astype(int)] = 1

        users_count = R_train.shape[0]
        items_count = R_train.shape[1]


        not_zero_mask = (R_train > 0)
        zero_mask = (R_train == 0)
        implicit_mask = (R_test > 0)

        full_mean = R_train[not_zero_mask].mean()

        X = self._init_X(users_count)
        Y = self._init_Y(items_count)


        #init biases
        # user_bias = np.true_divide(R_train.sum(1),1+(R_train!=0).sum(1))#R_train.mean(1)
        # item_bias = np.true_divide(R_train.sum(0),1+(R_train!=0).sum(0))
        # R_train = R_train - not_zero_mask*full_mean

        user_bias = np.repeat(0.0, users_count)
        # for u in range(users_count):
        #     d = R_train[u]
        #     if d[d>0].shape[0] == 0:
        #         user_bias[u] = full_mean
        #     else:
        #         user_bias[u] = d.mean()
        #user_bias += 0.1 * np.random.randn(users_count)

        item_bias = np.repeat(0.0, items_count)
        # for i in range(items_count):
        #     d = R_train[:, i]
        #     if d[d>0].shape[0] == 0:
        #         item_bias[i] = full_mean
        #     else:
        #         item_bias[i] = d.mean()
        #item_bias += 0.1 * np.random.randn(items_count)

       # R_train = R_train - not_zero_mask*user_bias[ :, None] - not_zero_mask*item_bias[None, :]

        lrmat = self.reg_emb * np.eye(self.n_factors, self.n_factors)


        P = R_train + 0.5 * (zero_mask& ~implicit_mask)*( full_mean  + user_bias[ :, None] + item_bias[None, :]) +\
             implicit_mask * (full_mean  + user_bias[ :, None] + item_bias[None, :])
        #P = R_train - full_mean #+ (implicit_mask)*(user_bias[ :, None] + item_bias[None, :])

        W = not_zero_mask + implicit_mask
        W = W.astype(int)
        #
        C = self._confidense_log_2(R_train, self.alpha, self.eps)
        #C += 0.001*np.random.randn(C.shape[0], C.shape[1])
        C[implicit_mask] = 0
        # C[zero_mask & ~implicit_mask] = C[zero_mask & ~implicit_mask] * 1.2

        if self.verbose:
            epochs_range = tqdm(range(self.n_epochs))
        else:
            epochs_range = range(self.n_epochs)


        train_scores = []
        test_scores = []

        for num_epoch in epochs_range:

            # if num_epoch == 300:
            #     self.n_factors += 1
            #     X = np.c_[X, 0.001*np.random.randn(X.shape[0])]
            #     Y = np.c_[Y, 0.001*np.random.randn(Y.shape[0])]
            #     lrmat = self.reg_emb * np.eye(self.n_factors, self.n_factors)
            #     #self.reg_emb +=

            # if num_epoch % 20 == 0:
            #     self.n_factors += 1
            #     X = np.c_[X, 1e-30 * np.zeros(X.shape[0])]
            #     Y = np.c_[Y, 1e-30 * np.zeros(Y.shape[0])]
            #     lrmat = self.reg_emb * np.eye(self.n_factors, self.n_factors)
            #     self.reg_emb += 100
                # self.reg_emb = self.reg_emb * 0.9
                # self.reg_bias = self.reg_bias * 0.9
                # X -= 0.01 * np.random.randn(X.shape[0], X.shape[1])
                # Y -= 0.01 * np.random.randn(Y.shape[0], Y.shape[1])




            # user step

           # YtY = np.matmul(Y.T, Y)
            Puser = C*(P - full_mean - item_bias[None, :] - user_bias[:, None])

            for u in range(users_count):
                yt_inv = np.linalg.inv(np.matmul(Y.T*(C[u, :]), Y) + lrmat)
                _a = np.matmul(Y.T, Puser[u, :])
                X[u, :] = np.matmul(yt_inv, _a) + 1e-3*np.random.randn(X[u, :].shape[0])


            # item step
           # XtX = np.matmul(X.T, X)
            Pitem = C*(P - full_mean - user_bias[:, None] - item_bias[None, :])

            for i in range(items_count):
                xt_inv = np.linalg.inv(np.matmul(X.T*(C[:, i]), X) + lrmat)
                _a = np.matmul( X.T, Pitem[:,i] )
                Y[i, :] = np.matmul(xt_inv, _a)+ 1e-3*np.random.randn(Y[i, :].shape[0])

            #bias step
            # for u in range(users_count):
            #     user_bias[u] = 1/self.lr * (C[u,:] * (P[u, :]  - X[u, :].dot(Y.T))).mean()

            full_mean = 1.0 / (C.sum()) * (C * (P - user_bias[:, None] - item_bias[None, :] - np.matmul(X, Y.T))).sum()

            user_bias = 1.0 / (C.sum(1) + self.reg_bias) * (C*(P - item_bias[None, :] - full_mean - np.matmul(X, Y.T))).sum(1)
            item_bias = 1.0 / (C.sum(0) + self.reg_bias) * (C*(P - user_bias[:, None] - full_mean - np.matmul(X, Y.T))).sum(0)



            self.user_bias = user_bias
            self.item_bias = item_bias
            self.X = X
            self.Y = Y
            self.full_mean = full_mean

            if y_test is not None:
                test_score = self.score(test_data, y_test)
                train_score = self.score(train_data, y_train)
                train_scores.append(train_score)
                test_scores.append(test_score)

                if verbose > 0 and num_epoch%verbose == 0 or num_epoch == self.n_epochs-1:
                    print('iter: {0} train_score: {1}   test_score: {2}'.format(num_epoch, train_score, test_score))

        return np.array(train_scores), np.array(test_scores)

    def estimate(self, u, i):
        score = 0
        score += self.full_mean * self.substract_mean
        score += self.user_bias[u] + self.item_bias[i]
        score += np.matmul(self.X[u],self.Y[i])
        if score > 5:
            return 5
        if score < 1:
            return 1

        return score

    def predict(self, test_data):
        score = np.zeros(test_data.shape[0])
        score += self.full_mean * self.substract_mean
        score += self.user_bias[test_data[:, 0].astype(int)]
        score += self.item_bias[test_data[:, 1].astype(int)]
        score += (self.X[test_data[:, 0].astype(int)] *
                          self.Y[test_data[:, 1].astype(int)]).sum(1)

        score[score > 5] = 5
        score[score < 1] = 1
        return score

    def score(self, test_data, y_test):
        y_pred = self.predict(test_data)
        return np.sqrt(np.mean((y_test - y_pred)**2))



class PPImplicitCF(BaseEstimator):
    def describe(self):
        res = 'my_PPimplicitCF_f-{0}_ep-{1}_alpha-{2}_eps-{3}_lr-{4}_cfunc-{5}_test_impl-{6}_smean-{7}'
        res = res.format(self.n_factors,
                         self.n_epochs,
                         self.alpha,
                         self.eps,
                         self.reg_emb,
                         self.reg_bias,
                         self.c_function,
                         self.use_test_as_implicit,
                         self.substract_mean)
        return res

    def __init__(self,
                 n_factors = 4,
                 n_epochs = 20,
                 alpha = 5,
                 eps = 0.01,
                 init_mean = 0,
                 init_std_dev= 0.1,
                 reg_emb = 0.07,
                 reg_bias = 0.01,
                 random_state = None,
                 verbose = False,
                 c_function = 'linear',
                 use_test_as_implicit = True,
                 substract_mean = True,
                 mean_decrease = 0.85):

        self.n_factors =  n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.reg_emb = reg_emb
        self.reg_bias = reg_bias
        self.random_state = random_state
        self.verbose = verbose
        self.alpha = alpha
        self.eps = eps
        self.use_test_as_implicit = use_test_as_implicit
        self.substract_mean = substract_mean
        self.c_function = c_function
        self.mean_decrease = mean_decrease

        if c_function is 'linear':
            self.confidense_func = self._confidense_linear
        if c_function is 'log':
            self.confidense_func = self._confidense_log


        pass

    def get_confidense_func(self):
        if self.c_function is 'linear':
            return  self._confidense_linear
        if self.c_function is 'log':
            return  self._confidense_log
        return self._confidense_linear

    @staticmethod
    def _confidense_linear(R, alpha, eps):
        C = np.ones(R.shape) + alpha*R
        return C

    @staticmethod
    def _confidense_log(R, alpha, eps):
        C =  np.log(np.ones(R.shape) + R / eps)
        return C

    def _init_X(self, users_count):
        X = np.empty((users_count, self.n_factors))
        X = self.init_std_dev * np.random.randn(users_count, self.n_factors)
        return X

    def _init_Y(self, items_count):
        Y = np.empty((items_count, self.n_factors))
        Y = self.init_std_dev * np.random.randn(items_count, self.n_factors)
        return Y

    def fit(self, train_data, y_train, test_data, shape, y_test = None, verbose = -1):
        R_train  = np.zeros(shape)
        R_train[train_data[:, 0].astype(int), train_data[:, 1].astype(int)] = y_train

        R_test = np.zeros(shape)
        R_test[test_data[:, 0].astype(int), test_data[:, 1].astype(int)] = 1

        users_count = R_train.shape[0]
        items_count = R_train.shape[1]


        not_zero_mask = (R_train > 0)
        zero_mask = (R_train == 0)
        implicit_mask = (R_test > 0)

        full_mean = R_train[not_zero_mask].mean()

        X = self._init_X(users_count)
        Y = self._init_Y(items_count)
        Z = self._init_Y(items_count)

        N = (R_train > 0 | R_test > 0).astype(int)


        #init biases
        # user_bias = np.true_divide(R_train.sum(1),1+(R_train!=0).sum(1))#R_train.mean(1)
        # item_bias = np.true_divide(R_train.sum(0),1+(R_train!=0).sum(0))
        # R_train = R_train - not_zero_mask*full_mean

        user_bias = np.repeat(0.0, users_count)
        # for u in range(users_count):
        #     d = R_train[u]
        #     if d[d>0].shape[0] == 0:
        #         user_bias[u] = 0.0
        #     else:
        #         user_bias[u] = d.mean()/2

        item_bias = np.repeat(0.0, items_count)
        # for i in range(items_count):
        #     d = R_train[:, i]
        #     if d[d>0].shape[0] == 0:
        #         item_bias[i] = 0.0
        #     else:
        #         item_bias[i] = d.mean()/2

       # R_train = R_train - not_zero_mask*user_bias[ :, None] - not_zero_mask*item_bias[None, :]

        lrmat = self.reg_emb * np.eye(self.n_factors, self.n_factors)


        P = R_train + 0.5 * (zero_mask& ~implicit_mask)*( full_mean  + user_bias[ :, None] + item_bias[None, :]) +\
             implicit_mask * (full_mean  + user_bias[ :, None] + item_bias[None, :])
        #P = R_train - full_mean #+ (implicit_mask)*(user_bias[ :, None] + item_bias[None, :])

        W = not_zero_mask + implicit_mask
        W = W.astype(int)
        #
        C = self._confidense_log(R_train + R_test, self.alpha, self.eps)

        if self.verbose:
            epochs_range = tqdm(range(self.n_epochs))
        else:
            epochs_range = range(self.n_epochs)


        train_scores = []
        test_scores = []

        for num_epoch in epochs_range:

            # if num_epoch%10== 0 and num_epoch > 0:
            #     self.reg_emb = self.reg_emb * 1.2
            #     self.reg_bias = self.reg_bias * 1.2

            # user step
            YtY = np.matmul(Y.T, Y)
            Puser = C*(P - full_mean - item_bias[None, :] - user_bias[:, None])

            for u in range(users_count):
                yt_inv = np.linalg.inv(np.matmul(Y.T*(C[u, :]), Y) + lrmat)
                _a = np.matmul(Y.T, Puser[u, :])
                X[u, :] = np.matmul(yt_inv, _a)


            # item step
            XtX = np.matmul(X.T, X)
            Pitem = C*(P - full_mean - user_bias[:, None] - item_bias[None, :])

            for i in range(items_count):
                xt_inv = np.linalg.inv(np.matmul(X.T*(C[:, i]), X) + lrmat)
                _a = np.matmul( X.T, Pitem[:,i] )
                Y[i, :] = np.matmul(xt_inv, _a)

            #bias step
            # for u in range(users_count):
            #     user_bias[u] = 1/self.lr * (C[u,:] * (P[u, :]  - X[u, :].dot(Y.T))).mean()
            user_bias = 1.0 / (C.sum(1) + self.reg_bias) * (C*(P - item_bias[None, :] - full_mean - np.matmul(X, Y.T))).sum(1)
            item_bias = 1.0 / (C.sum(0) + self.reg_bias) * (C*(P - user_bias[:, None] - full_mean - np.matmul(X, Y.T))).sum(0)

            self.user_bias = user_bias
            self.item_bias = item_bias
            self.X = X
            self.Y = Y
            self.full_mean = full_mean

            if y_test is not None:
                test_score = self.score(test_data, y_test)
                train_score = self.score(train_data, y_train)
                train_scores.append(train_score)
                test_scores.append(test_score)

                if verbose > 0 and num_epoch%verbose == 0 or num_epoch == self.n_epochs-1:
                    print('iter: {0} train_score: {1}   test_score: {2}'.format(num_epoch, train_score, test_score))

        return np.array(train_scores), np.array(test_scores)

    def estimate(self, u, i):
        score = 0
        score += self.full_mean * self.substract_mean
        score += self.user_bias[u] + self.item_bias[i]
        score += np.matmul(self.X[u],self.Y[i])
        if score > 5:
            return 5
        if score < 1:
            return 1

        return score

    def predict(self, test_data):
        score = np.zeros(test_data.shape[0])
        score += self.full_mean * self.substract_mean
        score += self.user_bias[test_data[:, 0].astype(int)]
        score += self.item_bias[test_data[:, 1].astype(int)]
        score += (self.X[test_data[:, 0].astype(int)] *
                          self.Y[test_data[:, 1].astype(int)]).sum(1)

        score[score > 5] = 5
        score[score < 1] = 1
        return score

    def score(self, test_data, y_test):
        y_pred = self.predict(test_data)
        return np.sqrt(np.mean((y_test - y_pred)**2))

def cross_val_iCF(model, X, y, shape, cv, verbose = 10):

    cv_train_scores = []
    cv_test_scores = []

    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_scores, test_scores = model.fit(X_train, y_train, X_test, shape, y_test, verbose = verbose)
        cv_train_scores.append(train_scores)
        cv_test_scores.append(test_scores)

    cv_test_scores = np.array(cv_test_scores)
    cv_train_scores = np.array(cv_train_scores)

    return cv_train_scores, cv_test_scores


class MeanModel(BaseEstimator):
    def describe(self):
        res = 'mean-model_'
        for m in self.models:
            res += m.model_name()+'_'
        return res

    def __init__(self, models):
        self.models = models



    def fit(self, train_data, y_train, test_data, shape, y_test=None, verbose = False):
        train_data = train_data.astype(int)
        for m in self.models:
            m.fit(train_data, y_train, test_data, shape)

        if y_test is not None:
            train_score = self.score(train_data, y_train)
            test_score = self.score(test_data, y_test)
            return train_score, test_score

        return None, None



    def estimate(self, uid, item_id):
        test_data = np.array([[uid, item_id]])
        test_data = test_data.astype(int)
        pred = 0
        for m in self.models:
            pred += m.estimate(uid, item_id)

        pred = pred/ len(self.models)
        return pred

    def predict(self, test_data):
        test_data = test_data.astype(int)
        y_pred = []
        for m in self.models:
            y_pred.append(m.predict(test_data))
        y_pred = np.array(y_pred)
        y_pred = y_pred.mean(0)
        return y_pred

    def score(self, test_data, y_test):
        y_pred = self.predict(test_data)
        return np.sqrt(np.mean((y_test - y_pred)**2))




class MetaALS(BaseEstimator):

    def describe(self):
        res = 'meta-model_'
        res = res + self.ials_model.describe()
        return res

    def __init__(self, ials_model, meta_model):
        self.meta_model = meta_model
        self.ials_model = ials_model

    def get_features(self, uid, vid):
        to_stack = []
        full_mean = self.ials_model.full_mean
        to_stack.append(full_mean)
        if hasattr(self.ials_model, 'user_bias'):
            u_bias = self.ials_model.user_bias[uid]
            to_stack.append(u_bias)

        if hasattr(self.ials_model, 'item_bias'):
            v_bias = self.ials_model.item_bias[vid]
            to_stack.append(v_bias)

        u_emb = self.ials_model.X[uid]
        v_emb = self.ials_model.Y[vid]
        to_stack.append(u_emb*v_emb)

        return np.hstack(to_stack)


    def create_meta_train(self, data, y=None):

        features_size = self.get_features(data[0, 0],
                                          data[0, 1]).shape[0]

        meta_x_train = np.empty((data.shape[0], features_size))
        to_stack = []

        full_mean_arr = np.repeat(self.ials_model.full_mean, data.shape[0])
        to_stack.append(full_mean_arr[:, None],)

        if hasattr(self.ials_model, 'user_bias'):
            user_bias_arr = self.ials_model.user_bias[data[:, 0]]
            to_stack.append(user_bias_arr[:, None],)

        if hasattr(self.ials_model, 'item_bias'):
            item_bias_arr = self.ials_model.item_bias[data[:, 1]]
            to_stack.append(item_bias_arr[:, None],)

        product_arr = self.ials_model.X[data[:, 0], :] * self.ials_model.Y[data[:, 1], :]
        to_stack.append(product_arr)

        meta_x_train =np.hstack(to_stack) #np.hstack([full_mean_arr[:, None],user_bias_arr[:, None],item_bias_arr[: , None], product_arr])

        return meta_x_train, y


    def fit(self, train_data, y_train, test_data, shape):
        train_data = train_data.astype(int)
        self.ials_model.fit(train_data, y_train, test_data, shape)
        meta_x_train, meta_y_train = self.create_meta_train(train_data, y_train)
        self.meta_model.fit(meta_x_train, meta_y_train)


    def estimate(self, uid, item_id):
        test_data = np.array([[uid, item_id]])
        test_data = test_data.astype(int)
        x_test, _ = self.create_meta_train(test_data)
        y_pred = self.meta_model.predict(x_test)
        return y_pred.ravel()[0]

    def predict(self, test_data):
        test_data = test_data.astype(int)
        x_test, _ =self.create_meta_train(test_data)
        y_pred = self.meta_model.predict(x_test)
        return y_pred

    def score(self, test_data, y_test):
        y_pred = self.predict(test_data)
        return np.sqrt(np.mean((y_test - y_pred)**2))








