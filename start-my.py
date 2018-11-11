import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

data_folder = '../../Data/hw-ials/'

def load_data(filename, train=True):
    fio = open(filename, 'r')
    rows = []
    cols = []
    data = []

    for l in fio.readlines():
        args = l.split('\t')
        args = [x for x in args if len(x) > 0]
        i = int(args[0])
        j = int(args[1])
        p = 1
        if train:
            p = float(args[2])
        rows.append(i)
        cols.append(j)
        data.append(p)


    fio.close()
    res = sp.csr_matrix((data, (rows, cols)))
    return res


def load_test_data(filename):
    fio = open(filename, 'r')
    res = []
    for l in fio.readlines():
        args = l.split('\t')
        args = [x for x in args if len(x)> 0]
        uid = int(args[0])
        item_id = int(args[1])
        res.append((uid, item_id))
    fio.close()
    return res

def matrix_to_table(R):
    idxs = np.argwhere(R>0)
    res = np.empty((idxs.shape[0], 2))
    res[:, 0:2] = idxs.astype(int)
    y = R[idxs[:, 0], idxs[:, 1]]
    return res, y

def table_to_matrix(T, shape):
    res = np.zeros(shape)
    res[T[:, 0].astype(int), T[:, 1].astype(int)] = T[:, 2]
    return res

def predict(model, test_data:list):
    res = []
    for pair in tqdm(test_data):
        uid, item_id = pair
        p = model.estimate(uid, item_id)
        res.append((uid, item_id, p))
    return res

def save_predict(preds: list, filename):
    fio = open(filename, 'w')
    fio.write('Id,Score\n')

    for i, tup in enumerate(preds):
        uid, item_id, p = tup
        fio.write('{0},{1}\n'.format(i+1, p))

    fio.close()


from model import IBiasedAlsModel,  ISimpleAlsModel, ExplicitCF
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

import pickle
def grid_search_params():
    R_train = load_data(data_folder + 'train.txt', True).toarray()
    users_count = R_train.shape[0]
    items_count = R_train.shape[1]
    R_shape = (users_count, items_count)

    R_test = load_data(data_folder + 'test.txt', False)
    R_test.resize(R_train.shape)
    R_test = R_test.toarray()

    test_pairs = load_test_data(data_folder + 'test.txt')

    train_data, y_train = matrix_to_table(R_train)
    test_data, y_test = matrix_to_table(R_test)

    gs = GridSearchCV(estimator=IBiasedAlsModel(),
                      fit_params={'test_data': test_data,
                                  'shape': R_shape},
                      scoring=make_scorer(rmse),

                      param_grid={'n_factors': [5,10,20],
                                  'n_epochs': [10,20,30],
                                  'alpha': [10,20,40],
                                  'lr': [0.007, 0.01, 0.1],
                                  'c_function': ['linear', 'log'],
                                  'use_test_as_implicit': [True, False],
                                   'substract_mean': [True, False]},
                      verbose=20,
                      n_jobs=1)
    gs.fit(train_data, y_train)

    print('BEST SCORE: {0}'.format(gs.best_score_))
    print('BEST_ESTIMATOR: {0}'.format(gs.best_estimator_))
    pickle.dump(gs, open('grid_search_res.pkl', 'wb'))


from model import ImplicitCF, MeanModel
def get_model():
    # ImplicitCF(n_epochs=200, n_factors=6, reg_emb=55, reg_bias=0.02, alpha=500, eps=0.01,
    #            use_test_as_implicit=False, c_function='log',
    #            mean_decrease=0.1,
    #            substract_mean=True,
    #            verbose=True)

    als_model = ImplicitCF(n_epochs=500, n_factors=6, reg_emb=55, reg_bias=0.001, alpha=500, eps=100,
                           c_function='log', random_state=6756,
                           verbose=True)
    als_model1 = ImplicitCF(n_epochs=50, n_factors=3, reg_emb=10, reg_bias=0.001, alpha=500, eps=100,
                           c_function='log', random_state=98398,
                           verbose=True)

    als_model2 = IBiasedAlsModel(n_epochs=200, n_factors=3, lr=0.5, alpha=500, eps=0.02,
                           c_function='log', random_state=3232,
                           verbose=True)
    meta_model = MeanModel(models=[als_model,als_model1, als_model2])

    return meta_model

from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut, train_test_split
import xgboost as xgb
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics.scorer import  _BaseScorer


class TrainTestSplitter:
    def __init__(self, test_size = 0.2, random_state = None):
        self.test_size = 0.2
        self.random_state = random_state

    def split(self, X):
        indices = np.arange(X.shape[0])
        indices_train, indices_test = train_test_split(indices,
                                                       test_size=self.test_size,
                                                       random_state=self.random_state)
        return  [(indices_train, indices_test)]



from model import cross_val_iCF
import matplotlib.pyplot as plt
def cross_val():
    R_train = load_data(data_folder + 'train.txt', True).toarray()
    users_count = R_train.shape[0]
    items_count = R_train.shape[1]
    R_shape = (users_count, items_count)

    R_test = load_data(data_folder + 'test.txt', False)
    R_test.resize(R_train.shape)
    R_test = R_test.toarray()

    test_pairs = load_test_data(data_folder + 'test.txt')

    train_data, y_train = matrix_to_table(R_train)
    test_data, y_test = matrix_to_table(R_test)

    # model = IAlsModel(n_epochs=300, n_factors=4,  lr=0.07, alpha=5.0, eps = 0.1,
    #
    #                   use_test_as_implicit=False, c_function='log',
    #                   mean_decrease=0.85,
    #                   verbose=True)
    # als_model = IAlsModel(n_epochs=100, n_factors=3, lr = 0.01, alpha = 8.0, eps=0.1,
    #                   use_test_as_implicit=False, c_function='log',
    #                   substract_mean=True,
    #                   verbose=True)
    # xgb_reg = xgb.XGBRegressor(n_estimators=200, silent=True, max_depth=5)
    # lin_reg = SGDRegressor()
    #
    # meta_model = MetaALS(ials_model=als_model, meta_model=xgb_reg)

    model = get_model()
    cv = TrainTestSplitter(random_state=123)
    cv_train_scores, cv_test_scores = cross_val_iCF(model,train_data, y_train, R_shape, cv, verbose=10)

    cv_train_scores = cv_train_scores.mean(0)
    cv_test_scores = cv_test_scores.mean(0)


    try:
        plt.figure()
        plt.plot(cv_train_scores[25:], label = 'train')
        plt.plot(cv_test_scores[25:], label='test')
        plt.legend()
        plt.show()
        print('MEAN SCORE: ', cv_test_scores[-1])
    except:
        pass

    print('MEAN SCORE: ', cv_test_scores)



    # cv = cross_val_score(model, train_data, y_train,
    #                      fit_params={'test_data': train_data,
    #                                  'shape': R_shape},
    #                      scoring=make_scorer(rmse),
    #                      verbose=10,
    #                      cv = KFold(n_splits=3, shuffle=True, random_state=4322),
    #                      n_jobs=3
    #                      )
    # print("CV_MEAN: ", cv.mean())


def cross_val_meta():
    R_train = load_data(data_folder + 'train.txt', True).toarray()
    users_count = R_train.shape[0]
    items_count = R_train.shape[1]
    R_shape = (users_count, items_count)

    R_test = load_data(data_folder + 'test.txt', False)
    R_test.resize(R_train.shape)
    R_test = R_test.toarray()

    test_pairs = load_test_data(data_folder + 'test.txt')

    train_data, y_train = matrix_to_table(R_train)
    test_data, y_test = matrix_to_table(R_test)

    model = get_model()

    cv = cross_val_score(model, train_data, y_train,
                         fit_params={'test_data': train_data,
                                     'shape': R_shape},
                         scoring=make_scorer(rmse),
                         verbose=10,
                         cv = TrainTestSplitter(random_state=123),
                         n_jobs=3
                         )
    print("CV_MEAN: ", cv.mean())





if __name__ == "__main__":
    # cross_val()
    # exit(0)
    # grid_search_params()
    # exit(0)



    R_train = load_data(data_folder + 'train.txt', True).toarray()
    users_count = R_train.shape[0]
    items_count = R_train.shape[1]
    R_shape = (users_count, items_count)

    R_test = load_data(data_folder + 'test.txt', False)
    R_test.resize(R_train.shape)
    R_test = R_test.toarray()

    test_pairs = load_test_data(data_folder + 'test.txt')

    train_data, y_train = matrix_to_table(R_train)
    test_data, y_test = matrix_to_table(R_test)

    meta_model = get_model()

    model_name = meta_model.describe()#'my_ials_model_300_4f_007_50'
    meta_model.fit(train_data, y_train, test_data, R_shape)

    preds = predict(meta_model, test_pairs)

    prediction_filename = model_name + '_predictions.txt'
    prediction_filename = 'predictions/current_' + prediction_filename

    save_predict(preds, prediction_filename)