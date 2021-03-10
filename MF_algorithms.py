import pandas as pd
import numpy as np


class eALS():
    def __init__(self, data_path, R, factors=10, init_mean=0, init_std=0.01, reg=0.01):
        self.data_path = data_path
        self.R = R
        self.nb_items = len(pd.read_csv(data_path)['item_id'].unique())  # nb of items
        self.nb_users = len(pd.read_csv(data_path)['user_id'].unique())  # nb of users
        self.factors = factors
        self.reg = reg
        self.init_mean = init_mean
        self.init_std = init_std

    def predict(self, u, i, U, V):
        pred_matrix = U @ V.T
        pred_value = pred_matrix[u, i]
        return pred_value

    def weight_init(self, alpha=0.5, w0=1):
        summ = 0
        Z = 0
        p = np.zeros(self.nb_items)
        Wi = np.zeros(self.nb_items)
        W = np.zeros(shape=(self.nb_users, self.nb_items))

        for i in range(self.nb_items):
            p[i] = np.count_nonzero(self.R[:, i])
            summ += p[i]

        for i in range(self.nb_items):
            p[i] /= summ
            p[i] = p[i] ** alpha
            Z += p[i]

        for i in range(self.nb_items):
            Wi[i] = w0 * p[i] / Z

        for u in range(self.nb_users):
            idx = np.nonzero(self.R[u, :])[0]
            for i in idx:
                W[u, i] = 1

        return W, Wi

    def update_user(self, u, W, Wi, U, V, SU, SV, prediction_items, rating_items, w_items):
        item_list = np.nonzero(self.R[u, :])[0]  # array of items rated by user u

        # prediction cache for the user
        for i in item_list:
            prediction_items[i] = self.predict(u, i, U, V)  # predict needs to be defined
            rating_items[i] = self.R[u, i]
            w_items[i] = W[u, i]

        oldVector = U[u, :]  # MB get row of specific user from random matrix U
        for f in range(self.factors):
            # MB initialize numerator and denominator
            numer = 0
            denom = 0

            for k in range(self.factors):
                if k != f:
                    numer -= U[u, k] * SV[f, k]  # MB eq.[12], numerator, second part
            for i in item_list:
                prediction_items[i] -= U[u, f] * V[i, f]  # MB prediction without the component of latent f
                numer += (w_items[i] * rating_items[i] - (w_items[i] - Wi[i]) * prediction_items[i]) * V[
                    i, f]  # MB eq.[12], numerator, first part
                denom += (w_items[i] - Wi[i]) * V[i, f] * V[i, f]
            denom += SV[f, f] + self.reg  # MB eq.[12], denominator, second part

            # Parameter Update
            U[u, f] = numer / denom

            # Update the prediction cache
            for i in range(self.nb_items):
                prediction_items[i] += U[u, f] * V[i, f]

            # Update SU cache
        for f in range(self.factors):
            for k in range(f):
                val = SU[f, k] - oldVector[f] * oldVector[k] + U[u, f] * U[u, k]
                SU[f, k] = val
                SU[k, f] = val
        return U[u]

    def update_item(self, i, W, Wi, U, V, SU, SV, prediction_users, rating_users, w_users):
        user_list = np.nonzero(self.R[:, i])[0]  # array of items rated by user u

        # prediction cache for the user
        for u in user_list:
            prediction_users[u] = self.predict(u, i, U, V)
            rating_users[u] = self.R[u, i]
            w_users[u] = W[u, i]

        oldVector = V[i, :]  # MB get row of specific item from random matrix V
        for f in range(self.factors):
            # MB initialize numerator and denominator
            numer = 0
            denom = 0

            for k in range(self.factors):
                if k != f:
                    numer -= V[i, k] * SU[f, k]  # MB eq.[12], numerator, second part
            numer *= Wi[i]
            for u in user_list:
                prediction_users[u] -= U[u, f] * V[i, f]  # MB prediction without the component of latent f
                numer += (w_users[u] * rating_users[u] - (w_users[u] - Wi[i]) * prediction_users[u]) * U[
                    u, f]  # MB eq.[12], numerator, first part
                denom += (w_users[u] - Wi[i]) * U[u, f] * U[u, f]
            denom += Wi[i] * SU[f, f] + self.reg  # MB eq.[12], denominator, second part

            # Parameter Update
            V[i, f] = numer / denom

            # Update the prediction cache for the item
            for u in user_list:
                prediction_users[u] += U[u, f] * V[i, f]

            # Update SV cache
        for f in range(self.factors):
            for k in range(f):
                val = SV[f, k] - oldVector[f] * oldVector[k] * Wi[i] + V[i, f] * V[i, k] * Wi[i]
                SV[f, k] = val
                SV[k, f] = val

        return V[i]

    def fit(self, iterations):
        W, Wi = self.weight_init()
        prediction_items = np.zeros(self.nb_items)
        rating_items = np.zeros(self.nb_items)
        w_items = np.zeros(self.nb_items)
        prediction_users = np.zeros(self.nb_users)
        rating_users = np.zeros(self.nb_users)
        w_users = np.zeros(self.nb_users)
        U = np.random.normal(self.init_mean, self.init_std, self.nb_users*self.factors).reshape((self.nb_users, self.factors))
        V = np.random.normal(self.init_mean, self.init_std, self.nb_items *self. factors).reshape((self.nb_items, self.factors))
        SU = U.T @ U
        SV = np.zeros(shape=(self.factors, self.factors))

        for f in range(self.factors):
            for k in range(f):
                val = 0
                for i in range(self.nb_items):
                    val += self.V[i, f] * self.V[i, k] * Wi[i]
                SV[f, k] = val
                SV[k, f] = val

        SU = sc.broadcast(SU)
        SV = sc.broadcast(SV)

        for i in iterations:
            Users = sc.parallelize




