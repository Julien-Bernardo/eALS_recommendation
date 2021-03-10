import pandas as pd
import numpy as np

class eALS():
    def __init__(self, R, factors=10, init_mean=0, init_std=0.01, reg=0.01):
        self.R = R # rating matrix
        self.nb_items = R.shape[1]  # nb of items
        self.nb_users = R.shape[0]  # nb of users
        self.factors = factors # number of latent factors
        self.reg = reg # regularization parameter (lambda in the paper formula)
        self.init_mean = init_mean # mean and std deviation to initialize the matrices U and V
        self.init_std = init_std

    def predict(self, u, i, U, V):
        '''
        Returns the predicted rating of a user u for an item i by
        estimating the rating matrix with the product U @ V.T and returning
        the element (u,i)
        '''
        pred_matrix = U @ V.T
        pred_value = pred_matrix[u, i]
        return pred_value

    def squared_sum(self, M):
        summ = 0
        rows, columns = M.shape
        for row in range(rows):
            norm = np.linalg.norm(M[row])
            summ += norm ** 2
        return summ

    def loss(self, U, V, W, Wi, SV):
        L = self.reg * (self.squared_sum(U) + self.squared_sum(V))
        for u in range(self.nb_users):
            l = 0
            idx = np.nonzero(self.R[u, :])[0]
            for i in idx:
                pred = self.predict(u, i, U, V)
                l += W[u, i] * (self.R[u, i] - pred) ** 2
                l -= Wi[i] * pred ** 2
            l += SV @ U[u, :] @ U[u, :].T
            L += l
        return L

    def weight_init(self, alpha=0.5, w0=1):
        '''
        Computes the weights for both positive and missing entries
        '''

        # parameter initialization
        summ = 0
        Z = 0
        p = np.zeros(self.nb_items)
        Wi = np.zeros(self.nb_items) # weights for missing instances on items.
        W = np.zeros(shape=(self.nb_users, self.nb_items)) # weights for positive instances

        for i in range(self.nb_items):
            # p[i]: total number of users that interact with item i
            p[i] = np.count_nonzero(self.R[:, i])
            summ += p[i]

        for i in range(self.nb_items):
            p[i] /= summ # frequency of item i (numerator of eq. [8] in the paper)
            p[i] = p[i] ** alpha
            Z += p[i]

        for i in range(self.nb_items):
            # We parametrize Wi based on item’s popularity
            # Equation [8] in the paper
            Wi[i] = w0 * p[i] / Z

        for u in range(self.nb_users):
            '''
            By default, the weight for positive instance is uniformly 1.
            For each user, get all the items with which he interacted and set 
            the corresponding value in weight matrix to 1
            '''
            idx = np.nonzero(self.R[u, :])[0]
            for i in idx:
                W[u, i] = 1

        return W, Wi

    def update_user(self, u, W, Wi, U, V, SU, SV, prediction_items, rating_items, w_items):
        '''
        Update the u-th latent vector in the user matrix U by using Fast eALS.
        That is, we update each one of the latent components in the vector
        one by one while keeping the others fixed.
        '''
        item_list = np.nonzero(self.R[u, :])[0]  # array of items rated by user u
        SU = SU.value
        SV = SV.value

        # prediction cache for the user
        for i in item_list:
            prediction_items[i] = self.predict(u, i, U, V)  # predicted rating for (u, i)
            rating_items[i] = self.R[u, i] # actual rating for the pair (u, i)
            w_items[i] = W[u, i]

        oldVector = U[u, :] # get latent vector u from matrix U
        for f in range(self.factors):
            # initialize numerator and denominator
            numer = 0
            denom = 0

            for k in range(self.factors):
                # Note that the expression to update user [12] is split into latent
                # factor f and the rest of the factors
                if k != f:
                    numer -= U[u, k] * SV[f, k]  # eq.[12], second part of numerator
            for i in item_list:
                prediction_items[i] -= U[u, f] * V[i, f]  # prediction without the latent component f
                numer += (w_items[i] * rating_items[i] - (w_items[i] - Wi[i]) * prediction_items[i]) * V[
                    i, f]  # eq.[12] first part of the numerator
                denom += (w_items[i] - Wi[i]) * V[i, f] * V[i, f]
            denom += SV[f, f] + self.reg  #eq.[12] second part of the denominator

            # update of component f in latent user vector u, element (u,f) in matrix U
            U[u, f] = numer / denom

            # update the prediction cache
            for i in range(self.nb_items):
                prediction_items[i] += U[u, f] * V[i, f]

            # update the SU cache
        for f in range(self.factors):
            for k in range(f):
                val = SU[f, k] - oldVector[f] * oldVector[k] + U[u, f] * U[u, k]
                SU[f, k] = val
                SU[k, f] = val
        return U[u]

    def update_item(self, i, W, Wi, U, V, SU, SV, prediction_users, rating_users, w_users):
        '''
        Update the i-th latent vector in the item matrix V by using Fast eALS.
        That is, we update each one of the latent components in the vector
        one by one while keeping the others fixed.
        '''
        user_list = np.nonzero(self.R[:, i])[0] # array of users interacting with item i
        SU = SU.value
        SV = SV.value
        # prediction cache for the user
        for u in user_list:
            prediction_users[u] = self.predict(u, i, U, V) # predicted rating for (u, i)
            rating_users[u] = self.R[u, i] # actual rating for (u, i)
            w_users[u] = W[u, i]

        oldVector = V[i, :] # get latent vector i from matrix V
        for f in range(self.factors):
            # initialize numerator and denominator
            numer = 0
            denom = 0

            for k in range(self.factors):
                # Note that the expression to update item [13] is split into latent
                # factor f and the rest of the factors
                if k != f:
                    numer -= V[i, k] * SU[f, k]  # MB eq.[12], numerator, second part
            numer *= Wi[i]
            for u in user_list:
                prediction_users[u] -= U[u, f] * V[i, f]  # prediction without the component of latent f
                numer += (w_users[u] * rating_users[u] - (w_users[u] - Wi[i]) * prediction_users[u]) * U[u, f]  # eq.[13], first part of numerator
                denom += (w_users[u] - Wi[i]) * U[u, f] * U[u, f]
            denom += Wi[i] * SU[f, f] + self.reg  # eq.[13], second part of denominator

            # update of component f in latent item vector i, element (i,f) in matrix V
            V[i, f] = numer / denom

            # update the prediction cache
            for u in user_list:
                prediction_users[u] += U[u, f] * V[i, f]

            #update the SV cache
        for f in range(self.factors):
            for k in range(f):
                val = SV[f, k] - oldVector[f] * oldVector[k] * Wi[i] + V[i, f] * V[i, k] * Wi[i]
                SV[f, k] = val
                SV[k, f] = val

        return V[i]

    def fit(self, max_iterations, sc):
        '''
        Initialize matrices and apply eALS for Matrix Factorization using RDDs
        in parallel for efficient memory use and computation speed.

        '''
        # initialization of parameters
        W, Wi = self.weight_init()
        prediction_items = np.zeros(self.nb_items)
        rating_items = np.zeros(self.nb_items)
        w_items = np.zeros(self.nb_items)
        prediction_users = np.zeros(self.nb_users)
        rating_users = np.zeros(self.nb_users)
        w_users = np.zeros(self.nb_users)
        U = np.random.normal(self.init_mean, self.init_std, self.nb_users * self.factors).reshape(
            (self.nb_users, self.factors))
        V = np.random.normal(self.init_mean, self.init_std, self.nb_items * self.factors).reshape(
            (self.nb_items, self.factors))
        SU = U.T @ U
        SV = np.zeros(shape=(self.factors, self.factors))

        # building SV
        for f in range(self.factors):
            for k in range(f):
                val = 0
                for i in range(self.nb_items):
                    val += V[i, f] * V[i, k] * Wi[i]
                SV[f, k] = val
                SV[k, f] = val

        # broadcasting SU and SV
        SU = sc.broadcast(SU)
        SV = sc.broadcast(SV)

        U_update = U
        V_update = V

        for iteration in range(max_iterations):
            # build a RDD for users and update users matrix in parallel
            U = sc.parallelize(range(self.nb_users)).map(
                lambda u: self.update_user(u, W, Wi, U_update, V_update, SU, SV, prediction_items, rating_items,
                                           w_items)).collect()
            U_update = np.array(U)
            # build a RDD for items and update items matrix in parallel
            V = sc.parallelize(range(self.nb_items)).map(
                lambda i: self.update_item(i, W, Wi, U_update, V_update, SU, SV, prediction_users, rating_users,
                                           w_users)).collect()
            V_update = np.array(V)

            # calculate loss to keep track of training
            current_loss = self.loss(np.array(U), np.array(V), W, Wi, SV.value)
            if iteration%5 == 0:
                print('Iteration:', iteration + 1, 'Loss:', current_loss)

        return np.array(U), np.array(V)