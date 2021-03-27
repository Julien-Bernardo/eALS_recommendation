import numpy as np
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, ArrayType
from pyspark.sql import SQLContext
import pandas as pd

class ALS():
    def __init__(self, R, rating_list, factors=10, init_mean=0,
                 init_std=0.01, reg=0.01):
        self.R = R
        self.nb_items = R.shape[1]  # nb of items
        self.nb_users = R.shape[0]  # nb of users
        self.factors = factors
        self.reg = reg
        self.init_mean = init_mean
        self.init_std = init_std
        self.rating_list = rating_list  # rating_list = rating_rdd.collect()

    def weight_init(self):
        # creating 0 matrix user * items
        A = np.full((self.nb_users, self.nb_items), 0, dtype=np.float64)
        # all ratings set to weight = 1
        for data in self.rating_list:
            A[data[1], data[2]] = 1
        return A

    def update_user(self, A, u, U, V):
        Au = A[u, :]
        U[u] = np.linalg.solve(np.dot(V, np.dot(np.diag(Au), V.T)) + \
                               self.reg * np.eye(self.factors), \
                               np.dot(V, np.dot(np.diag(Au), self.R[u].T))).T
        return U[u].tolist()

    def update_item(self, A, i, U, V):
        Ai = A.T[i, :]
        V[:, i] = np.linalg.solve(np.dot(U.T, np.dot(np.diag(Ai), U)) + \
                                  self.reg * np.eye(self.factors), \
                                  np.dot(U.T, np.dot(np.diag(Ai), self.R[:, i])))
        return V[:, i].tolist()

    def fit_rdd(self, iterations, sc):
        A = self.weight_init()
        U = np.random.normal(self.init_mean, self.init_std, \
                             self.nb_users * self.factors).reshape((self.nb_users, self.factors))
        V = np.random.normal(self.init_mean, self.init_std, \
                             self.nb_items * self.factors) \
            .reshape((self.nb_items, self.factors)).T

        U_update = U
        V_update = V

        for iteration in range(iterations):
            # build a RDD for users and update users matrix in parallel
            U = sc.parallelize(range(self.nb_users)).map(lambda u: self.update_user(A, u, U_update, V_update)).collect()
            U_update = np.array(U)

            V = sc.parallelize(range(self.nb_items)).map(lambda i: self.update_item(A, i, U_update, V_update)).collect()
            V_update = np.array(V).T

            #print('Iteration:', iteration + 1)

        return U_update, V_update

    def fit_df(self, iterations, sc, spark, sqlContext):
        A = self.weight_init()
        U = np.random.normal(self.init_mean, self.init_std, \
                             self.nb_users * self.factors).reshape((self.nb_users, self.factors))
        V = np.random.normal(self.init_mean, self.init_std, \
                             self.nb_items * self.factors) \
            .reshape((self.nb_items, self.factors)).T

        U_update = U
        V_update = V

        df_U = spark.range(self.nb_users).toDF('Users')
        df_V = spark.range(self.nb_items).toDF('Items')

        for iteration in range(iterations):
            # build a RDD for users and update users matrix in parallel
            df_update_user = udf(lambda u: self.update_user(A, u, U_update, V_update), ArrayType(FloatType()))
            spark.udf.register("df_update_user", df_update_user)
            df_U = df_U.withColumn('Factors', df_update_user('Users'))
            U_update = np.array(np.squeeze(df_U.select('Factors').collect()))

            df_update_item = udf(lambda i: self.update_item(A, i, U_update, V_update), ArrayType(FloatType()))
            spark.udf.register("df_update_item", df_update_item)
            df_V = df_V.withColumn('Factors', df_update_item('Items'))
            V_update = np.array(np.squeeze(df_V.select('Factors').collect())).T


            #print('Iteration:', iteration + 1)

        return U_update, V_update