import pandas as pd
import numpy as np


def data_preprocessing(yelp_file_txt, data_path):
    '''
    Function outputs a clean dataframe with columns user, item, rating
    from the yelp dataset
    '''

    # Loading yelp datset and keep only required columns
    raw_data = pd.read_table(yelp_file_txt)
    raw_data = raw_data.drop(raw_data.columns[-1], axis=1)
    raw_data.columns = ['user', 'item', 'rating']

    sample = raw_data[:5000]
    # Drop rows with missing values
    data = sample.dropna()

    # Convert artists names into numerical IDs
    data['user_id'] = data['user'].astype("category").cat.codes
    data['item_id'] = data['item'].astype("category").cat.codes

    # Keeping only user_id and item_id
    data = data.drop(['user', 'item'], axis=1)

    data.to_csv(data_path, sep=",", header=True, index=False)
    return data

class tools():

    def __init__(self, rating_list, data_path):
        self.nb_items = len(pd.read_csv(data_path)['item_id'].unique())  # nb of items
        self.nb_users = len(pd.read_csv(data_path)['user_id'].unique())  # nb of users
        self.rating_list = rating_list

    def set_data_matrix(self):
        '''
        Function returns matrix MxN (users vs items) with true ratings
        '''
        R = np.full((self.nb_users, self.nb_items), 0, dtype=np.float64)  # creating 0 matrix user * items
        for data in self.rating_list:
            R[data[1], data[2]] = data[0]
        return R

    def set_R_hat(self, U, V):
        '''
        Function returns matrix MxN (users vs items) with predicted ratings
        '''
        R_hat_empty = np.full((self.nb_users, self.nb_items), 0, dtype=np.float64)  # creating 0 matrix user * items
        pred_matrix = U @ V.T
        for data in self.rating_list:
            R_hat_empty[round(data[1]), round(data[2])] = pred_matrix[data[1], data[2]]
        return R_hat_empty

    def rmse(self, U, V):
        '''
        Compute RMSE between predicted ratings R_hat and true ratings R
        '''
        R_hat = self.set_R_hat(U, V)
        R = self.set_data_matrix()
        diff = R - R_hat
        return np.sqrt(np.sum(np.power(diff, 2)) / (self.nb_users * self.nb_items))



