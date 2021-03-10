import pandas as pd
import numpy as np

def data_preprocessing(self, yelp_file_txt):
    raw_data = pd.read_table(yelp_file_txt)
    raw_data = raw_data.drop(raw_data.columns[-1], axis=1)
    raw_data.columns = ['user', 'item', 'rating']

    sample = raw_data[:5000]
    # Drop rows with missing values
    data = sample.dropna()

    # Convert artists names into numerical IDs
    data['user_id'] = data['user'].astype("category").cat.codes
    data['item_id'] = data['item'].astype("category").cat.codes

    # Create a lookup frame so we can get the artist names back in
    # readable form later.
    item_lookup = data[['item_id', 'item']].drop_duplicates()
    item_lookup['item_id'] = item_lookup.item_id.astype(str)

    data = data.drop(['user', 'item'], axis=1)

    data.to_csv(self.data_path, sep=",", header=True, index=False)
    return data

class tools():

    def __init__(self, rating_list, data_path):
        self.data_path = data_path
        self.nb_items = len(pd.read_csv(data_path)['item_id'].unique())  # nb of items
        self.nb_users = len(pd.read_csv(data_path)['user_id'].unique())  # nb of users
        self.rating_list = rating_list

    def set_data_matrix(self):

        M = np.full((self.nb_users, self.nb_items), 0, dtype=np.float64)  # creating 0 matrix user * items
        for data in self.rating_list:
            M[data[1], data[2]] = data[0]
        return M

