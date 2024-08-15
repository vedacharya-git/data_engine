import os
import pandas as pd

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder

class DataProcessor:
    def __init__(self, df):
        self.df = df
        self.mappings = {}
        self.DTYPES = {
            'i': 'numerical',
            'f': 'numerical',
            'o': 'object',
            'b': 'boolean',
            'M': 'datetime'
        }
        self.VGM_parameters = {}
        self.OHE = {}
        self.original_columns = list(df.columns)
        self.m = 0  # Number of numerical columns
        self.D = 0  # Number of discrete columns
        for i in self.df.columns:
            if 'date' in i.lower():
                self.mappings[i] = 'datetime'
            else:
                self.mappings[i] = self.DTYPES[str(self.df[i].dtype)[0]]

    def fit(self):
        GMM = GaussianMixture(n_components=3, random_state=42)
        for i in self.mappings:
            if self.mappings[i] in ['numerical', 'datetime']:
                self.m += 1
                if self.mappings[i] == 'datetime':
                    self.df[i] = pd.to_datetime(self.df[i])
                    self.df[i] = self.df[i].astype(int) // 10**9
                GMM.fit(self.df[[i]])
                self.VGM_parameters[i] = {
                    'means': GMM.means_.flatten(),
                    'std_devs': np.sqrt(GMM.covariances_).flatten(),
                    'weights': GMM.weights_
                }
            elif self.mappings[i] in ['object', 'boolean']:
                self.D += 1
                self.OHE[i] = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                self.OHE[i].fit(self.df[[i]])

    def transformer(self, df_to_transform=None):
        if df_to_transform is None:
            df_to_transform = self.df
        transformed_df = df_to_transform.copy()
        for i in self.original_columns:
            if i in self.mappings:
                if self.mappings[i] in ['numerical', 'datetime']:
                    if i in self.VGM_parameters:
                        for j in range(3):
                            transformed_df[f"{i}_VGM_{j}"] = self.VGM_parameters[i]['means'][j]
                elif self.mappings[i] == 'object':
                    if i in self.OHE:
                        ohe_result = self.OHE[i].transform(transformed_df[[i]])
                        ohe_df = pd.DataFrame(ohe_result, columns=self.OHE[i].get_feature_names_out([i]))
                        transformed_df = pd.concat([transformed_df, ohe_df], axis=1)
                        transformed_df.drop(i, axis=1, inplace=True)
        return transformed_df

    def inverse_transform(self, transformed_df):
        inverse_df = pd.DataFrame()
        for i in self.original_columns:
            if i in self.mappings:
                if self.mappings[i] in ['numerical', 'datetime']:
                    inverse_df[i] = transformed_df[i]
                elif self.mappings[i] == 'object':
                    if i in self.OHE:
                        ohe_columns = [col for col in transformed_df.columns if col.startswith(f"{i}_")]
                        ohe_data = transformed_df[ohe_columns]
                        inverse_categories = self.OHE[i].inverse_transform(ohe_data)
                        inverse_df[i] = inverse_categories.flatten()

        # Convert datetime columns back to datetime
        for i in inverse_df.columns:
            if self.mappings.get(i) == 'datetime':
                inverse_df[i] = pd.to_datetime(inverse_df[i], unit='s')

        return inverse_df

    def fit_transform(self):
        self.fit()
        encoded_df = self.transformer()
        self.D = encoded_df.shape[1] - self.m
        return encoded_df, self.m, self.D