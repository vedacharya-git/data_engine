import os
import pandas as pd

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
from torch.nn import functional
from torch import optim
from torch.utils.data import TensorDataset


# Loading data in pandas dataframe
current_directory = os.path.abspath(os.getcwd())
parent_directory = os.path.join(current_directory, '..')
grandparent_directory = os.path.join(parent_directory, '..')
data_directory = os.path.join(grandparent_directory, 'data')
csv_path = os.path.join(data_directory, 'healthcare_dataset.csv')

df = pd.read_csv(csv_path)

df.drop(['Name', 'Hospital', 'Doctor'], axis=1, inplace=True)

class DataProcessor:
    def __init__(self, df):
        self.df = df
        self.mappings = {}
        self.DTYPES = {
            'i': 'numerical', 'f': 'numerical', 'o': 'object',
            'b': 'boolean', 'M': 'datetime'
        }
        self.VGM_parameters = {}
        self.OHE = {}
        self.original_columns = list(df.columns)
        for i in self.df.columns:
            if 'date' in i.lower():
                self.mappings[i] = 'datetime'
            else:
                self.mappings[i] = self.DTYPES[str(self.df[i].dtype)[0]]

    def fit(self):
        GMM = GaussianMixture(n_components=3, random_state=42)
        for i in self.mappings:
            if self.mappings[i] in ['numerical', 'datetime']:
                if self.mappings[i] == 'datetime':
                    self.df[i] = pd.to_datetime(self.df[i])
                    self.df[i] = self.df[i].astype(int) // 10**9
                GMM.fit(self.df[[i]])
                self.VGM_parameters[i] = {
                    'means': GMM.means_.flatten(),
                    'std_devs': np.sqrt(GMM.covariances_).flatten(),
                    'weights': GMM.weights_
                }
            elif self.mappings[i] == 'object':
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
        return self.transformer()

## Generator
class Generator(nn.Module):
    def __init__(self, z_dim, cond_dim):
        super(Generator, self).__init__()
        
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        
        if self.z_dim is None or self.cond_dim is None:
            exit("Please enter the dimensions of z and cond")
        
        self.h0 = nn.Linear(z_dim + cond_dim, 256)
        self.h1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.h2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.alpha = nn.Sequential(nn.Linear(256, 512), nn.Tanh())
        self.beta = nn.Sequential(nn.Linear(256, 512), nn.Sigmoid())
        self.delta = nn.Sequential(nn.Linear(256, 512), nn.Sigmoid())
        self.out = nn.Linear(512, cond_dim)
        
    def forward(self, z, cond):
        h0 = torch.cat([z, cond], dim=1)
        h0 = self.h0(h0)
        
        h1 = self.h1(h0)
        h2 = self.h2(h1)
        
        alpha = self.alpha(h2)
        beta = self.beta(h2)
        delta = self.delta(h2)
        
        # Combine α, β, and δ to produce the final output
        out = alpha * beta + (1 - alpha) * delta
        
        return self.out(out)

## CRTIC
class Critic(nn.Module):
    def __init__(self, cond_dim, pac):
        super(Critic, self).__init__()
        
        self.input_dim = cond_dim * pac  # Update this to match the pac
        self.h0 = nn.Linear(self.input_dim, 256)
        self.h1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.h2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.out = nn.Linear(256, 1)

    def forward(self, cond):
        h0 = self.h0(cond)
        h1 = self.h1(h0)
        h2 = self.h2(h1)
        
        return self.out(h2)

class PacGAN(nn.Module):
    def __init__(self, generator, critic, pac):
        super(PacGAN, self).__init__()
        
        self.generator = generator
        self.critic = critic
        self.pac = pac

    def forward(self, z, cond):
        fake_cond = self.generator(z, cond)
        fake_cond_pac = fake_cond.view(-1, self.pac, cond.size(1))
        real_cond_pac = cond.view(-1, self.pac, cond.size(1))
        
        fake_output = self.critic(fake_cond_pac.view(-1, self.pac * cond.size(1)))
        real_output = self.critic(real_cond_pac.view(-1, self.pac * cond.size(1)))
        
        return fake_output, real_output

class CTGAN:
    def __init__(self, df, epochs=300, pac=10, batch_size=500):
        self.df = df
        self.size = df.shape[0]
        self.epochs = epochs
        self.pac = pac
        self.batch_size = batch_size

        self.df = self.df.sample(frac=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_mask(self):
        masks = []
        ind = np.random.choice(self.df.index, self.batch_size, replace=False)
        batch = self.df.loc[ind]
        number_of_batches = max(self.df.shape[0] // self.batch_size, 1)
        
        for _ in range(number_of_batches):
            batches = []
            for _, rows in batch.iterrows():
                condvec = rows.values.flatten().tolist()
                batches.append(condvec)
            masks.append(batches)
        return np.array(masks)

    def sampler(self):
        return torch.randn(self.batch_size, 100)    # Set here a value of 100

    def create_batches(self):
        normalized_df = (self.df - self.df.min()) / (self.df.max() - self.df.min())
        tensor_data = torch.tensor(normalized_df.values, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def gradient_penalty(self, critic, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1).to(real_data.device)
        alpha = alpha.expand(real_data.size())
    
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        d_interpolates = critic(interpolates)
        fake = torch.ones(d_interpolates.size()).to(real_data.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # Training the CTGAN model
    def train(self):
        data_loader = self.create_batches()
        latent_dim = 100
        cond_dim = self.df.shape[1]
    
        generator = Generator(latent_dim, cond_dim)
        critic = Critic(cond_dim, self.pac)  # Pass the pac value here
        pacgan = PacGAN(generator, critic, self.pac)
    
        generator.to(self.device)
        critic.to(self.device)

        optimizerG = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6)
        optimizerC = optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6)
    
        for epoch in range(self.epochs):
            for real_data in data_loader:
                real_data = real_data[0].to(self.device)
                z = self.sampler().to(self.device)

                optimizerG.zero_grad()
                optimizerC.zero_grad()

                fake_cond, real_cond = pacgan(z, real_data)

                lossG = -torch.mean(real_cond)
                lossG.backward()
                optimizerG.step()

                optimizerC.zero_grad()
            
                fake_cond, real_cond = pacgan(z, real_data)
            
                lossC = -torch.mean(real_cond) + torch.mean(fake_cond)
                gp = self.gradient_penalty(critic, real_data, fake_cond)
                lossC += gp
                lossC.backward()
                optimizerC.step()
            
                print(f"Epoch: {epoch}, LossG: {lossG.item()}, LossC: {lossC.item()}")

# Testing the code
encoded_data = DataProcessor(df).fit_transform()
ctgan = CTGAN(encoded_data)
ctgan.train()