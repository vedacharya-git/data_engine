import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        return self.model(x)

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        return self.model(x)

def create_masks(N_d, m):
    return torch.randint(0, 2, (m, N_d), dtype=torch.float32)

def sample_z(m, latent_dim):
    return torch.randn(m, latent_dim)

def create_pacs(data, pac):
    m = data.size(0)
    return torch.stack([torch.bitwise_xor(data[k*pac:(k+1)*pac].int()).float() for k in range(m // pac)])

def gradient_penalty(critic, real, fake, cond):
    alpha = torch.rand(real.size(0), 1, device=real.device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = critic(interpolates, cond)
    fake_grad = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad_penalty = ((fake_grad.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty

def preprocess_data(data):
    # Separate numerical and categorical columns
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = data.select_dtypes(include=['object']).columns

    # Encode categorical variables
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col])

    # Scale numerical variables
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])

    return data, num_cols, cat_cols, le, scaler

def algorithm(data, pac, latent_dim, num_epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess data
    data, num_cols, cat_cols, le, scaler = preprocess_data(data)
    
    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    N_d = data.shape[1]
    
    generator = Generator(latent_dim, N_d).to(device)
    critic = Critic(N_d).to(device)
    
    optim_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_c = optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for epoch in range(num_epochs):
        for i in range(0, len(train_data), batch_size):
            batch = train_data.iloc[i:i+batch_size]
            m = len(batch)
            
            # Convert batch to tensor
            real_data = torch.tensor(batch.values, dtype=torch.float32).to(device)
            
            # Step 1-2
            masks = create_masks(N_d, m).to(device)
            cond = masks
            
            # Step 3-5
            z = sample_z(m, latent_dim).to(device)
            fake_data = generator(z, cond)
            
            # Step 6-8
            cond_pac = create_pacs(cond, pac)
            fake_data_pac = create_pacs(fake_data, pac)
            real_data_pac = create_pacs(real_data, pac)
            
            # Critic training
            for _ in range(5):
                optim_c.zero_grad()
                
                # Step 9
                critic_real = critic(real_data_pac, cond_pac)
                critic_fake = critic(fake_data_pac, cond_pac)
                critic_loss = -torch.mean(critic_real) + torch.mean(critic_fake)
                
                # Step 10-12
                gp = gradient_penalty(critic, real_data_pac, fake_data_pac, cond_pac)
                critic_loss += 10 * gp
                
                critic_loss.backward(retain_graph=True)
                optim_c.step()
            
            # Generator training
            optim_g.zero_grad()
            
            # Step 14
            fake_data = generator(z, cond)
            fake_data_pac = create_pacs(fake_data, pac)
            
            # Step 15
            generator_loss = -torch.mean(critic(fake_data_pac, cond_pac))
            
            generator_loss.backward()
            optim_g.step()
        
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Critic Loss: {critic_loss.item():.4f}, Generator Loss: {generator_loss.item():.4f}")
    
    return generator, critic, le, scaler

# Load data
data = pd.read_csv('healthcare_dataset.csv')

# Set hyperparameters
pac = 10
latent_dim = 64
num_epochs = 1000
batch_size = 32

# Run the algorithm
generator, critic, le, scaler = algorithm(data, pac, latent_dim, num_epochs, batch_size)

# Generate synthetic data
def generate_synthetic_data(generator, latent_dim, num_samples, le, scaler, num_cols, cat_cols):
    z = sample_z(num_samples, latent_dim)
    cond = create_masks(data.shape[1], num_samples)
    
    with torch.no_grad():
        synthetic_data = generator(z, cond).cpu().numpy()
    
    # Create a DataFrame from the synthetic data
    synthetic_df = pd.DataFrame(synthetic_data, columns=data.columns)
    
    # Inverse transform numerical columns
    synthetic_df[num_cols] = scaler.inverse_transform(synthetic_df[num_cols])
    
    # Inverse transform categorical columns
    for col in cat_cols:
        synthetic_df[col] = le.inverse_transform(synthetic_df[col].astype(int))
    
    return synthetic_df

# Generate 1000 synthetic samples
synthetic_data = generate_synthetic_data(generator, latent_dim, 1000, le, scaler, num_cols, cat_cols)

# Save synthetic data to CSV
synthetic_data.to_csv('synthetic_data.csv', index=False)

print("Synthetic data generated and saved to 'synthetic_data.csv'")