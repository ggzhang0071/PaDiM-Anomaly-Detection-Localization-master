import torch
import torch.nn as nn

class net(nn.Module):
    def __init__(self, model_dim, embedding_dim, output_dim):
        super(net, self).__init__()
        self.fc1 = nn.Linear(model_dim, embedding_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        a = self.fc2(x)
        return a, x

class gaussian_net(nn.Module):
    def __init__(self, model_dim, embedding_dim, output_dim, L=1):
        super(gaussian_net, self).__init__()
        self.mu = nn.Linear(model_dim, embedding_dim)
        self.sigma = nn.Linear(model_dim, embedding_dim)
        self.L = L
        self.leaky_relu = nn.LeakyReLU()
        self.epsilon_dist = torch.distributions.MultivariateNormal(torch.zeros(embedding_dim),torch.eye(embedding_dim))
        self.fc = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, x):
        mu = self.leaky_relu(self.mu(x))
        sigma = self.leaky_relu(self.sigma(x))
        epsilon = self.epsilon_dist.sample((self.L,))
        epsilon = torch.sum(epsilon, dim=0) / self.L
        epsilon = epsilon.cuda()
        gaus_embedding = mu + epsilon * sigma
        z = self.fc(gaus_embedding)
        return z, (mu, sigma)
        



if __name__ == '__main__':
    n = net(2048, 64, 9)