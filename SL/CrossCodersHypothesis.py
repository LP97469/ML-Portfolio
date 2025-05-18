import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from sklearn.datasets import make_regression
#from sklearn.decomposition import PCA
# unused for now but could precompress features to save on TSNE time
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

# Next step would be model diffing

n_samples = 4000
n_features=8
noise=0.2
random_state=0


regressor_hidden_dim = 32
regressor_num_layers = 2
regressor_n_epochs = 500
regressor_batch_size = 256


sparse_autoencoder_features = 128
sparse_autoencoder_l1 = 0.2
sparse_autoencoder_n_epochs = 500
sparse_autoencoder_batch_size = 256

top_n_features_to_plot = 6



class NN(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=2):
        super().__init__()
        self.inLayer = nn.LazyLinear(hidden_dim)
        self.FFNUpperLayers = nn.ModuleList([nn.LazyLinear(hidden_dim*2) for _ in range(num_layers)])
        self.FFNLowerLayers = nn.ModuleList([nn.LazyLinear(hidden_dim) for _ in range(num_layers)])
    
    def forward(self, x):
        x = self.inLayer(x)

        for L1, L2 in zip(self.FFNUpperLayers, self.FFNLowerLayers):
            x2 = f.leaky_relu(L1(x))
            x2 = f.layer_norm(L2(x2), x.shape[1:])
            x = x + x2
        
        return x
    def activationsForward(self, x):
        x = self.inLayer(x)
        intermediateActivations = []
        for L1, L2 in zip(self.FFNUpperLayers, self.FFNLowerLayers):
            x2 = f.leaky_relu(L1(x))
            x2 = f.layer_norm(L2(x2), x.shape[1:])
            x = x + x2
            intermediateActivations.append(x2)
        
        return x, torch.concat(intermediateActivations, dim=-1)

class Regressor(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=2):
        super().__init__()
        self.nn = NN(hidden_dim=hidden_dim, num_layers=num_layers)
        self.valueHead = nn.LazyLinear(1)
        self.optim = optim.AdamW(self.parameters(), lr=3e-4)
    def forward(self, x):
        x = self.nn(x)
        return self.valueHead(x).squeeze(-1)
    def trainStep(self, X, y):
        self.optim.zero_grad()
        yHat = self(X)
        loss = torch.square(y-yHat).mean()
        loss.backward()
        self.optim.step()
        return loss.item()
    def activationForward(self, x):
        x, intermediateActivations = self.nn.activationsForward(x)
        return intermediateActivations




X, y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    noise=noise,
    random_state=random_state
)

df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
df["y"] = y

tsne = TSNE(n_components=2)

print("Started TSNE")
tsnedX = tsne.fit_transform(X)
print("Finished TSNE")


regressor = Regressor(hidden_dim=regressor_hidden_dim, num_layers=regressor_num_layers)

outerPbar = tqdm(range(regressor_n_epochs), position=0, postfix={"Loss":0})
for epoch in outerPbar:
    shuffledX, shuffledY = shuffle(X,y)
    
    for batch in range(0, len(X), regressor_batch_size):
        batchX = shuffledX[batch:batch+regressor_batch_size]
        batchY = shuffledY[batch:batch+regressor_batch_size]
        loss = regressor.trainStep(torch.tensor(batchX, dtype=torch.float32), torch.tensor(batchY, dtype=torch.float32))
        outerPbar.set_postfix({"Loss":loss})
        
    

intermediateActivations = []
outerPbar = tqdm(range(0, len(X), regressor_batch_size), position=0)
for batch in outerPbar:
    batchX = X[batch:batch+regressor_batch_size]
    with torch.no_grad():
        intermediateActivations.append(regressor.activationForward(torch.tensor(batchX, dtype=torch.float32)))

intermediateActivations = torch.concat(intermediateActivations, dim=0)




class SparseAutoencoder(nn.Module):
    def __init__(self, num_features, num_outputs, sparsity_term=0.1):
        super().__init__()
        self.encoder = nn.LazyLinear(num_features)
        self.activation = nn.ReLU()
        self.decoder = nn.LazyLinear(num_outputs)
        self.optim = optim.AdamW(self.parameters())
        self.sparsityTerm = sparsity_term
    def forward(self, x):
        features = self.activation(self.encoder(x))
        outputs = self.decoder(features)
        return outputs, features
    def trainStep(self, X):
        self.optim.zero_grad()
        outputs, features = self(X)
        sparsity = (features <= 0.01).float().mean()
        nonzero_per_sample = (features >= 0.01).sum(dim=1) 
        avg_nonzero = nonzero_per_sample.float().mean() 
        loss = torch.square(outputs-X).mean() + (torch.abs(features).mean() * self.sparsityTerm)
        loss.backward()
        self.optim.step()
        return loss.item(), sparsity.item(), avg_nonzero.item()






sparseAutoencoder = SparseAutoencoder(sparse_autoencoder_features, intermediateActivations.shape[-1], sparsity_term=sparse_autoencoder_l1)

outerPbar = tqdm(range(sparse_autoencoder_n_epochs), position=0, postfix={"Loss":0, "Sparsity":0, "avg non zero":0})
for epoch in outerPbar:
    shuffledIntermediateActivations = shuffle(intermediateActivations)
    
    for batch in range(0, len(X), sparse_autoencoder_batch_size):
        batchIntermediateActivations = shuffledIntermediateActivations[batch:batch+sparse_autoencoder_batch_size]
        loss, sparsity, avg_nonzero = sparseAutoencoder.trainStep(batchIntermediateActivations)
        outerPbar.set_postfix({"Loss":loss, "Sparsity":sparsity, "avg non zero":avg_nonzero})



Features = []
outerPbar = tqdm(range(0, len(X), sparse_autoencoder_batch_size), position=0)
for batch in outerPbar:
    batchIntermediateActivations = intermediateActivations[batch:batch+sparse_autoencoder_batch_size]
    with torch.no_grad():
        Features.append(sparseAutoencoder(batchIntermediateActivations)[1])

Features = torch.concat(Features, dim=0)





variances = torch.var(Features, dim=0, unbiased=False)  



top_idxs = torch.argsort(variances, descending=True)[:top_n_features_to_plot] 
top_idxs = top_idxs.tolist()


cols = min(top_n_features_to_plot, 3)
rows = int(np.ceil(top_n_features_to_plot/cols))
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)


for plot_i, feat_idx in enumerate(top_idxs):
    r, c = divmod(plot_i, cols)
    ax = axes[r][c]
    sc = ax.scatter(
        tsnedX[:,0], tsnedX[:,1],
        c=Features[:,feat_idx],
        cmap='viridis',
        s=20, edgecolor='none'
    )
    ax.set_title(f'Feature {feat_idx} (var={variances[feat_idx]:.3f})')
    ax.set_xlabel('TSNE component 1')
    ax.set_ylabel('TSNE component 2')
    plt.colorbar(sc, ax=ax, label=f'Feat {feat_idx}')


for plot_i in range(top_n_features_to_plot, rows*cols):
    r, c = divmod(plot_i, cols)
    axes[r][c].axis('off')

plt.tight_layout()
plt.show()