import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import numpy as np


num_items     = 35
dim           = 6
comparisons_per_judge = 60  


mu    = np.zeros(dim)
cov   = np.eye(dim)
true_embeddings = np.random.multivariate_normal(mu, cov, size=num_items)


triples = []
for judge in range(num_items):
    others = [i for i in range(num_items) if i != judge]
    sampled_pairs = np.random.choice(others, size=(comparisons_per_judge, 2), replace=True)
    for a, b in sampled_pairs:
        triples.append((judge, a, b))

triples = np.array(triples)  

def simulate_choice(j, a, b, embeddings, beta=1.0):
    d1 = np.linalg.norm(embeddings[j] - embeddings[a])
    d2 = np.linalg.norm(embeddings[j] - embeddings[b])
    p = 1 / (1 + np.exp(-beta * (d2 - d1)))
    return np.random.rand() < p

comparisons = np.array([
    simulate_choice(j, a, b, true_embeddings, beta=5.0) 
    for j, a, b in triples
], dtype=np.float32).reshape(-1, 1)






class DispositionVectorLearner(nn.Module):
    def __init__(self, num_items, dimensionality):
        super().__init__()
        self.embeddings = nn.Embedding(num_items, dimensionality, max_norm=10)
        self.Uncertainties = nn.Parameter(torch.randn((num_items,)))
        self.optim = optim.AdamW(self.parameters(), weight_decay=1e-4)
    
    def trainStep(self, pairs, comparisons):
        self.optim.zero_grad()
        judge = self.embeddings(pairs[:, 0])#.detach()
        judgeVar = torch.exp(self.Uncertainties[pairs[:, 0]])+1e-8
        evaluee1 = self.embeddings(pairs[:, 1])
        evaluee1Var = torch.exp(self.Uncertainties[pairs[:, 1]])+1e-8
        evaluee2 = self.embeddings(pairs[:, 2])
        evaluee2Var = torch.exp(self.Uncertainties[pairs[:, 2]])+1e-8


        judge_score1 = f.pairwise_distance(judge, evaluee1)
        judge_score1Var = torch.sqrt(judgeVar+evaluee1Var+1e-8)

        judge_score2 = f.pairwise_distance(judge, evaluee2)
        judge_score2Var = torch.sqrt(judgeVar+evaluee2Var+1e-8)


        p_hat = f.sigmoid(judge_score2 - judge_score1)
        p_hatSigma = torch.sqrt(p_hat.detach()*(1-p_hat.detach())*torch.sqrt(judge_score1Var+judge_score2Var+1e-8))

        z = p_hat / p_hatSigma

        dist = torch.distributions.Normal(0., 1.)
        p = dist.cdf(z)

        eps = 1e-8
        p = p.clamp(min=eps, max=1-eps)

        #loss =  (f.binary_cross_entropy(p.squeeze(), comparisons.squeeze())*(p_hatSigma.detach())).mean() + (self.embeddings.weight.mean()*0.000001) # With beta Nll scaling
        loss =  (f.binary_cross_entropy(p.squeeze(), comparisons.squeeze())).mean() + (self.embeddings.weight.mean()*0.000001) # without beta Nll scaling
        loss.backward()
        self.optim.step()

        metricLoss =  f.binary_cross_entropy(p_hat.squeeze().detach(), comparisons.squeeze().detach()).mean().detach()

        """
        loss =  f.binary_cross_entropy(p_hat.squeeze(), comparisons.squeeze()).mean() + (self.embeddings.weight.mean()*0.000001)
        loss.backward()
        self.optim.step()
        """
        return metricLoss.item()
    
    def returnWeights(self):
        return self.embeddings.weight.detach().cpu().numpy()


# Training

epochs = 2000
batch_size=64

learner = DispositionVectorLearner(num_items, 6)

pbar = tqdm(range(epochs))
for epoch in pbar:
    shuffledPairs, shuffledComparisons = shuffle(triples, comparisons)
    shuffledPairs, shuffledComparisons = torch.tensor(shuffledPairs, dtype=torch.int64), torch.tensor(shuffledComparisons, dtype=torch.float32)
    losses = []
    for batch in range(0, len(shuffledComparisons), batch_size):
        batchPairs, batchComparisons = shuffledPairs[batch:batch+batch_size,:], shuffledComparisons[batch:batch+batch_size,:]

        loss = learner.trainStep(batchPairs, batchComparisons)

        losses.append(loss)


    pbar.set_postfix({"Loss":sum(losses)/len(losses)})

print(learner.Uncertainties.detach().numpy())


# Evaluation
mtx1, mtx2, _ = procrustes(true_embeddings, learner.returnWeights())


d_true = pdist(mtx1, 'euclidean')
d_pred = pdist(mtx2, 'euclidean')

rho, pval = spearmanr(d_true, d_pred)
print(f"Spearman after Procrustes:  {rho:.3f}, p={pval:.1e}")




def procrustes_disparity(true_emb, pred_emb):
    mtx1, mtx2, disparity = procrustes(true_emb, pred_emb)
    return disparity

def orthogonal_procrustes_error(true_emb, pred_emb, scale):

    X = true_emb - true_emb.mean(axis=0, keepdims=True)
    Y = pred_emb - pred_emb.mean(axis=0, keepdims=True)

    R, c = orthogonal_procrustes(Y, X)
    Y_aligned = (Y @ R) * (c if scale else 1.0)

    rmse = np.sqrt(np.mean((X - Y_aligned) ** 2))
    return rmse


disp = procrustes_disparity(true_embeddings, learner.returnWeights())
err  = orthogonal_procrustes_error(true_embeddings, learner.returnWeights())

print(f"Procrustes disparity (sum‐sq error / var): {disp:.4f}")
print(f"Orthogonal‐Procrustes RMSE:         {err:.4f}")


D = pairwise_distances(learner.returnWeights(), metric='euclidean')


mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
matrix = mds.fit_transform(D)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(matrix[:,0],matrix[:,1],matrix[:,2], c=learner.Uncertainties.detach().numpy())


plt.show()



