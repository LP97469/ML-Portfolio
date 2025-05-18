import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.distributions as distributions
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import numpy as np


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

    

def discountRewards(rewards, gamma=0.99):
    prev = 0
    outR = []
    for i, r in enumerate(reversed(rewards)):
        d = (prev*gamma) + r
        outR.append(d)
        prev = d
    return list(reversed(outR))


class VPG(nn.Module):
    def __init__(self, output_dim, clip_norm = 0.5):
        super().__init__()

        self.states = []
        self.actions = []
        self.returns = []

        self.nn = NN()
        self.outputHead = nn.LazyLinear(output_dim)
        
        self.optim = optim.AdamW(self.parameters())
        self.clipNorm = clip_norm

    def addExperience(self, states, actions, returns):
        self.states.extend(states)
        self.actions.extend(actions)
        self.returns.extend(returns)

    def clearBuffers(self):
        self.states = []
        self.actions = []
        self.returns = []


    def forward(self, x):
        x = self.nn(x)
        x = self.outputHead(x)
        return x

    def takeAction(self, x):
        self.eval()
        logits = self(x)
        dist = distributions.Categorical(logits=logits)
        action = dist.sample()
        self.train()
        return action.item()
    
    def train_step(self):

        self.optim.zero_grad()

        stateTensor = torch.tensor(np.array(self.states), dtype=torch.float32).squeeze().detach()
        actionTensor = torch.tensor(np.array(self.actions), dtype=torch.int64).squeeze().detach()
        returnTensor = torch.tensor(np.array(self.returns), dtype=torch.float32).squeeze().detach()

        logits = self(stateTensor)

        dist = distributions.Categorical(logits=logits)

        logProbs = dist.log_prob(actionTensor)
        
        loss = ((-logProbs) * returnTensor).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.clipNorm)
        self.optim.step()

        self.clearBuffers()

        return loss



            
if __name__ == "__main__":

    vpg = VPG(2)
    batch_size = 16
    num_iterations = 100

    env = gym.make("CartPole-v1")
    outerPbar = tqdm(range(num_iterations), desc="Iterations", position=0, postfix={"Mean Return":0})
    for iterations in outerPbar:
        returnMetric = []
        returnMean = 0
        pbar = tqdm(range(batch_size), desc="Episodes", position=1, leave=False, postfix={"Mean Return":returnMean})
        for _ in pbar:
            done = False

            obs, info = env.reset()
            states, actions, rewards = [],[],[]
            rewardSum = 0

            while not done:
                action = vpg.takeAction(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                newObs, reward, terminated, truncated, info = env.step(action)
                rewardSum += reward
                done = terminated or truncated
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                obs = newObs
            
            returnMetric.append(rewardSum)
            returnMean = sum(returnMetric)/len(returnMetric)
            pbar.set_postfix({"Mean Return":returnMean})
            returns = discountRewards(rewards)

            vpg.addExperience(states, actions, returns)

        vpg.train_step()
        outerPbar.set_postfix({"Mean Return":returnMean})


    env.close()