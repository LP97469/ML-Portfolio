import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.distributions as distributions
import torch.optim as optim
#from galore_torch import GaLoreAdamW
import gymnasium as gym
from tqdm import tqdm
import numpy as np

#TODO next step add in a drop in slot for a feature extractor

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



def compute_gae(deltas, dones, gamma, lam):
    T = deltas.size(0)
    adv = torch.zeros_like(deltas)
    lastgaelam = 0

    for t in reversed(range(T)):
        mask = 1.0 - dones[t].float()
        lastgaelam = deltas[t] + gamma * lam * lastgaelam * mask
        adv[t] = lastgaelam
    return adv

class PPO(nn.Module):
    def __init__(self, output_dim, clip_norm = 0.5, gamma=0.99, gae_lambda=0.95, actor_lr=5e-4, critic_lr=1e-3, discrete=True, batch_size=32, hidden_dim=32, num_layers=2):
        super().__init__()

        self.states = []
        self.actions = []
        self.rewards = []
        self.nextStates = []
        self.dones = []
        self.priorLogProbs = []

        self.outputNN = NN(hidden_dim=hidden_dim, num_layers=num_layers)
        self.outputHead = nn.LazyLinear(output_dim)
        self.valueNN = NN(hidden_dim=hidden_dim, num_layers=num_layers)
        self.valueHead = nn.LazyLinear(1)

        self.discrete = discrete
        if discrete:
            self.distFunction = distributions.Categorical
        else:
            self.distFunction = distributions.Multinomial
        

        self.optim = optim.AdamW([
            {   
                'params': list(self.outputNN.parameters()) + list(self.outputHead.parameters()),
                'lr': actor_lr
            },
            {   
                'params': list(self.valueNN.parameters()) + list(self.valueHead.parameters()),
                'lr': critic_lr
            }
        ])
        
        self.clipNorm = clip_norm
        self.gamma = gamma
        self.gaeLambda = gae_lambda
        self.batch_size = batch_size

    def addExperience(self, states, actions, rewards, nextStates, dones, logProbs):
        self.states.extend(states)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.nextStates.extend(nextStates)
        self.dones.extend(dones)
        self.priorLogProbs.extend(logProbs)

    def clearBuffers(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.nextStates = []
        self.dones = []
        self.priorLogProbs = []


    def forward(self, x):
        outputFeatures = self.outputNN(x)
        logits = self.outputHead(outputFeatures)
        valueFeatures = self.valueNN(x)
        value = self.valueHead(valueFeatures).squeeze()
        return logits, value

    def takeAction(self, x):
        with torch.no_grad():
            logits, _ = self(x)
        dist   = self.distFunction(logits=logits)
        action = dist.sample()
        logProbs = dist.log_prob(action)
        if self.discrete:
            return action.item(), logProbs.item()
        else:
            return action.numpy().squeeze(), logProbs.numpy().squeeze()
    
    def train_step(self):
        self.optim.zero_grad()
        device = next(self.parameters()).device

        S  = torch.tensor(np.array(self.states),   dtype=torch.float32, device=device).squeeze().detach()
        A  = torch.tensor(np.array(self.actions),  dtype=torch.int64,   device=device).squeeze().detach()
        R  = torch.tensor(np.array(self.rewards),  dtype=torch.float32, device=device).squeeze().detach()
        S2 = torch.tensor(np.array(self.nextStates),dtype=torch.float32, device=device).squeeze().detach()
        D  = torch.tensor(np.array(self.dones),    dtype=torch.float32, device=device).squeeze().detach()
        L  = torch.tensor(np.array(self.priorLogProbs),dtype=torch.float32, device=device).squeeze().detach()


        #print(S.shape, A.shape, R.shape, S2.shape, D.shape)

        logits, V   = self(S)
        _,      V2  = self(S2)

        V  = V.squeeze()
        V2 = V2.squeeze().detach()

        td_target  = R + ((self.gamma * V2 * (1.0 - D)) )
        deltas         = td_target - V

        A2 = compute_gae(deltas, D, self.gamma, self.gaeLambda)

        A2         = A2.squeeze()
        VT         = V + A2
        NA         = (A2 - A2.mean())/(A2.std()+0.001)
        for epoch in range(8):
            n = NA.shape[0]
            perm = torch.randperm(n) 

            for batch in range(0, n, self.batch_size):
                batchS = S[perm[batch:batch+self.batch_size]].detach()
                batchA = A[perm[batch:batch+self.batch_size]].detach()
                batchNA = NA[perm[batch:batch+self.batch_size]].detach()
                batchVT = VT[perm[batch:batch+self.batch_size]].detach()
                batchPriorLogP = L[perm[batch:batch+self.batch_size]].detach()


                batchlogits, batchV   = self(batchS)

                batchdist       = self.distFunction(logits=batchlogits)
                batchlogp       = batchdist.log_prob(batchA)

                #print(A2.shape, logp.shape)
                ratio = torch.exp(batchlogp-batchPriorLogP)
                surrogate1 = (ratio *batchNA.detach())
                surrogate2 = torch.clip(ratio, 0.8, 1.2)*batchNA.detach()
                policy_loss = -torch.minimum(surrogate1, surrogate2).mean()

                value_loss  = 0.5 * (batchVT - batchV).pow(2).mean()

                entropy_loss = (-batchdist.entropy()*0.01).mean()


                loss        = policy_loss + value_loss + entropy_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.clipNorm)
                self.optim.step()
        self.clearBuffers()
        return loss



            
if __name__ == "__main__":

    ppo = PPO(2)
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
            states, actions, rewards, nextStates, dones, logProbs = [],[],[],[],[],[]
            rewardSum = 0

            while not done:
                action, logProb = ppo.takeAction(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                newObs, reward, terminated, truncated, info = env.step(action)
                rewardSum += reward
                done = terminated or truncated
                #env.render()
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                nextStates.append(newObs)
                dones.append(int(done))
                logProbs.append(logProb)
                obs = newObs
            
            returnMetric.append(rewardSum)
            returnMean = sum(returnMetric)/len(returnMetric)
            pbar.set_postfix({"Mean Return":returnMean})

            ppo.addExperience(states, actions, rewards, nextStates, dones, logProbs)

        ppo.train_step()
        outerPbar.set_postfix({"Mean Return":returnMean})


    env.close()