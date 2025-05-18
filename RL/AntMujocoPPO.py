
from CartPolePPO import PPO
import torch
from tqdm import tqdm
import gymnasium as gym
import time
from torch.optim.lr_scheduler import LambdaLR




if __name__ == "__main__":

    ppo = PPO(8, discrete=False, batch_size=128)
    batch_size = 32
    num_iterations = 500

    lr_lambda = lambda current_iter: 1 - min(current_iter, num_iterations) / num_iterations

    scheduler = LambdaLR(ppo.optim, lr_lambda=lr_lambda)

    env = gym.make("Ant-v5")#, render_mode="human")
    outerPbar = tqdm(range(num_iterations), desc="Iterations", position=0, postfix={"Mean Return":0, "LR":0})
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
        scheduler.step()
        outerPbar.set_postfix({"Mean Return":returnMean, "LR":[i["lr"] for i in ppo.optim.param_groups]})


    env.close()