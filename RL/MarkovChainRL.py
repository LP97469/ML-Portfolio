import torch
import random
from tqdm import tqdm

torch.set_printoptions(precision=6, sci_mode=False, linewidth=120)

NUM_OPTIONS = 6
WORLD_SIZE = 5
TEMPERATURE = 0.05
ITERATIONS = 50
ROW_STOCHASTIC_REWARDS = True
transition_matrix = torch.nn.functional.softmax(torch.rand(NUM_OPTIONS,WORLD_SIZE,WORLD_SIZE)/TEMPERATURE, dim=-1)
print(transition_matrix)
reward_matrix = torch.nn.functional.softmax(torch.rand(WORLD_SIZE,WORLD_SIZE)/TEMPERATURE, dim=-1) if ROW_STOCHASTIC_REWARDS else torch.rand(WORLD_SIZE,WORLD_SIZE)
print(reward_matrix)

baseline_rewards =[]
for i in range(50):
     total_reward = 0
     initial_position = 0
     for j in range(50):
             choice = random.choice(range(NUM_OPTIONS))
             new_position = random.choices(list(range(WORLD_SIZE)), weights=transition_matrix[choice,initial_position,:].tolist(), k=1)[0]
             total_reward += reward_matrix[initial_position, new_position]
             initial_position = new_position
     print(total_reward)
     baseline_rewards.append(total_reward.item())

baseline_mean = sum(baseline_rewards)/len(baseline_rewards)
baseline_std = (sum([(b-baseline_mean)**2 for b in baseline_rewards])/len(baseline_rewards))**0.5

class TabularQ(torch.nn.Module):
     def __init__(self):
             super().__init__()
             self.qs = torch.nn.Parameter(torch.rand(WORLD_SIZE,NUM_OPTIONS))
             self.q_targets = torch.rand(WORLD_SIZE,NUM_OPTIONS)
             self.optim = torch.optim.Adam([self.qs])
             self.state_buffer = []
             self.next_state_buffer = []
             self.action_buffer = []
             self.reward_buffer = []
             self.done_buffer = []
     def reset_buffers(self):
             self.state_buffer = self.state_buffer[-1000:]
             self.next_state_buffer = self.next_state_buffer[-1000:]
             self.action_buffer = self.action_buffer[-1000:]
             self.reward_buffer = self.reward_buffer[-1000:]
             self.done_buffer = self.done_buffer[-1000:]
     def add_experience(self, state, next_state, action, reward, done):
             self.state_buffer.append(state)
             self.next_state_buffer.append(next_state)
             self.action_buffer.append(action)
             self.reward_buffer.append(reward)
             self.done_buffer.append(done)
     def take_action(self, state, epsilon=0.05):
             if random.random() < epsilon:
                     return random.choices(range(NUM_OPTIONS), k=1)[0]
             else:
                     with torch.no_grad():
                        return torch.argmax(self.qs[state]).item()
     def update(self):
             states = torch.tensor(self.state_buffer, dtype=torch.int64).detach()
             next_states = torch.tensor(self.next_state_buffer, dtype=torch.int64).detach()
             actions = torch.tensor(self.action_buffer, dtype=torch.int64).detach()
             rewards = torch.tensor(self.reward_buffer, dtype=torch.float32).detach()
             dones = torch.tensor(self.done_buffer, dtype=torch.float32).detach()
             indexes = list(range(len(self.state_buffer)))


             if len(indexes) < 256: return None
             for epoch in range(5):
                     random.shuffle(indexes)
                     losses = []
                     for batch in range(0, len(indexes), 32):
                            batch_states = states[indexes[batch: batch+32]]
                            batch_next_states = next_states[indexes[batch: batch+32]]
                            batch_actions = actions[indexes[batch: batch+32]]
                            batch_rewards = rewards[indexes[batch: batch+32]]
                            batch_dones = dones[indexes[batch: batch+32]]


                             
                            self.optim.zero_grad()
                            with torch.no_grad():
                                    max_next_qs, _ = self.q_targets[batch_next_states].max(dim=1)
                                    targets = batch_rewards + ((1-batch_dones)*0.99*max_next_qs)
                            qs = self.qs[batch_states, batch_actions]
                            loss = ((qs-targets)**2).mean()
                            loss.backward()
                            losses.append(loss.item())
                            self.optim.step()

                     with torch.no_grad():
                        self.q_targets.mul_(0.99).add_(self.qs.detach(), alpha=1 - 0.99)
             
             self.reset_buffers()
             return sum(losses)/len(losses)

bandit = TabularQ()

test_rewards = []
pbar = tqdm(range(ITERATIONS))
for step in pbar:
        rewards =[]
        lr = (1-(step/(ITERATIONS+1)))*3e-4
        for g in bandit.optim.param_groups:
                g['lr'] = lr
        epsilon = max(0.0,(1-((step+1)/ITERATIONS)))*0.25
        for i in range(5):
            total_reward = 0
            initial_position = 0
            step = 0
            while True:
                    choice = bandit.take_action(initial_position, epsilon=epsilon)
                    new_position = random.choices(list(range(WORLD_SIZE)), weights=transition_matrix[choice,initial_position,:].tolist(), k=1)[0]
                    total_reward += reward_matrix[initial_position, new_position]
                    step += 1
                    done = int(not step<50)
                    bandit.add_experience(initial_position, new_position, choice, reward_matrix[initial_position, new_position], done)
                    initial_position = new_position
                    loss = bandit.update()
                    pbar.set_description(f"loss: {loss}")
                    if not step<50: break
            
                
            

            print(total_reward)
            rewards.append(total_reward.item())
        
        print(bandit.qs.detach())

        test_rewards.append(sum(rewards)/len(rewards))

advantages = [test_reward-baseline_mean  for test_reward in test_rewards]
print(advantages)



test_rewards =[]
for i in range(50):
            total_reward = 0
            initial_position = 0
            step = 0
            while True:
                    choice = bandit.take_action(initial_position, epsilon=0)
                    new_position = random.choices(list(range(WORLD_SIZE)), weights=transition_matrix[choice,initial_position,:].tolist(), k=1)[0]
                    total_reward += reward_matrix[initial_position, new_position]
                    step += 1
                    done = int(not step<50)
                    initial_position = new_position
                    if not step<50: break
            test_rewards.append(total_reward.item())

test_mean = sum(test_rewards)/len(test_rewards)
test_std = (sum([(b-test_mean)**2 for b in test_rewards])/len(test_rewards))**0.5


sp = (((len(test_rewards)-1)*test_std**2 + (len(baseline_rewards)-1)*baseline_std**2) / (len(test_rewards)+len(baseline_rewards)-2))**0.5
cohens_d = (test_mean - baseline_mean) / sp

print(f"cohens delta between baseline and test policy: {cohens_d}")