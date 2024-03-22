import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import copy

GAMMA = 0.98
LAMBDA = 0.96

def ppo_loss(policy, obs, act, log_prob, advantage, epsilon = 0.1):
    p_theta = Categorical(logits = policy(obs)).log_prob(act)
    rat = torch.exp(p_theta - log_prob)
    n_clip = torch.sum((advantage > 0) * (rat > 1+epsilon)) + \
        torch.sum((advantage < 0) * (rat < 1-epsilon))

    dS = torch.mean(0.5 * (p_theta - log_prob) ** 2)
    
    return (
        -torch.sum(
            torch.minimum(advantage * rat, 
                advantage * torch.clip(rat, 1-epsilon, 1+epsilon))),
        n_clip,
        dS
    )

def policy_entropy(policy, obs, act):
    p_theta = Categorical(logits = policy(obs)).log_prob(act)
    return -torch.dot(p_theta, torch.exp(p_theta))

def episode_train(env, policy, t_lim):
    obs, info = env.reset()
    obs_all = []
    act_all = []
    log_prob_all = []
    reward_all = []

    n_steps = 0
    while n_steps < t_lim:
        logits = policy.forward(torch.Tensor(obs))
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob_all.append(dist.log_prob(action).detach().numpy().item())
        obs_all.append(obs)
        act_all.append(action.item())
        
        obs, reward, terminated, truncated, info = env.step(action.item())
        reward_all.append(reward)
        n_steps +=1
        
        if terminated or truncated:
            break
            
    return obs_all, act_all, log_prob_all, reward_all

def epoch_train(
    env, 
    policy, 
    V, 
    policy_optimizer, 
    V_optimizer, 
    V_criterion = torch.nn.MSELoss(), 
    n_samples = int(1e4),
    inner_batch_size = int(1e3),
    gamma = GAMMA,
    lambd = LAMBDA
):
    def _shift(x, fill = 0.0, dir = 1):
        rolled = np.roll(x.copy(), dir)
        if dir == 1:
            rolled[0] = fill
        elif dir == -1:
            rolled[-1] = fill
        else:
            raise KeyError
        return rolled

    def _geo_sum(vector, factor):
        out = vector.copy()
        for i in range(len(vector)-1):
            out[-i-2] += factor * out[i-1]
        return out

    def _zify(vector, epsilon = 1e-4):
        return (vector - np.mean(vector)) / (np.std(vector) + epsilon)

    obs_all = []
    act_all = []
    reward_all = []
    log_prob_all = []
    
    n_episodes = 0 # serves as (reciprocal) reward for CartPole
    while len(obs_all) < n_samples:
        t_lim = n_samples - len(obs_all)
        obs, act, log_prob, reward = episode_train(env, policy, t_lim)
        
        # basic data of training episode
        obs_all.extend(obs)
        act_all.extend(act)
        log_prob_all.extend(log_prob)
        reward_all.extend(reward)
        n_episodes += 1
    
    V_all = V(torch.Tensor(obs_all))[:,0]
    V_labels = _geo_sum(reward_all, factor = gamma) # for subsequent training
    V_labels = torch.tensor(V_labels, dtype=torch.float32)
    V_quality = 1 - torch.var(V_all - V_labels) / torch.var(V_labels)
    
    # Advantage estimate 
    delta_0 = reward_all + gamma * _shift(V_all.detach().numpy(), dir = -1) - V_all.detach().numpy()
    advantage_all = _geo_sum(delta_0, factor = gamma * lambd)
    
    obs_all = torch.tensor(obs_all)
    act_all = torch.tensor(act_all)
    log_prob_all = torch.tensor(log_prob_all)

    adv_mean = np.mean(advantage_all)
    adv_sd = np.std(advantage_all)
    advantage_all = _zify(advantage_all)
    advantage_all = torch.tensor(advantage_all)
    
    
    # train policy
    perm = torch.randperm(n_samples)
    n_inner_batches = int(np.ceil(n_samples/inner_batch_size))

    n_clip_total = 0
    av_dS = 0
    av_S = 0
    
    for i in range(n_inner_batches):
        inds = perm[i*inner_batch_size:(i+1)*inner_batch_size-1]
        policy_loss, n_clip, dS = ppo_loss(
            policy,
            obs_all[inds],
            act_all[inds],
            log_prob_all[inds],
            advantage_all[inds]
        )
        n_clip_total += n_clip
        av_dS += dS/n_inner_batches
        av_S += policy_entropy(policy, obs_all[inds], act_all[inds]) / n_inner_batches
        
        policy_loss.backward()
        policy_optimizer.step()
    
    ## train V
    V_optimizer.zero_grad()
    V_loss = V_criterion(V_all, V_labels)
    V_loss.backward()
    V_optimizer.step()

    return ( 
        np.mean(reward_all), 
        1e2 * n_clip_total.item()/n_samples, 
        1e4 * av_dS.item(),
        av_S.item(), 
        1e4 * adv_mean.item(),
        adv_sd.item(),
        1e2 * V_quality.item()
    )
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = nn.ModuleList()

        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Combine all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
        
def plot_ppo_output(output_all):
    output = np.asarray(output_all)
    fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize = [30,20])
    
    axs[0,0].plot(output[:,0])
    axs[0,0].set_title("reward")
    
    axs[0,1].plot(output[:,3])
    axs[0,1].set_title("policy entropy")
    
    axs[1,0].plot(output[:,1])
    axs[1,0].set_title("n clip")
    
    axs[1,1].plot(output[:,2])
    axs[1,1].set_title("delta S")
    
    axs[2,0].plot(output[:,(4,5)])
    axs[2,0].set_title("V mean & sd")
    
    axs[2,1].plot(output[:,6])
    axs[2,1].set_title("V quality")