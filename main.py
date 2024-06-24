import gymnasium as gym
from gym_example import *
import numpy as np
import pygame

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #print("x",x.shape)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# BATCH_SIZE = 30
# GAMMA = 0.99
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 1000
# TAU = 0.005
# LR = 1e-4

# # Get number of actions from gym action space
# #n_actions = env.action_space.n
# n_actions = 4
# # Get the number of state observations
# #state, info = env.reset()

# n_observations = 2

# policy_net = DQN(n_observations, n_actions).to(device)
# target_net = DQN(n_observations, n_actions).to(device)
# target_net.load_state_dict(policy_net.state_dict())
# for param_tensor in policy_net.state_dict():
#     print(param_tensor, "\t", policy_net.state_dict()[param_tensor].size())
# #torch.save(policy_net.state_dict(), '/mount/policy.pt')
# optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# memory = ReplayMemory(10000)


#steps_done = 0
def select_action(state,env,EPS_END,EPS_START,EPS_DECAY,policy_net,steps_done):
    #global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)




def plot_durations(show_result=False,episode_durations=[]):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
        print("episode_durations",episode_durations) #number step archive
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# BATCH_SIZE = 30
# GAMMA = 0.99
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 1000
# TAU = 0.005
# LR = 1e-4

def optimize_model(BATCH_SIZE,GAMMA,memory,policy_net,target_net,optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

#Maze config
def main():
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))


    BATCH_SIZE = 30
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    episode_durations = []
    # Get number of actions from gym action space
    #n_actions = env.action_space.n
    n_actions = 4
    # Get the number of state observations
    #state, info = env.reset()

    n_observations = 2

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    for param_tensor in policy_net.state_dict():
        print(param_tensor, "\t", policy_net.state_dict()[param_tensor].size())
    #torch.save(policy_net.state_dict(), '/mount/policy.pt')
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)


    steps_done = 0

    maze = [
        ['S', '', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '.'],
        ['#', '.', '#', 'G'],
    ]
    # Test the environment
    env = gym.make('testenv-v0',maze=maze)
    #obs = env.reset()

    #env.render()

    done = False
    if torch.cuda.is_available():
        num_episodes = 150
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
    #while True:
        pygame.event.get()
        state,dummy_infomation = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        #input("Press Enter to continue...")
        env.render()
        for t in count():
            print("ttttttttttt",t)
            #print("state",state)
            #print("state", state.shape)
            action = select_action(state,env,steps_done=steps_done,EPS_END=EPS_END,EPS_START=EPS_START,EPS_DECAY=EPS_DECAY,policy_net=policy_net)
            #action = env.action_space.sample()
            action=torch.tensor(action, dtype=torch.int64, device=device)
            
            #print("action", action)
            observation, reward, done, _ = env.step(action.item())
            
            reward = torch.tensor([reward], device=device)
            if done:
                next_state = None
                #print("Done")
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            #input("Press Enter to continue...")
            env.render()
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model(BATCH_SIZE=BATCH_SIZE,GAMMA=GAMMA,memory=memory,policy_net=policy_net,target_net=target_net,optimizer=optimizer)
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations=episode_durations)
                break



        #action = env.action_space.sample()  # Random action selection
        #obs, reward, done, _ = env.step(action)
        #print("obs222",obs)
        #env.render()
        #print('Reward:', reward)
        #print('Done:', done)
    plot_durations(show_result=True,episode_durations=episode_durations)
    pygame.time.wait(200)
    plt.ioff()
    plt.show()

def main_can_remove():
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))


    BATCH_SIZE = 30
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    episode_durations = []
    # Get number of actions from gym action space
    #n_actions = env.action_space.n
    n_actions = 4
    # Get the number of state observations
    #state, info = env.reset()

    n_observations = 2

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    for param_tensor in policy_net.state_dict():
        print(param_tensor, "\t", policy_net.state_dict()[param_tensor].size())
    #torch.save(policy_net.state_dict(), '/mount/policy.pt')
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)


    steps_done = 0

    maze = [
        ['S', '', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '.'],
        ['#', '.', '#', 'G'],
    ]
    # Test the environment
    env = gym.make('Dynamic_env-v0',maze=maze)
    #obs = env.reset()

    #env.render()

    done = False
    if torch.cuda.is_available():
        num_episodes = 150
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
    #while True:
        pygame.event.get()
        state,dummy_infomation = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        #input("Press Enter to continue...")
        env.render()
        for t in count():
            #print("state",state)
            #print("state", state.shape)
            action = select_action(state,env,steps_done=steps_done,EPS_END=EPS_END,EPS_START=EPS_START,EPS_DECAY=EPS_DECAY,policy_net=policy_net)
            #action = env.action_space.sample()
            action=torch.tensor(action, dtype=torch.int64, device=device)
            
            #print("action", action)
            observation, reward, done, _ = env.step(action.item())
            
            reward = torch.tensor([reward], device=device)
            if done:
                next_state = None
                print("Done")
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            #input("Press Enter to continue...")
            env.render()
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model(BATCH_SIZE=BATCH_SIZE,GAMMA=GAMMA,memory=memory,policy_net=policy_net,target_net=target_net,optimizer=optimizer)
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations=episode_durations)
                break



        #action = env.action_space.sample()  # Random action selection
        #obs, reward, done, _ = env.step(action)
        #print("obs222",obs)
        #env.render()
        #print('Reward:', reward)
        #print('Done:', done)
    plot_durations(show_result=True,episode_durations=episode_durations)
    pygame.time.wait(200)
    plt.ioff()
    plt.show()
def play_env():
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    #env = gym.make('Dynamic_env-v0',maze=maze)

def main_stock():
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))


    BATCH_SIZE = 30
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-3
    episode_durations = []
    # Get number of actions from gym action space
    #n_actions = env.action_space.n
    n_actions = 3
    # Get the number of state observations
    #state, info = env.reset()

    n_observations = 22  #check

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    for param_tensor in policy_net.state_dict():
        print(param_tensor, "\t", policy_net.state_dict()[param_tensor].size())
    #torch.save(policy_net.state_dict(), '/mount/policy.pt')
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)


    steps_done = 0

    maze = [
        ['S', '', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '.'],
        ['#', '.', '#', 'G'],
    ]

    csv_path="/home/src/stock/RL/price_quotes.csv"


    # Test the environment
    #env = gym.make('stockenv-v0',maze=maze)
    env = gym.make('stockenv-v0',csv=csv_path)
    #obs = env.reset()
    #input("Press Enter to continue...")
    #env.render()

    done = False
    if torch.cuda.is_available():
        num_episodes = 500
    else:
        num_episodes = 2000

    for i_episode in range(num_episodes):
    #while True:
        print("start ")
        state,dummy_infomation = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        #input("Press Enter to continue...")
        #env.render()
        for t in count():
            #print("start ")
            #print("state", state.shape)
            action = select_action(state,env,steps_done=steps_done,EPS_END=EPS_END,EPS_START=EPS_START,EPS_DECAY=EPS_DECAY,policy_net=policy_net)
            #action = env.action_space.sample()
            #action=torch.tensor(action, dtype=torch.int64, device=device)
            action=action.clone().detach().to(device)
            #print("action", action)
            observation, reward, done,truncated, _ = env.step(action.item())
            
            reward = torch.tensor([reward], device=device)
            if done or truncated:
                next_state = None
                print("Done")
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            #input("Press Enter to continue...")
            #env.render()
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model(BATCH_SIZE=BATCH_SIZE,GAMMA=GAMMA,memory=memory,policy_net=policy_net,target_net=target_net,optimizer=optimizer)
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            if done or truncated:
                #print("reward",reward)
                #print("reward",type(reward))
                #tensor_value.cpu().item()
                #episode_durations.append(reward.cpu().item())
                episode_durations.append(t + 1)

                plot_durations(episode_durations=episode_durations)
                break



        #action = env.action_space.sample()  # Random action selection
        #obs, reward, done, _ = env.step(action)
        #print("obs222",obs)
        #env.render()
        #print('Reward:', reward)
        #print('Done:', done)
    plot_durations(show_result=True,episode_durations=episode_durations)
    #pygame.time.wait(200)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main_stock()