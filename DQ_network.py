import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np
import random

import subprocess

import imageio
from ball_graphics import ball_GUI

import os


class DQN(nn.Module):

    def __init__(self,input_feat,output_feat):
        super(DQN,self).__init__()
        self.fc1 = nn.Linear(input_feat,64) # network layers
        self.fc2 = nn.Linear(64,64)
        self.out = nn.Linear(64,output_feat)

    def forward(self,X): # forward steps with actiavation functions
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        return self.out(X)

# HYPER-PARAMETERS

INPUTS = 2
ACTIONS = 3 # no. of OUTPUT features
lr = 0.0001
GAMMA = 0.99 # discount factor / care about the future
EPSILON = 1.0 # fully random
MIN_EPSILON = 0.02 # only 10% randomness
# EPSILON_DECAY = 0.995 # decay rate
EPSILON_DECAY = 0.9995 # lesser decay rate

# ITERATIONS VARIABLES

EPISODES = 4500 # makes possible to keeps running the network for learning, similar to no. of EPOCHS

# GUI RENDER SETTINGS

RENDER_EPI = 200 # each 200 episodes/epoch
RENDER_FPS = 60 # clock ticks

gui = ball_GUI() # initializing the graphics class for the gui of the ball
gui.ball_gui()   # setting the gui of the ball

# Initializing main brain and frozen brain copy

torch.manual_seed(42)

brain_net = DQN(INPUTS,ACTIONS)
frozen_net = DQN(INPUTS,ACTIONS)
frozen_net.load_state_dict(brain_net.state_dict())

optimizer = optim.Adam(brain_net.parameters(),lr=lr)
memory = deque(maxlen=10000) # replay buffer

episodes_for_record = [0,200,400,800,1600,3000,3600,4000,4500]

for episode in range(EPISODES):# the learning loop

    render_episode = (episode % RENDER_EPI == 0) # true or false
    record_episode = (episode in episodes_for_record) # capture episodes outputs true or false check
    
    if record_episode:
        gif_frames=[] # gif frames list for saving
    
    proc = subprocess.Popen(
        ['./object_physics'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )

    def read_env():
        line = proc.stdout.readline()
        h,v,reward,done,terminal = line.split()

        return float(h), float(v), float(reward), bool(int(done)), bool(int(terminal))

    h, v , reward, done, terminal  = read_env()

    steps = 0
    episode_loss = 0
    total_reward = 0

    min_h = float('inf')
    max_h = float('-inf')

    episode_steps = []
    epi_rewards = []

    while not done :
        
        steps+= 1

        min_h = min(min_h,h)
        max_h = max(max_h,h)

        if random.random() < EPSILON:
            action = random.randint(0,ACTIONS-1)
        else:
            states = torch.FloatTensor([h/20,v/10]) # scaled values for h and v. Netowork friendly to avoid biasy to greater value vairable. (v/10 cause 10 is enough for the current target postitons)

            with torch.no_grad(): # stopping learning phase on hold
                net_values = brain_net(states)
                action = torch.argmax(net_values).item() # chooses the highest value in the list of output values

        if render_episode or record_episode: # just so we can capture the very beginning of step-0
            gui.renderer()
            
            gui.setting_balls_height(h=h,
                                     v=v,
                                     action=int(abs(action)),
                                     fps=RENDER_FPS)
            
            frames = np.array(gui.capture_gif())
            
            if record_episode and len(gif_frames)< 300: # limit the no. gif frames
                    gif_frames.append(frames)
            
        proc.stdin.write(f'{action}\n')
        proc.stdin.flush()

        h_new,v_new,reward,done, terminal = read_env()

        total_reward += reward
        # scaled_reward = max(min(reward,1),-1)

        memory.append((h/20,v/10,action,reward,h_new/20,v_new/10,terminal ))

        if len(memory) > 128: # base size to fire the learning process

            batch = random.sample(memory,64)

            # unpacking variable from the batch (boilerplate)

            states = torch.FloatTensor([x[0:2] for x in batch]) # previous height and velocity h,v
            action = torch.LongTensor([x[2]for x in batch])
            rewards = torch.FloatTensor([x[3] for x in batch])
            next_state = torch.FloatTensor([x[4:6] for x in batch])
            terminals = torch.FloatTensor([x[6] for x in batch])

            current_q = brain_net(states).gather(1,action.unsqueeze(1)).squeeze() # (Prediction) for the actual action we took

            # calulating the expected or the actual action to be taken (reality) using the Frozen net

            with torch.no_grad():
                max_next_q = frozen_net(next_state).max(1)[0] # Best possible future reward estimate using newer states
                expected_q = rewards + (GAMMA * max_next_q * (1 - terminals))

            loss = nn.MSELoss()(current_q,expected_q) # loss function for the prediction and actual expected result

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain_net.parameters(),max_norm=1.0) # normalizes the gradients after a maax limit, avoids exploding gradients..

            optimizer.step()

            episode_loss += loss.item()

        h, v = h_new, v_new # for next iteration of the learning network

        episode_steps.append(steps)
        epi_rewards.append(total_reward)

    proc.terminate()

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY) #keeps decreasing the random ness

    # updating the Frozen Net weights occassionally
    if episode % 100 == 0:
        frozen_net.load_state_dict(brain_net.state_dict())
    
    # training logs/infos    
    print(f"Episode : {episode}, Steps : {steps}, Loss : {episode_loss:.2f}, Reward : {total_reward:.2f}, h_range : [{min_h:.2f},{max_h:.2f}] Epsilon : {EPSILON:.2f}")
    
    # saving the gifs
    
    if record_episode:
        
        gif_dir = 'gifs'
        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
        
        gif_path = os.path.join(gif_dir,f"hover_episode{episode}.gif")  # avoids issues with '/' or '\'
        
        imageio.mimsave(gif_path, gif_frames, fps=30) # saves to the specified gif folder
    