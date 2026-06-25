import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque
import pygame
import random
import time
class envirmment():
    def __init__(self):
        pygame.init()
        self.width=640
        self.height=480
        self.screen = pygame.display.set_mode((640, 480))
        self.clock = pygame.time.Clock()
    def get_object(self):
        self.clock.tick(30)
        return self.cx/self.width,self.cy/self.height,self.crosshair_x/self.width,self.crosshair_y/self.height
    def reset (self):
        self.fill=self.screen.fill((0,0,0))
        self.crosshair_x=np.random.randint(0,self.width)
        self.crosshair_y=np.random.randint(0,self.height)
        self.cx=np.random.randint(0,self.width)
        self.cy=np.random.randint(0,self.height)
        self.cross=pygame.draw.circle(self.screen, (0, 255, 0), (self.crosshair_x, self.crosshair_y), 10)
        self.target=pygame.draw.rect(self.screen,(255,0,0),(self.cx,self.cy,50,50))
        self.prev_distance=np.sqrt((self.cx/self.width-self.crosshair_x/self.width)**2 + (self.cy/self.height-self.crosshair_y/self.height)**2)
        pygame.display.flip()
        return self.get_object()
    def step(self,action):
        cx,cy,_,_=self.get_object()
        if action==0:
            self.crosshair_y=min(self.height,self.crosshair_y+10)
        if action ==1:
            self.crosshair_y=max(0,self.crosshair_y-10)
        if action==2:
            self.crosshair_x=min(self.width,self.crosshair_x+10)
        if action==3:
            self.crosshair_x=max(0,self.crosshair_x-10)
        distance=np.sqrt((cx-self.crosshair_x/self.width)**2 + (cy-self.crosshair_y/self.height)**2)
        if distance <0.05:
            done=True
            reword=1
        else:
            reword=self.prev_distance-distance
            done=False
        self.prev_distance=distance
        self.screen.fill((0,0,0))
        pygame.draw.rect(self.screen, (255,0,0), (self.cx, self.cy, 50, 50))
        pygame.draw.circle(self.screen, (0,255,0), (self.crosshair_x, self.crosshair_y), 10)
        pygame.display.flip()
        return [cx, cy, self.crosshair_x/self.width, self.crosshair_y/self.height], done, reword

class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(4,36)
        self.hidden=nn.Linear(36,64)
        self.hidden2=nn.Linear(64,64)
        self.output=nn.Linear(64,4)
    def forward(self,x):
        x=nn.functional.relu(self.input(x))
        x=nn.functional.relu(self.hidden(x))
        x=nn.functional.relu(self.hidden2(x))
        x=self.output(x)
        return x

class buffer:
    def __init__(self,capcity):
        self.buffer=deque(maxlen=capcity)
    def addd(self,state,reword,action,next_state,done):
        self.buffer.append((state,reword,action,next_state,done))
    def randomiser(self,batch_size):
        return random.sample(self.buffer,batch_size)
    def __len__(self):
        return len(self.buffer)

epsilon=1
epsilon_decay=0.995
epsilon_min=0.1
gamma=0.9
net=network()
target_net=network()
target_net.load_state_dict(net.state_dict())
capacity=10000
buffer_b=buffer(capacity)
buffer_batch=64
target_s=10
loss=nn.MSELoss()
optimize=torch.optim.Adam(net.parameters(),lr=0.001)
losss = None
env=envirmment()
train=False
if train==True:
    for episode in range(10000):
        done=False 
        steps=0
        current_state = env.reset()
        if current_state is None or current_state[0] is None:
            continue
        while not done and steps<500:
            n=0
            steps+=1
            if epsilon > np.random.rand():
                action=np.random.randint(0,4)
            else:
                with torch.no_grad():
                    action=net(torch.FloatTensor(current_state)).argmax().item()
            next_state,done,reword=env.step(action)
            buffer_b.addd(current_state,reword,action,next_state,done)
            current_state=next_state
            if len(buffer_b) > buffer_batch:
                exp=buffer_b.randomiser(buffer_batch)   
                current_states,rewords,actions,next_states,dones=zip(*exp)
                current_state_b=torch.FloatTensor(current_states)
                reword_b=torch.FloatTensor(rewords)
                action_B=torch.LongTensor(actions)
                next_state_b=torch.FloatTensor(next_states)
                done_b=torch.FloatTensor([float(d) for d in dones])
                with torch.no_grad():
                    target=reword_b +target_net(next_state_b).max(1)[0]*gamma*(1-done_b)
                prediction=net(current_state_b).gather(1,action_B.unsqueeze(1)).squeeze(1)
                optimize.zero_grad()
                losss=loss(target,prediction)
                losss.backward()
                optimize.step()
        epsilon=max(epsilon*epsilon_decay,epsilon_min)
        if episode % 10==0:
            target_net.load_state_dict(net.state_dict())
        if episode % 1==0:
            print(f"episode = {episode} || steps ={steps} || loss = {losss} || action={action}|| epsilon = {epsilon}")
            torch.save(net.state_dict(), 'nett.pth')
            print(f"saved succc")
if train==False:
    net.load_state_dict(torch.load('nett.pth'))
    net.eval()
    state = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            action = net(torch.FloatTensor(state)).argmax().item()
        state, done, reward = env.step(action)
        time.sleep(0.3)
        
        print(f"action={action} | crosshair=({env.crosshair_x},{env.crosshair_y})")