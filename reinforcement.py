# import required libraries
import math, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim       # for DQN model optimizer  
import torch.autograd as autograd 
import torch.nn.functional as F

import matplotlib.pyplot as plt
from collections import deque, namedtuple
from features import *           # import get descriptor image


# related to Cuda
USE_CUDA = torch.cuda.is_available()
with torch.no_grad():
    Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

# define the constants
observation_space = 25112
action_space = 6
visual_descriptor_size = 25088    # as in common block, Fig 3, 25088+(6*4)=25112
actions_of_history = 4            # actions captured in histor tensor
iou_threshold = 0.5               # consider a positive detection
reward_movement_action = 1        # reward on a good movement action
reward_terminal_action = 3        # reward on a good terminal action

# FOR REPLAY BUFFER 
Transition = namedtuple('Transition', ('State', 'action', 'reward', 'next_state', 'done'))

# create Types of tensors (float, long, and byte tensor)
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor 
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

Tensor = FloatTensor    # initialize tensor as floatTensor

# define the Replay Memory Class
class ReplayBuffer(object):
    # Function to Initialize replay memory
    def __init__(self, capacity):
        """
            input : size of buffer
            output : just initialize buffer
        """
        # initialize list as experience replay memory
        self.capacity = capacity
        self.buffer = []
        self.position = 0          # index of transition
        
    # function to push the transition (st, at, rt, st+1, _) into buffer
    def push(self, *args):
        """
            input : current transition of states
            output : Just append transition into buffer
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
        # check if not None then expand the dimensions
#         if state != None:
#             state = torch.unsqueeze(state, 0)     # insert new dimension at 0
#         if next_state != None:
#             next_state = torch.unsqueeze(next_state, 0)  # insert dimension at 0
        
        # store the current tansition 
#         self.buffer.append((state, action, reward, next_state, done))
        
    # function to get a sample of trnasitions
    def sample(self, batch_size):
        """
            input : batch size of sample requested
            output : sample of size = batch size
        """
        # get a sample of transitions from replay buffer
        # zip group them, and store them in variables
#         state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))     
        
        # return the sample
        # also concatenate multiple np.arrays present in var 
#         return torch.cat(state), action, reward, torch.cat(next_state), done  

        # return a sample of transitions
        return random.sample(self.buffer, batch_size)   # using random.sample()
    
    def __len__(self):
        """
            output : length of the buffer
        """
        return len(self.buffer)
        

# define the DQN class
class DQN(nn.Module):
    # define the constructor 
    def __init__(self, num_inputs, num_actions):
        """
            input : observation space and action space
            output : Just initialize the model
        """
        super(DQN, self).__init__()
        
        # define the multilayer network
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 1024),      # as in Fig 3
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_actions)
        )
    
    # define the forward method
    def forward(self, x):
        """
            input : state of the environment
            ouput : its corresponding q_value
        """
        # return the q_value of state
        return self.layers(x)
    
    # define the function to get action
    def get_action(self, state, epsilon):
        """
            input : state and probability epsilon
            output : preferred action
        """
        # epsilon time out of 1.00 return good
        if random.random() > epsilon and state != None:
            
            # get the state (st)
            # no need of cpu # state = state.cpu()    # load tensor into cpu
            state = Variable(state)   # resize it  from [25112, 1] to [1, 25112, 1] (done without it)
            
            # get the q_value of st
            q_value = self.forward(state)
            
            # get the best action
            # action = q_value.data.max(1)[1].item()     # ERROR - only one element tensor can be convert to python scalar
            _, predicted = torch.max(q_value.data, 1)    # get predicted value from q_value
            action = predicted[0] + 1                    # calculate action
            action = action.item()                       # get the single item
            
        # other time return the random action
        else:
            action = random.randint(1, 6)    # randomly select b/w [1, 7]
            
        # return the choosen action
        return action
        

def get_q_network():
    """
        output : return a DQN network model
    """
    # we will create a DQN model using DQN class
    # AS GIVEN, OBSERVATION SPACE = 25112 AND ACTION SPACE = 6
    model = DQN(observation_space, action_space)   # try with original size
    
    # return the DQN model
    return model


# define function to get current state
def get_state(current_image, history_tensor, model_vgg):
    """
        input : current region image, history tensor for past actions, pretrained vgg16 model
        output : return current state = stack of tensors
    """
    # get the descriptor image for current image
    descriptor_image = get_descriptor_image(current_image, model_vgg)
    
    # AS IN COMMON BLOCK
    # reshape the descriptor image
    descriptor_image = torch.reshape(descriptor_image, (visual_descriptor_size, 1))   # (input, (shape))
    
    # reshape the history tensor, (24, 1)
    history_tensor = torch.reshape(history_tensor, (action_space*actions_of_history, 1))   # no need of .cpu()
    history_tensor = history_tensor.cuda()  # load on any GPU  .to(torch.device('cuda:0'))   # load on cuda
    
    # create a state, using descriptor image and history tensor
    # stack the tensors  (vertically)
    state = torch.vstack([descriptor_image, history_tensor])
    
    # make it correct shape, (1, 25112)
    state = torch.transpose(state, 0, 1)
    
    # return the state created
    return state


# define function to get reward/penalty on terminal action
def terminal_reward(curr_IOU):
    """
        input : IOU of object mask in current region image
        output : reward/penalty at termination action
    """
    # check if current IOU > threshold IOU
    if curr_IOU > iou_threshold:
        reward = reward_terminal_action     # reward 3
    else:
        reward = - reward_terminal_action    # penalty -3
    
    # return terminal action reward
    return reward

# define function to get reward/penalty on current action
def get_reward(IOU, new_IOU):
    """
        input : previou IOU, and new IOU (after action)
        output : return reward or penalty 
    """
    # check if beneficial action
    if new_IOU > IOU:
        reward = reward_movement_action   # give reward 1
    else:
        reward = - reward_movement_action   # give penalty -1
    
    # return the reward or penalty
    return reward

# define function to update history tensor
def update_history(history_tensor, action):
    """
        input : history tensor (history till now), action take
        output : updated history tensor
    """
    # Initialize an action tensor
    action_tensor = torch.zeros(action_space)
    
    # add the action taken into action tensor
    action_tensor[action - 1] = 1
    
    # get size of old history tensor
    HT_nonZero = torch.nonzero(history_tensor)     # get tensor of indices of nonZero elements
    history_tensor_size = HT_nonZero.shape[0] * HT_nonZero.shape[1]   # get product of length of dimensions
    
    # Initialize a new history tensor (for updated action)
    updated_history_tensor = torch.zeros(action_space * actions_of_history)

    
    # check if size of history tensor < 6
    if history_tensor_size < actions_of_history:
        tmp = 0     # initialize a temporary variable
        
        # iterate through history vector
        for i in range(action_space * history_tensor_size, action_space * history_tensor_size + action_space - 1):   # 6*size to 6*size + 3
            
            # copy action from action tensor to history tensor
            history_tensor[i] =  action_tensor[tmp]
            tmp += 1          # move forward
        
        # return the updated history tensor
        return history_tensor
    else:
        
        # iterate through old history tensor
        for i in range(0, action_space * (actions_of_history - 1) - 1):   # 0 to 17
            updated_history_tensor[i] = history_tensor[i + action_space]   # copy value at there
        
        tmp = 0    # initialize tmp variable
        
        for i in range(action_space * (actions_of_history - 1), action_space * actions_of_history):    # 18 to 24
            updated_history_tensor[i] = action_tensor[tmp]    # copy value at there
            tmp += 1    # move forward
        
        # return the updated history tensor
        return updated_history_tensor