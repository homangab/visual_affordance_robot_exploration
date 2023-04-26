import numpy as np
import torch
from torch.nn import *
import torch.nn as _nn
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

USE_GPU = torch.cuda.is_available()
CUDA_DEVICE = torch.device('cuda')
CPU_DEVICE = torch.device('cpu')


def default_device():
    if USE_GPU:
        return CUDA_DEVICE
    else:
        return CPU_DEVICE

# define a new metaclass which overrides the "__call__" function
# https://stackoverflow.com/questions/16017397/injecting-function-call-after-init-with-decorator
class _NewInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj._post_init()
        return obj

@six.add_metaclass(_NewInitCaller)
class Module(_nn.Module):
    def __init__(self):
        super(Module, self).__init__()

    def _post_init(self):
        self.to(device=default_device())
class CrossEntropyLoss(Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean', label_smoothing=0):
        super(CrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate
        self.label_smoothing = label_smoothing

    def forward(self, input, target, weights=None):
        ce = cross_entropy_with_weights(input, target, weights, self.label_smoothing)
        if self.aggregate == 'sum':
            return ce.sum()
        elif self.aggregate == 'mean':
            return ce.mean()
        elif self.aggregate is None:
            return ce
def cross_entropy_with_weights(logits, target, weights=None, label_smoothing=0):
    assert logits.dim() == 2
    assert not target.requires_grad
    target = target.squeeze(1) if target.dim() == 2 else target
    assert target.dim() == 1
    loss = torch.logsumexp(logits, dim=1) - (1-label_smoothing) * class_select(logits, target) - label_smoothing * logits.mean(dim=1)
    if weights is not None:
        # loss.size() = [N]. Assert weights has the same shape
        assert list(loss.size()) == list(weights.size())
        # Weight the loss
        loss = loss * weights
    return loss
def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    one_hot_mask = (torch.arange(0, num_classes)
                               .long()
                               .repeat(batch_size, 1)
                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask.to('cuda:0'))

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if USE_GPU:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class FCNetwork(Module):
    """
    A fully-connected network module
    """
    def __init__(self, dim_input, dim_output, layers=[256, 256],
            nonlinearity=torch.nn.ReLU, dropout=0):
        super(FCNetwork, self).__init__()
        net_layers = []
        dim = dim_input
        for i, layer_size in enumerate(layers):
          net_layers.append(torch.nn.Linear(dim, layer_size))
          net_layers.append(nonlinearity())
          if dropout > 0:
              net_layers.append(torch.nn.Dropout(0.4))
          dim = layer_size
        net_layers.append(torch.nn.Linear(dim, dim_output))
        self.layers = net_layers
        self.network = torch.nn.Sequential(*net_layers)

    def forward(self, states):
        print("states",states.shape)
        return self.network(states)

class Policy(object):
    def __init__(self):
        pass

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        return self.act_vectorized(state)[0]

    def act_vectorized(self, state):
        raise NotImplementedError()
    
    def reset(self):
        pass

class GoalConditionedPolicy(object):
    def __init__(self):
        pass

    def act(self, state, goal_state):
        state = np.expand_dims(state, axis=0)
        goal_state = np.expand_dims(goal_state, axis=0)
        return self.act_vectorized(state, goal_state)[0]

    def act_vectorized(self, state, goal_state):
        raise NotImplementedError()
    
    def reset(self):
        pass

class DiscreteStochasticGoalPolicy(Module, GoalConditionedPolicy):
    def __init__(self,feature_extract=True):
        super(DiscreteStochasticGoalPolicy, self).__init__()
        self.feature_extract = feature_extract        
        if self.feature_extract:
            dim_in = 512 + 512 
        else:
            dim_in = 4*32*32 + 4*32*32        
        self.n_dims = 6
        self.granularity = 384
        dim_out = 7 #self.n_dims * self.granularity
        self.net = FCNetwork(dim_in, dim_out, layers=[512, 512])

    def forward(self, obs,goal):

        if obs.shape[0] < 333: ## if there is extra dim...
            c = torch.cat((obs,goal),dim=1).to('cuda:0')
        else:
            c = torch.cat((obs,goal)).to('cuda:0')
        return self.net.forward(c)

    def act_vectorized(self, obs, goal, noise = 0.1):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)                
        logits = self.forward(obs, goal)
        logits = logits.view(-1, self.n_dims, self.granularity)
        noisy_logits = logits  * (1 - noise)
        probs = torch.softmax(noisy_logits, 2)
        samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        samples = self.flattened(samples)
        return to_numpy(samples)
    
    def nll(self, obs, goal, actions, horizon=None):        
        actions_perdim = self.unflattened(actions)
        # print(actions, self.flattened(actions_perdim))
        actions_perdim = actions_perdim.view(-1)
        logits = self.forward(obs, goal)
        logits_perdim = logits.view(-1, self.granularity) 
        loss_perdim = CrossEntropyLoss(aggregate=None, label_smoothing=0)(logits_perdim, actions_perdim, weights=None)
        loss = loss_perdim.reshape(-1, self.n_dims)
        return loss.sum(1)
    
    def probabilities(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        probs = torch.softmax(logits, 1)
        return probs

    def entropy(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        probs = torch.softmax(logits, 1)
        Z = torch.logsumexp(logits, dim=1)
        return Z - torch.sum(probs * logits, 1)

    def flattened(self, tensor):
        # tensor expected to be n x self.n_dims
        multipliers = self.granularity ** torch.tensor(np.arange(self.n_dims))
        flattened = (tensor * multipliers).sum(1)
        return flattened.int()
    
    def unflattened(self, tensor):
        # tensor expected to be n x 1
        digits = []
        output = tensor
        for _ in range(self.n_dims):
            digits.append(output % self.granularity)
            output = output // self.granularity
        uf = torch.stack(digits, dim=-1)
        return uf

class FeatureExtract(nn.Module):
    def __init__(self):
        super(FeatureExtract, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.image_features = nn.ModuleList(resnet18.children())[:-1]
        self.image_features = nn.Sequential(*self.image_features)

    def forward(self, x, y):
       #shape of x is (b_s, 32,32,1)        
        features = F.relu(self.image_features(x))
        features = features.view(-1, 512)
        goal_features = F.relu(self.image_features(y))
        goal_features = goal_features.view(-1, 512)
        return features, goal_features

