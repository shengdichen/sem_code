import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, MultivariateNormal, Categorical
import copy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


def init_layers(layers):
    for l in layers:
        if isinstance(l, torch.nn.Linear):
            torch.nn.init.zeros_(l.bias.data)
            torch.nn.init.xavier_uniform_(l.weight.data)


class MLP(nn.Module):
    def __init__(self, layer_dims, output_dim, nonlin=torch.nn.Tanh(), bias=True):
        super(MLP, self).__init__()
        self.layers = []
        for i in range(1, len(layer_dims) - 1):
            self.layers += [
                torch.nn.Linear(
                    in_features=layer_dims[i - 1], out_features=layer_dims[i], bias=bias
                ),
                nonlin,
            ]

        self.layers += [
            torch.nn.Linear(
                in_features=layer_dims[-1], out_features=output_dim, bias=bias
            ),
            nonlin,
        ]

        init_layers(self.layers)

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.mlp(x)


class CNN(nn.Module):
    def __init__(
        self,
        h,
        w,
        outputs,
        nr_kernels=[3, 16, 32, 32],
        kernel_sizes=[5, 5, 5],
        strides=[2, 2, 2],
        transpose=False,
    ):
        super(CNN, self).__init__()
        self.transpose = transpose
        self.conv1 = nn.Conv2d(
            nr_kernels[0], nr_kernels[1], kernel_size=kernel_sizes[0], stride=strides[0]
        )
        self.bn1 = nn.BatchNorm2d(nr_kernels[1])
        self.conv2 = nn.Conv2d(
            nr_kernels[1], nr_kernels[2], kernel_size=kernel_sizes[1], stride=strides[1]
        )
        self.bn2 = nn.BatchNorm2d(nr_kernels[2])
        self.conv3 = nn.Conv2d(
            nr_kernels[3], nr_kernels[3], kernel_size=kernel_sizes[2], stride=strides[2]
        )
        self.bn3 = nn.BatchNorm2d(nr_kernels[3])

        # CNN(screen_height, screen_width, n_actions)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=kernel_sizes[-1], stride=strides[-1]):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * nr_kernels[3]
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        if self.transpose:
            x.transpose_(1, 3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class MiniGridCNN(nn.Module):
    def __init__(self, layer_dims, use_actions=False):
        super(MiniGridCNN, self).__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )
        if use_actions:
            self.lin = nn.Linear(65, layer_dims[1])
        else:
            self.lin = nn.Linear(64, layer_dims[1])

        self.use_actions = use_actions

    def forward(self, x, a=None):
        x = torch.transpose(x, 1, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        if a is not None and self.use_actions:
            if len(x.shape) != len(a.shape):
                ac = torch.unsqueeze(a, -1)
            x = self.lin(torch.cat([x, torch.unsqueeze(a, -1)], 1))
        else:
            x = self.lin(x)
        return x


class ActorCritic(nn.Module):
    def __init__(
        self,
        layer_dims,
        ob_shape,
        ac_shape,
        nr_actions,
        name,
        trainable,
        base_type="mlp",
        dist="Normal",
        init_std=0.1,
        nonlin=torch.nn.Tanh(),
        bias=True,
    ):
        super(ActorCritic, self).__init__()

        self.name = name
        self.trainable = trainable
        self.log_std = torch.nn.Parameter(
            torch.full([ac_shape[0]], init_std), requires_grad=True
        )

        ## set up base layers
        if base_type == "cnn":
            self.base = CNN(
                ob_shape[0],
                ob_shape[1],
                layer_dims[-1],
                kernel_sizes=[3, 3, 3],
                strides=[1, 1, 1],
            )
            base_out_dim = layer_dims[-1]
        elif base_type == "minigridcnn":
            self.base = MiniGridCNN(layer_dims, use_actions=False)
            base_out_dim = layer_dims[-1]
        elif base_type == "identity":
            self.base = torch.nn.Identity()
            base_out_dim = ob_shape[-1]
        else:
            self.base = MLP(layer_dims, layer_dims[-1], nonlin)
            base_out_dim = layer_dims[-1]

        actor_layer_dims = copy.deepcopy(layer_dims)
        critic_layer_dims = copy.deepcopy(layer_dims)
        actor_layer_dims[0] = base_out_dim
        critic_layer_dims[0] = base_out_dim
        self.base_v = copy.deepcopy(self.base)

        ## set up actor layers
        self.actor_layers = []
        for i in range(1, len(layer_dims)):
            self.actor_layers += [
                torch.nn.Linear(
                    in_features=actor_layer_dims[i - 1],
                    out_features=actor_layer_dims[i],
                    bias=bias,
                ),
                nonlin,
            ]
        # add last layer
        if dist == "Normal":
            self.actor_layers += [
                torch.nn.Linear(
                    in_features=actor_layer_dims[-1],
                    out_features=ac_shape[0],
                    bias=bias,
                )
            ]
            init_layers(self.actor_layers)

            self.actor = nn.Sequential(*self.actor_layers)
            if ac_shape[0] > 1:
                self.a_dist = lambda x: self.dist_fwd(x)
            else:
                self.a_dist = lambda x: Normal(self.actor(x), torch.exp(self.log_std))
        else:
            self.actor_layers += [
                torch.nn.Linear(
                    in_features=actor_layer_dims[-1], out_features=nr_actions, bias=bias
                ),
                torch.nn.LogSoftmax(),
            ]
            init_layers(self.actor_layers)
            self.actor = nn.Sequential(*self.actor_layers)
            self.a_dist = lambda x: Categorical(logits=self.actor(x))

        ## setup critic layers
        self.critic_layers = []
        for i in range(1, len(layer_dims)):
            self.critic_layers += [
                torch.nn.Linear(
                    in_features=critic_layer_dims[i - 1],
                    out_features=critic_layer_dims[i],
                    bias=bias,
                ),
                nonlin,
            ]
        # add last layer
        self.critic_layers += [
            torch.nn.Linear(
                in_features=critic_layer_dims[-1], out_features=1, bias=bias
            )
        ]
        init_layers(self.critic_layers)
        self.critic = nn.Sequential(*self.critic_layers)

        # set up GP for sampling actions

        if not trainable:
            for parameter in self.base.parameters():
                parameter.requires_grad = False
            for parameter in self.base_v.parameters():
                parameter.requires_grad = False
            for parameter in self.actor.parameters():
                parameter.requires_grad = False
            for parameter in self.critic.parameters():
                parameter.requires_grad = False
            self.log_std.requires_grad = False

    def get_actor_parameters(self):
        bl = list(self.base.parameters())
        l = list(self.actor.parameters())
        bl.extend(l)
        l.append(self.log_std)
        return l

    def get_critic_parameters(self):
        bl = list(self.base_v.parameters())
        l = list(self.critic.parameters())
        bl.extend(l)
        return bl

    def sample_policy_action(self, obs):
        h = self.base(obs)
        sample = self.a_dist(h).sample()
        return sample

    def dist_fwd(self, x):
        h = self.actor(x)
        # print(self.log_std)
        # print(h)
        # print([p.data.norm(2) for p in self.actor.parameters()])
        if not np.isfinite(h.detach().numpy()).all():
            print(h)
            print([p.data.norm(2) for p in self.actor.parameters()])
        # diag = torch.Tensor([self.log_std]).expand_as(h)
        dist = MultivariateNormal(h, torch.diag_embed(self.log_std))
        return dist

    def lprobs_action(self, obs, acs):
        h = self.base(obs)
        # return torch.sum(self.a_dist(h).log_prob(acs), -1, keepdim=True)
        return self.a_dist(h).log_prob(acs)

    def get_value(self, obs):
        h = self.base_v(obs)
        value = self.critic(h)
        return value

    def forward(self, obs, acs):
        h = self.base(obs)
        h_v = self.base_v(obs)
        pi = self.a_dist(h)
        log_prob = pi.log_prob(acs)
        value = self.critic(h_v)
        return pi, log_prob, value
