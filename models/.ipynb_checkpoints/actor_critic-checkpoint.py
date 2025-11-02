# File: models/actor_critic.py (Modified Version)

import torch.nn as nn
from models.mlp import MLPActor
from models.mlp import MLPCritic
import torch.nn.functional as F
from models.graphcnn_congForSJSSP import GraphCNN
import torch


class ActorCritic(nn.Module):
    def __init__(self,
                 n_j, n_m, num_layers, learn_eps, neighbor_pooling_type,
                 input_dim, hidden_dim, num_mlp_layers_feature_extract,
                 num_mlp_layers_actor, hidden_dim_actor,
                 num_mlp_layers_critic, hidden_dim_critic,
                 device,
                 num_dispatching_rules # <-- New parameter
                 ):
        super(ActorCritic, self).__init__()
        self.n_j = n_j
        self.n_m = n_m
        self.device = device

        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        
        # New Actor: Takes global embedding (hidden_dim) and outputs scores for each rule
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim, hidden_dim_actor, num_dispatching_rules).to(device)
        
        # Critic is unchanged
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

    def forward(self,
                x, graph_pool, padded_nei, adj,
                candidate, # Ignored
                mask,      # Ignored
                ):

        # Feature extraction is unchanged
        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)
        
        # New, simpler actor logic
        rule_scores = self.actor(h_pooled)
        pi = F.softmax(rule_scores, dim=-1)

        # Critic logic is unchanged
        v = self.critic(h_pooled)
        
        return pi, v