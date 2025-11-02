# File: PPO_jssp_multiInstances.py (FINAL CORRECTED VERSION)

from mb_agg import *
from agent_utils import eval_actions, select_action
from models.actor_critic import ActorCritic
from copy import deepcopy
import torch
import time
import torch.nn as nn
import numpy as np
from Params import configs
from validation import validate

# Import the dispatching rules enum
from dispatching_rules import Rules

device = torch.device(configs.device)


class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        # These are now dummy placeholders for compatibility with the update function
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = [] # This will store the RULE index
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]


class PPO:
    def __init__(self,
                 lr, gamma, k_epochs, eps_clip, n_j, n_m,
                 num_layers, neighbor_pooling_type, input_dim, hidden_dim,
                 num_mlp_layers_feature_extract, num_mlp_layers_actor,
                 hidden_dim_actor, num_mlp_layers_critic, hidden_dim_critic,
                 num_dispatching_rules): # <-- Added parameter
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(n_j=n_j, n_m=n_m, num_layers=num_layers,
                                  learn_eps=False, neighbor_pooling_type=neighbor_pooling_type,
                                  input_dim=input_dim, hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic, device=device,
                                  num_dispatching_rules=num_dispatching_rules) # <-- Passed to model
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)
        self.V_loss_2 = nn.MSELoss()

    def update(self, memories, n_tasks, g_pool):
        # This method is generic and does NOT need to be changed.
        # It works because the modified ActorCritic forward pass ignores the
        # candidate/mask arguments that are passed in here.
        
        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef

        rewards_all_env = []
        adj_mb_t_all_env, fea_mb_t_all_env, candidate_mb_t_all_env, mask_mb_t_all_env = [], [], [], []
        a_mb_t_all_env, old_logprobs_mb_t_all_env = [], []
        
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
                if is_terminal: discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_all_env.append(rewards)
            
            adj_mb_t_all_env.append(aggr_obs(torch.stack(memories[i].adj_mb).to(device), n_tasks))
            fea_mb_t = torch.stack(memories[i].fea_mb).to(device)
            fea_mb_t = fea_mb_t.reshape(-1, fea_mb_t.size(-1))
            fea_mb_t_all_env.append(fea_mb_t)
            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(device).squeeze().detach())

        mb_g_pool = g_pool_cal(g_pool, torch.stack(memories[0].adj_mb).to(device).shape, n_tasks, device)

        for _ in range(self.k_epochs):
            loss_sum, vloss_sum = 0, 0
            for i in range(len(memories)):
                pis, vals = self.policy(x=fea_mb_t_all_env[i],
                                        graph_pool=mb_g_pool,
                                        adj=adj_mb_t_all_env[i],
                                        candidate=candidate_mb_t_all_env[i],
                                        mask=mask_mb_t_all_env[i],
                                        padded_nei=None)
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                advantages = rewards_all_env[i] - vals.view(-1).detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.squeeze(), rewards_all_env[i])
                p_loss = -torch.min(surr1, surr2).mean()
                ent_loss = -ent_loss.clone()
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss
                loss_sum += loss
                vloss_sum += v_loss
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item()


def main():
    from JSSP_Env import SJSSP
    envs = [SJSSP(n_j=configs.n_j, n_m=configs.n_m) for _ in range(configs.num_envs)]
    
    from uniform_instance_gen import uni_instance_gen
    data_generator = uni_instance_gen

    dataLoaded = np.load(f'./DataGen/generatedData{configs.n_j}_{configs.n_m}_Seed{configs.np_seed_validation}.npy')
    vali_data = [(dataLoaded[i][0], dataLoaded[i][1]) for i in range(dataLoaded.shape[0])]

    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)

    memories = [Memory() for _ in range(configs.num_envs)]

    num_rules = len(Rules)
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=configs.n_j, n_m=configs.n_m, num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type, input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim, num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor, hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic, hidden_dim_critic=configs.hidden_dim_critic,
              num_dispatching_rules=num_rules)

    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, configs.n_j*configs.n_m, configs.n_j*configs.n_m]),
                             n_nodes=configs.n_j*configs.n_m, device=device)
    
    rule_candidates = np.array([rule.value for rule in Rules])

    log, validation_log = [], []
    record = 100000
    for i_update in range(configs.max_updates):
        ep_rewards = [0 for _ in range(configs.num_envs)]
        adj_envs, fea_envs = [], []
        
        for i, env in enumerate(envs):
            adj, fea, _, _ = env.reset(data_generator(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high))
            adj_envs.append(adj)
            fea_envs.append(fea)
            ep_rewards[i] = -env.initQuality
            
        while True:
            fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(device) for fea in fea_envs]
            adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(device).to_sparse() for adj in adj_envs]
            
            with torch.no_grad():
                action_envs, a_idx_envs = [], []
                for i in range(configs.num_envs):
                    pi, _ = ppo.policy_old(x=fea_tensor_envs[i], graph_pool=g_pool_step,
                                           padded_nei=None, adj=adj_tensor_envs[i],
                                           candidate=None, mask=None)
                    
                    action, a_idx = select_action(pi, rule_candidates, memories[i])
                    action_envs.append(action)
                    a_idx_envs.append(a_idx)
            
            next_adj_envs, next_fea_envs = [], []
            
            for i in range(configs.num_envs):
                memories[i].adj_mb.append(adj_tensor_envs[i])
                memories[i].fea_mb.append(fea_tensor_envs[i])
                memories[i].candidate_mb.append(torch.from_numpy(rule_candidates).to(device))
                memories[i].mask_mb.append(torch.zeros(len(rule_candidates), dtype=torch.bool).to(device))
                memories[i].a_mb.append(a_idx_envs[i])

                adj, fea, reward, done, _, _ = envs[i].step(action_envs[i].item())
                
                next_adj_envs.append(adj)
                next_fea_envs.append(fea)
                ep_rewards[i] += reward
                memories[i].r_mb.append(reward)
                memories[i].done_mb.append(done)

            adj_envs, fea_envs = next_adj_envs, next_fea_envs

            if envs[0].done():
                break
                
        for j in range(configs.num_envs):
            ep_rewards[j] -= envs[j].posRewards

        loss, v_loss = ppo.update(memories, configs.n_j*configs.n_m, configs.graph_pool_type)
        for memory in memories:
            memory.clear_memory()
            
        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
        log.append([i_update, mean_rewards_all_env])
        
        print(f'Episode {i_update + 1}\t Last reward: {mean_rewards_all_env:.2f}\t Mean_Vloss: {v_loss:.8f}')
        
        if (i_update + 1) % 100 == 0:
            # NOTE: You must also modify the `validation.py` file for this to work.
            vali_result = -validate(vali_data, ppo.policy).mean()
            validation_log.append(vali_result)
            if vali_result < record:
                torch.save(ppo.policy.state_dict(), f'./{configs.n_j}_{configs.n_m}_{configs.low}_{configs.high}.pth')
                record = vali_result
            print('The validation quality is:', vali_result)
            
            with open(f'./log_{configs.n_j}_{configs.n_m}_{configs.low}_{configs.high}.txt', 'w') as f:
                f.write(str(log))
            with open(f'./vali_{configs.n_j}_{configs.n_m}_{configs.low}_{configs.high}.txt', 'w') as f:
                f.write(str(validation_log))

if __name__ == '__main__':
    main()