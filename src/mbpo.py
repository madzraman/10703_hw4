'''
    Main class for MBPO/TD3. Contains the training routine for both MBPO and TD3,
    as well as model rollout, evaluation, and graphing functions.
    You will implement part of this file.
'''
# pylint: disable=W0201, C0103,
import os
import numpy as np
import tensorflow as tf
import pybullet_envs
import gym
import matplotlib.pyplot as plt
import copy
import math

from src.utils import ReplayBuffer
from src.td3 import TD3
from src.pe_model import PE
from src.fake_env import FakeEnv


class MBPO:
    '''
        The main class for both TD3 and MBPO. Some of the attributes are only
        used for MBPO and not for TD3. But notice that the vast majority
        of code is shared.
    '''
    def __init__(self, train_kwargs, model_kwargs, TD3_kwargs):
        # shared training parameters
        self.enable_MBPO = train_kwargs["enable_MBPO"]
        self.policy_name = train_kwargs["policy"]
        self.env_name = train_kwargs["env_name"]
        self.seed = train_kwargs["seed"] #random-seed
        self.load_model = train_kwargs["load_model"]
        self.max_timesteps = train_kwargs["max_timesteps"] #maximum real-env timestemps
        self.start_timesteps = train_kwargs["start_timesteps"] #burn-in period
        self.batch_size = train_kwargs["batch_size"]
        self.eval_freq = train_kwargs["eval_freq"] #Model evaluation frequency
        self.save_model = train_kwargs["save_model"]
        self.expl_noise = train_kwargs["expl_noise"] #TD3 exploration noise

        # MBPO parameters. Pseudocode refers to MBPO pseudocode in writeup.
        self.model_rollout_batch_size = train_kwargs["model_rollout_batch_size"]
        self.num_rollouts_per_step = train_kwargs["num_rollouts_per_step"] #M in pseudocode
        self.rollout_horizon = train_kwargs["rollout_horizon"] #k in pseudocode
        self.model_update_freq = train_kwargs["model_update_freq"] #E in pseudocode
        self.num_gradient_updates = train_kwargs["num_gradient_updates"] #G in pseudocode
        self.percentage_real_transition = train_kwargs["percentage_real_transition"]

        # TD3 agent parameters
        self.discount = TD3_kwargs["discount"] #discount factor
        self.tau = TD3_kwargs["tau"] #target network update rate
        self.policy_noise = TD3_kwargs["policy_noise"] #sigma in Target Policy Smoothing
        self.noise_clip = TD3_kwargs["noise_clip"] #c in Target Policy Smoothing
        self.policy_freq = TD3_kwargs["policy_freq"] #d in TD3 pseudocode

        self.explore = "param" # where we'll change 
        self.iid_sigma = 0.3
        
        self.corr_sigma = 0.2
        self.theta = 0.15 # corr
        self.delta_t = 0.01 # corr

        self.param_sigma = 0.1

        # Dynamics model parameters
        self.num_networks = model_kwargs["num_networks"] #number of networks in ensemble
        self.num_elites = model_kwargs["num_elites"] #number of elites used to predict
        self.model_lr = model_kwargs["model_lr"] #learning rate for dynamics model

        # Since dynamics model remains unchanged every epoch
        # We can perform the following optimization:
        # instead of sampling M rollouts every step for E steps, sample B * M rollouts per
        # epoch, where each epoch is just E environment steps.
        self.rollout_batch_size = self.model_rollout_batch_size * self.num_rollouts_per_step
        # Number of steps in FakeEnv
        self.fake_env_steps = 0

    def eval_policy(self, eval_episodes=10):
        '''
            Runs policy for eval_episodes and returns average reward.
            A fixed seed is used for the eval environment.
            Do not modify.
        '''
        env_name = self.env_name
        seed = self.seed
        policy = self.policy

        eval_env = gym.make(env_name)
        eval_env.seed(seed + 100)

        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = policy.select_action(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward

    def init_models_and_buffer(self):
        '''
            Initialize the PE dynamics model, the TD3 policy, and the two replay buffers.
            The PE dynamics model and the replay_buffer_Model will not be used if MBPO is disabled.
            Do not modify.
        '''
        self.file_name = f"{self.policy_name}_{self.env_name}_{self.seed}"
        print("---------------------------------------")
        print(f"Policy: {self.policy_name}, Env: {self.env_name}, Seed: {self.seed}")
        print("---------------------------------------")

        if not os.path.exists("./results"):
            os.makedirs("./results")

        if self.save_model and not os.path.exists("./models"):
            os.makedirs("./models")

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        env = gym.make(self.env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.max_action = max_action

        td3_kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": self.discount,
            "tau": self.tau,
        }

        # Target policy smoothing is scaled wrt the action scale
        td3_kwargs["policy_noise"] = self.policy_noise * max_action
        td3_kwargs["noise_clip"] = self.noise_clip * max_action
        td3_kwargs["policy_freq"] = self.policy_freq

        model_kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "num_networks": self.num_networks,
            "num_elites": self.num_elites,
            "learning_rate": self.model_lr,
        }

        self.policy = TD3(**td3_kwargs) #TD3 policy
        self.model = PE(**model_kwargs) #Dynamics model
        self.fake_env = FakeEnv(self.model) #FakeEnv to help model unrolling

        if self.load_model != "":
            policy_file = self.file_name if self.load_model == "default" else self.load_model
            self.policy.load(f"./models/{policy_file}")

        self.replay_buffer_Env = ReplayBuffer(state_dim, action_dim)
        self.replay_buffer_Model = ReplayBuffer(state_dim, action_dim)


    def get_action_policy(self, state):
        '''
            Adds exploration noise to an action returned by the TD3 actor.
        '''
        if self.explore == "iid":
            action = (self.policy.select_action(np.array(state)) + np.random.normal(0, self.iid_sigma, size=self.action_dim)).clip(-self.max_action, self.max_action)
        
        if self.explore == "corr":  
            
            z = np.random.normal(0, 1, size = self.action_dim)
            ou_noise = self.prev_noise + self.theta * self.prev_noise * self.delta_t + self.corr_sigma * math.sqrt(self.delta_t) * z # neg in  second term cancels (mu = 0) 
            action = (self.policy.select_action(np.array(state)) + ou_noise).clip(-self.max_action, self.max_action)
            self.prev_noise = ou_noise
        
        if self.explore == "param":   
            state = tf.convert_to_tensor(np.array(state).reshape(1, -1))
            state = tf.cast(state, dtype = "float32")
            action = (self.policy.actor_perturb(state).numpy().flatten()).clip(-self.max_action, self.max_action)
        
        return action

    def get_action_policy_batch(self, state):
        '''
            Adds exploration noise to a batch of actions returned by the TD3 actor.
        '''
        assert len(state.shape) == 2 and state.shape[1] == self.state_dim
        action = (
            self.policy.select_action_batch(np.array(state))
            + np.random.normal(0, self.max_action * self.expl_noise,
                               size=(state.shape[0], self.action_dim))
        ).clip(-self.max_action, self.max_action)
        # Numpy array!
        return action.astype(np.float32)



    # TODO: # 2
    def model_rollout(self):
        '''
            This function performs the model-rollout in batch mode for MBPO.
            This rollout is performed once per epoch, and we sample B * M rollouts.
            First, sample B * M transitions from the real environment replay buffer.
            We get B * M states from these transitions.
            Next, predict the action with exploration noise at these states using the TD3 actor.
            Then, use the step() function in FakeEnv to get the next state, reward and done signal.
            Add the new transitions from model to the model replay buffer.
            Continue until you rollout k steps for each of your B * M starting states, or you
            reached episode end for all starting states.
        '''
        rollout_batch_size = self.rollout_batch_size # B times M
        print('[ Model Rollout ] Starting  Rollout length: {} | Batch size: {}'.format(
            self.rollout_horizon, rollout_batch_size
        ))
        unit_batch_size = self.model_rollout_batch_size # B?

        batch_pass = self.num_rollouts_per_step # M 
        print('Batch pass:', batch_pass)

        # populate this variable with total number of model transitions collected
        total_steps = 0
        for j in range(batch_pass): # M 
            # print(j)
            if j == batch_pass - 1: # deals with remainders
                if rollout_batch_size % unit_batch_size != 0:
                    unit_batch_size = rollout_batch_size % unit_batch_size
            # For M loops, use batch size B
            states, _ , _ , _ , _ = self.replay_buffer_Env.sample(batch_size=unit_batch_size) # B 
            done = np.zeros((2,), dtype = bool) # reset to handle loop gaurd
            k = 0 ## ^^
            while (k < self.rollout_horizon) and (not np.all(done)):
                # print("Total steps:", total_steps)
                noise = tf.clip_by_value(tf.random.normal((unit_batch_size, self.action_dim)) * self.policy_noise, -self.noise_clip, self.noise_clip)
                next_actions = tf.clip_by_value(self.policy.actor.call(states) + noise, -self.max_action, self.max_action)
                # print("clipped")
                next_states, rewards, done = self.fake_env.step(states, next_actions)
                # print("stepped")
                # hint: make use of self.fake_env. Checkout documentation for FakeEnv.py
                # print("sizer before", self.replay_buffer_Model.size)
                # print("REWARDS:", rewards)
                self.replay_buffer_Model.add_batch(states, next_actions, next_states, rewards, done)
                # print("Size after:", self.replay_buffer_Model.size)
                states = next_states # maybe don't need to copy? or tf.identity?
                total_steps = total_steps + unit_batch_size # each loop we transition model batch size times
                k = k + 1 # 
                # print("Done", done)
                # print("reduced done", tf.math.reduce_all(done))
            # raise NotImplementedError

        print('[ Model Rollout ] Added: {:.1e} | Model pool: {:.1e} (max {:.1e})'.format(
            total_steps, self.replay_buffer_Model.size, self.replay_buffer_Model.max_size
        ))

        self.fake_env_steps += total_steps


    # TODO: # 1
    # TODO: # 2 modify
    def prepare_mixed_batch(self):
        '''
            TODO: implement the mixed batch for MBPO
            Prepare a mixed batch of state, action, next_state, reward and not_done for TD3.
            This function should output 5 tf tensors:
            state, shape (self.batch_size, state_dim)
            action, shape (self.batch_size, action_dim)
            next_state, shape (self.batch_size, state_dim)
            reward, shape (self.batch_size, 1)
            not_done, shape (self.batch_size, 1)
            If MBPO is enabled, each of the 5 tensors should a mixture of samples from the
            real environment replay buffer and model replay buffer. Percentage of samples
            from real environment should match self.percentage_real_transition
            If MBPO is disabled, then simply sample a batch from real environment replay buffer.
        '''
        if self.enable_MBPO:
            real_number = int(self.percentage_real_transition * self.batch_size)
            model_number = int(self.batch_size - real_number)

            # Real
            state_e, action_e, next_state_e, reward_e, not_done_e = self.replay_buffer_Env.sample(real_number)

            # model 
            state_m, action_m, next_state_m, reward_m, not_done_m = self.replay_buffer_Model.sample(model_number)
      
            # concat
            state = tf.concat([state_e, state_m], 0)
            action = tf.concat([action_e, action_m], 0)
            next_state = tf.concat([next_state_e, next_state_m], 0)
            reward = tf.concat([reward_e, reward_m], 0)
            not_done = tf.concat([not_done_e, not_done_m], 0)         

        else: # not MBPO 
            state, action, next_state, reward, not_done = self.replay_buffer_Env.sample(self.batch_size)
        return  state, action, next_state, reward, not_done

    def plot_training_curves(self, evaluations, evaluate_episodes, evaluate_timesteps):
        '''
            Plotting script. You should include these plots in the writeup.
            Do not modify.
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot(evaluate_episodes, evaluations)
        ax1.set_xlabel("Training Episodes")
        ax1.set_ylabel("Evaluation Reward")
        ax1.set_title("Reward vs Training Episodes")
        ax2.plot(evaluate_timesteps, evaluations)
        ax2.set_xlabel("Training Timesteps")
        ax2.set_ylabel("Evaluation Reward")
        ax2.set_title("Reward vs Training Timesteps")
        if self.enable_MBPO:
            algo_str = "MBPO"
        else:
            algo_str = "TD3"
        fig.suptitle("Training Curves for " + algo_str, fontsize=20)
        fig.savefig("./results/training_curve_{}.png".format(algo_str))



    # TODO: # 1
    # TODO: # 2 modify
    def train(self):
        '''
            Main training loop for both TD3 and MBPO. See Figure 2 in writeup.
        '''
        E = 1000
        avg_training_rewards = np.zeros(E)
        for seed in range(3): # 3 seeds
            print(seed)
            self.seed += seed
            self.init_models_and_buffer()
            env = gym.make(self.env_name)
            # Set seeds
            env.seed(self.seed)

            # Evaluate untrained policy (HW 4)
            # evaluations = [self.eval_policy()]
            # evaluate_timesteps = [0]
            # evaluate_episodes = [0]

            state, done = env.reset(), False

            # You may want to set episode_reward appropriately
            episode_reward = 0
            episode_timesteps = 0
            episode_num = 0

            
            training_rewards = np.zeros(E)
            t = 0
            
            if self.explore == "corr":
                self.prev_noise = np.random.normal(self.action_dim) 

            if self.explore == "param": # ALSO COPY PASTED IN IF DONE *****
                self.policy.actor_perturb = copy.deepcopy(self.policy.actor)
                old_weights = np.array(self.policy.actor.trainable_weights) # why is this empty at the beginning?
                # print("OLD WEIGHTS\n", old_weights)
                new_weights = old_weights + self.param_sigma * np.random.normal(old_weights.shape)
                self.policy.actor_perturb.set_weights(new_weights) 
            while (episode_num < E):
            # for t in range(int(self.max_timesteps)):
                
                episode_timesteps += 1
                
                # Select action randomly or according to policy
                if t < self.start_timesteps:
                    action = env.action_space.sample()
                else:
                    action = self.get_action_policy(state)
                    # Perform model rollout and model training at appropriate timesteps 
                    if ((t-self.start_timesteps) % self.model_update_freq == 0) and self.enable_MBPO: # want to run at beginning 
                            self.model.train(self.replay_buffer_Env)
                            self.model_rollout()
    

                # Perform action 
                new_state, reward, done, _ = env.step(action)
                episode_reward+= reward
                # Store data in replay buffer
                self.replay_buffer_Env.add(state, action, new_state, reward, done)
                state = new_state


                # Train agent after collecting sufficient data
                if t >= self.start_timesteps:
                    if self.enable_MBPO:
                        # Perform multiple gradient steps per environment step for MBPO
                        for _ in range(self.num_gradient_updates):
                            # print(self.replay_buffer_Model.size)
                            state_t, action_t, next_state_t, reward_t, not_done_t  = self.prepare_mixed_batch()
                            self.policy.train_on_batch(state_t, action_t, next_state_t, reward_t, not_done_t)
                    else: # TD3
                        state_t, action_t, next_state_t, reward_t, not_done_t  = self.prepare_mixed_batch()
                        self.policy.train_on_batch(state_t, action_t, next_state_t, reward_t, not_done_t)


                
                

                if done:
                    print("REWARD:", reward)
                    # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                    print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                    
                    training_rewards[episode_num] = episode_reward
                    
                    # Reset environment
                    if self.explore == "corr":
                        self.prev_noise = np.random.normal(self.action_dim) 
                    if self.explore == "param":
                        self.policy.actor_perturb = copy.deepcopy(self.policy.actor)
                        old_weights = np.array(self.policy.actor.trainable_weights) # why is this empty at the beginning?
                        # print("OLD WEIGHTS 2\n", old_weights)
                        new_weights = old_weights + self.param_sigma * np.random.normal(old_weights.shape)
                        self.policy.actor_perturb.set_weights(new_weights) 

                    state, done = env.reset(), False
                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1

                t += 1
                # Evaluate episode (from HW 4)
                # if (t + 1) % self.eval_freq == 0:
                #     evaluations.append(self.eval_policy())
                #     evaluate_episodes.append(episode_num+1)
                #     evaluate_timesteps.append(t+1)
                #     if len(evaluations) > 5 and np.mean(evaluations[-5:]) > 990:
                #         self.plot_training_curves(evaluations, evaluate_episodes, evaluate_timesteps)
                #     np.save(f"./results/{self.file_name}", evaluations)
                #     if self.save_model:
                #         self.policy.save(f"./models/{self.file_name}")
            avg_training_rewards += training_rewards
        print("now plot")
        print("avg training rewards", avg_training_rewards)
        avg_training_rewards /= 3
        
        plt.figure(figsize=(12, 8))
        plt.plot([i for i in range(len(avg_training_rewards))], avg_training_rewards)
        plt.xlabel("Episode Num")
        plt.ylabel("Average training reward")
        plt.title("MountainCarContinuous-v0, Exploration type: %s" % self.policy.explore)
        plt.savefig(("MountainCarContinuous-v0_%s" % self.policy.explore) + ".png", dpi = 300)

            