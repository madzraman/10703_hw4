'''
    Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
    Paper: https://arxiv.org/abs/1802.09477
    Adopted from author's PyTorch Implementation
'''
# pylint: disable=C0103, R0913, R0901, W0221, R0902, R0914
import copy
import tensorflow as tf
from tensorflow import keras
import numpy as np
import math

class Actor(keras.Model):
    '''
        The actor in TD3. Architecture from authors of TD3
    '''
    def __init__(self, action_dim, max_action):
        super().__init__()

        self.l1 = keras.layers.Dense(256, activation="relu")
        self.l2 = keras.layers.Dense(256, activation="relu")
        self.l3 = keras.layers.Dense(action_dim)
        self.max_action = max_action


    def call(self, state):
        '''
            Returns the tanh normalized action
            Ensures that output <= self.max_action
        '''
        a = self.l1(state)
        a = self.l2(a)
        return self.max_action * keras.activations.tanh(self.l3(a))


class Critic(keras.Model):
    '''
        The critics in TD3. Architecture from authors of TD3
        We organize both critics within the same keras.Model
    '''
    def __init__(self):
        super().__init__()

        # Q1 architecture
        self.l1 = keras.layers.Dense(256, activation="relu")
        self.l2 = keras.layers.Dense(256, activation="relu")
        self.l3 = keras.layers.Dense(1)

        # Q2 architecture
        self.l4 = keras.layers.Dense(256, activation="relu")
        self.l5 = keras.layers.Dense(256, activation="relu")
        self.l6 = keras.layers.Dense(1)


    def call(self, state, action):
        '''
            Returns the output for both critics. Using during critic training.
        '''
        sa = tf.concat([state, action], 1)

        q1 = self.l1(sa)
        q1 = self.l2(q1)
        q1 = self.l3(q1)

        q2 = self.l4(sa)
        q2 = self.l5(q2)
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        '''
            Returns the output for only critic 1. Used to compute actor loss.
        '''
        sa = tf.concat([state, action], 1)

        q1 = self.l1(sa)
        q1 = self.l2(q1)
        q1 = self.l3(q1)
        return q1


class TD3():
    '''
        The TD3 main class. Wraps around both the actor and critic, and provides
        three public methods:
        train_on_batch, which trains both the actor and critic on a batch of
        transitions
        select_action, which outputs the action by actor given a single state
        select_action_batch, which outputs the actions by actor given a batch
        of states.
    '''
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.state_dim = state_dim
        self.actor = Actor(action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=3e-4)

        self.critic = Critic()
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=3e-4)

        self.explore = "iid" # where we'll change 

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip 
        self.policy_freq = policy_freq
        self.sigma = 0.2 # correlated
        self.theta = 0.15 # corr
        self.delta_t = 0.01 # corr

        self.total_it = 0


    def select_action(self, state):
        '''
            Select action for a single state.
            state: np.array, size (state_dim, )
            output: np.array, size (action_dim, )
        '''
        state = tf.convert_to_tensor(state.reshape(1, -1))
        return self.actor(state).numpy().flatten()

    def select_action_batch(self, state):
        '''
            Select action for a batch of states.
            state: np.array, size (batch_size, state_dim)
            output: np.array, size (batch_size, action_dim)
        '''
        if not tf.is_tensor(state):
            state = tf.convert_to_tensor(state)
        return self.actor(state).numpy()


    # TODO: # 1
    def train_on_batch(self, state, action, next_state, reward, not_done): 
        '''
            Trains both the actor and the critics on a batch of transitions.
            state: tf tensor, size (batch_size, state_dim)
            action: tf tensor, size (batch_size, action_dim)
            next_state: tf tensor, size (batch_size, state_dim)
            reward: tf tensor, size (batch_size, 1)
            not_done: tf tensor, size (batch_size, 1)
            You need to implement part of this function.
        '''
        self.total_it += 1

        # Select action according to policy and add clipped noise
        prev_noise = tf.random.normal(action.shape) # change? 
        if self.explore == "clipped_iid":
            noise = tf.clip_by_value(tf.random.normal(action.shape) * self.policy_noise,
                                    -self.noise_clip, self.noise_clip)

            next_action = tf.clip_by_value(self.actor_target(next_state) + noise,
                                        -self.max_action, self.max_action)
        if self.explore == "iid":
            noise = tf.random.normal(action.shape) * self.policy_noise

            next_action = tf.clip_by_value(self.actor_target(next_state) + noise,
                                        -self.max_action, self.max_action)
        if self.explore == "corr":  
            z = tf.random.normal(action.shape)
            ou_noise = prev_noise + self.theta * prev_noise * self.delta_t + self.sigma * math.sqrt(self.delta_t) * z # neg in  second term cancels (mu = 0) 
            next_action = tf.clip_by_value(self.actor_target(next_state) + ou_noise,
                                        -self.max_action, self.max_action)
            prev_noise = ou_noise



        
        # Compute the target Q value
        # print(not_done)
        critic_Q1, critic_Q2 = self.critic_target.call(next_state, next_action) # stop gradient? 
        min_Q = tf.math.minimum(critic_Q1, critic_Q2) # element wise minimum 
        # Y = tf.math.add(reward*not_done, tf.math.scalar_mul(self.discount, min_Q))
        Y = reward + (not_done * self.discount * min_Q)
      
        # Get current Q estimates
        # Compute critic loss
        with tf.GradientTape() as tape:
            current_Q1, current_Q2 = self.critic.call(state, action) # inside to track network for gradient 
            critic_loss_1 = tf.math.reduce_mean(tf.math.square(Y - current_Q1)) 
            critic_loss_2 = tf.math.reduce_mean(tf.math.square(Y - current_Q2))
            # critic_loss_1 = tf.keras.losses.MeanSquaredError(Y, current_Q1)
            # critic_loss_2 = tf.keras.losses.MeanSquaredError(Y, current_Q2)
            # print(critic_loss_1, critic_loss_2)
            critic_loss = critic_loss_1 + critic_loss_2
        # print("LOSS:", critic_loss)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        # print(len(critic_grads))
        # print(critic_grads)

        # Optimize the critic
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        if self.total_it % self.policy_freq == 0:
        #       t       mod     d
        # Delayed policy updates
            # Compute actor losses
            with tf.GradientTape() as tape_actor: # everything we perform GD on 
                curr_policy = self.actor.call(state)
                # print(state.dtype)
                # print(curr_policy.dtype)
                cond_current_Q = self.critic.Q1(state, curr_policy)
                actor_loss = -tf.math.reduce_mean(cond_current_Q) # negative since we maximize 
            actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
            # var = copy.deepcopy(self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            # print(np.array(var) - np.array(self.actor.trainable_variables))
            # print(self.actor.trainable_variables[0])
            # Update the frozen target models
            # critic: update both Q1 and Q2 in same loop 
            for (ct,c) in zip(self.critic_target.variables, self.critic.variables):
                ct.assign((self.tau*c) + ((1-self.tau)*ct))

            # actor
            for (at,a) in zip(self.actor_target.variables, self.actor.variables):
                at.assign((self.tau*a) + ((1 - self.tau)*at))

        return 


    def save(self, filename):
        '''
            Saves current weight of actor and critic. You may use this function for debugging.
            Do not modify.
        '''
        self.critic.save_weights(filename + "_critic")
        self.actor.save_weights(filename + "_actor")


    def load(self, filename):
        '''
            Loads current weight of actor and critic. Notice that we initialize the targets to
            be identical to the on-policy weights.
        '''
        self.critic.load_weights(filename + "_critic")
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_weights(filename + "_actor")
        self.actor_target = copy.deepcopy(self.actor)
