import numpy as np
from cost_functions import trajectory_cost_fn
import time
import pdb

import tf_util
import load_policy

class Controller():
	def __init__(self):
		pass

	def get_action(self, state):
		pass

class RandomController(Controller):
	def __init__(self, env):
		self.env = env

	def get_action(self, state):
		return self.env.action_space.sample()

class ExpertController(Controller):
	def __init__(self, env, policy_file, eps):
		self.env = env
		self.policy_fn = load_policy.load_policy(policy_file)
		self.eps = eps

	def get_action(self, state):
		if np.random.random() < self.eps:
			return self.env.action_space.sample()
		else:
			return self.policy_fn(state[None,:])

class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self,
				 env,
				 dyn_model,
				 horizon=5,
				 cost_fn=None,
				 num_simulated_paths=10,
				 ):

		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

		self.action_len = env.action_space.shape[0]
		self.state_len = env.observation_space.shape[0]
		self.action_low = env.action_space.low
		self.action_high = env.action_space.high

		self.actions_all = np.random.uniform(self.action_low, self.action_high, \
        	[self.horizon, self.num_simulated_paths, self.action_len])

		self.states_all = np.ones( [self.horizon, self.num_simulated_paths, self.state_len])*np.nan
		self.states_next_all = np.ones( [self.horizon, self.num_simulated_paths, self.state_len])*np.nan

	def get_action(self, state):

		self.actions_all = np.random.uniform(self.action_low, self.action_high, \
			[self.horizon, self.num_simulated_paths, self.action_len])

		self.states_all[0,:,:] = state # set initial state for each path to the arg state

		for i in range(0, self.horizon-1):
			self.states_all[i+1, :, :] = self.dyn_model.predict(self.states_all[i, :, :], self.actions_all[i,:,:])


		self.states_next_all[0:-1,:,:] = self.states_all[1:,:,:]
		self.states_next_all[-1, :, :] = self.dyn_model.predict(self.states_all[-1, :, :], self.actions_all[-1,:,:])

		'''
		# stupid check HERE
		states_all_check = np.ones( [self.horizon, self.num_simulated_paths, self.state_len])*np.nan
		states_next_all_check = np.ones( [self.horizon, self.num_simulated_paths, self.state_len])*np.nan

		states_all_check[0,:,:] = state

		for t in range(self.num_simulated_paths):
			for i in range(0, self.horizon-1):
				s = states_all_check[i, t, :].reshape([1,states_all_check.shape[2]])
				a = self.actions_all[i,t,:].reshape([1, self.actions_all.shape[2]])
				states_all_check[i+1, t, :] = self.dyn_model.predict(s,a)
				states_next_all_check[i,t,:] = states_all_check[i+1, t, :]

			s = states_all_check[-1, t, :].reshape([1,states_all_check.shape[2]])
			a = self.actions_all[-1,t,:].reshape([1, self.actions_all.shape[2]])
			states_next_all_check[-1, t, :] = self.dyn_model.predict(s,a)

		pdb.set_trace()
		'''

		costs_arr = trajectory_cost_fn(self.cost_fn, self.states_all, self.actions_all, \
        	self.states_next_all)

		opt_path_ind = np.argmin(costs_arr)
		act_opt = self.actions_all[0,opt_path_ind,:]
		pdb.set_trace()
		return act_opt

        """ Note: be careful to batch your simulations through the model for speed """
