import numpy as np

class DataBuffer():
	def __init__(self, data_init, max_size):
        self.obs = np.concatenate([path['observations'] for path in data_init])
        self.obs_n = np.concatenate([path['next_observations'] for path in data_init])
        self.act = np.concatenate([path['actions'] for path in data_init])
        self.rew = np.concatenate([path['reward'] for path in data_init])

        assert(self.obs.shape == self.obs_n.shape)
        assert(self.obs.shape[0] == self.act.shape[0])

        init_size = self.obs.shape[0]

        self.max_size = max(max_size, init_size)

	def append(self, data_new):
		obs = np.concatenate([path['observations'] for path in data_new])
        obs_n = np.concatenate([path['next_observations'] for path in data_new])
        act = np.concatenate([path['actions'] for path in data_new])
        rew = np.concatenate([path['reward'] for path in data_new])

        assert(obs.shape == obs_n.shape)
        assert(obs.shape[0] == act.shape[0])
        assert(obs.shape[1] == self.obs.shape[1])
        assert(act.shape[1] == self.act.shape[1])

        self.obs = np.concatenate((self.obs, obs), axis = 0)
        self.obs_n = np.concatenate((self.obs_n, obs_n), axis = 0)
        self.act = np.concatenate((self.act, act), axis = 0)
        self.rew = np.concatenate((self.rew, rew), axis = 0)

        if(self.obs.shape[0] > max_size):
            self.obs = self.obs[-max_size:, :]
            self.act = self.act[-max_size:, :]
            self.obs_n = self.obs_n[-max_size:, :]
            self.rew = self.rew[-max_size:, :]
