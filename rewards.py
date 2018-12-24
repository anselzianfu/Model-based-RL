import tensorflow as tf
import numpy as np
import pdb

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNRewardModel():
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):

        input_size = 2 * env.observation_space.shape[0] + env.action_space.shape[0] #mlp input dim
        output_size = 1                            #mlp output dim

        self.batch_size = batch_size       # batch size for training
        self.iterations = iterations       # iterations for training
        self.sess = sess                # tf.Session to run the model

        self.mlp_s_input = tf.placeholder(tf.float64, [None, env.observation_space.shape[0]])   # UNnormalized (s) input
        self.mlp_a_input = tf.placeholder(tf.float64, [None, env.action_space.shape[0]])        # UNnormalized (a) input
        self.mlp_sn_input =  tf.placeholder(tf.float64, [None, env.observation_space.shape[0]]) # UNnormalized (s') actual output
        self.mlp_r_target = tf.placeholder(tf.float64, [None, output_size])                            # UNnormalized (s') actual output

        mlp_s_input_norm = self.mlp_s_input
        mlp_a_input_norm = self.mlp_a_input
        mlp_sn_input_norm = self.mlp_sn_input

        mlp_input_norm = tf.concat([mlp_s_input_norm, mlp_a_input_norm, mlp_sn_input_norm], axis=1) #[None, input_size]

        # model to compute deltas
        self.r_model_output = build_mlp(mlp_input_norm, output_size, scope="rewards_model", \
            n_layers=n_layers, size=size, activation=activation, output_activation=output_activation)

        self.model_loss = tf.reduce_mean( tf.nn.l2_loss(self.r_model_output - self.mlp_r_target) )
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.model_loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        obs = np.concatenate([path['observations'] for path in data])         # s_t, N by 20
        obs_n = np.concatenate([path['next_observations'] for path in data])  # s_t+1, N by 20
        act = np.concatenate([path['actions'] for path in data])              # a_t, N by 61
        rew = np.concatenate([path['reward'] for path in data])         # r_t, N by 20

        inds = np.arange(obs.shape[0]) # i.e. (0,..., N-1) for doing mini-batch SGD

        for i in range(self.iterations):
            np.random.shuffle(inds)

            for j in range(0, len(inds), self.batch_size):
                # Get next minibatch
                inds_mb = inds[j:j+self.batch_size]
                obs_mb = obs[inds_mb,:]
                obs_n_mb = obs_n[inds_mb,:]
                act_mb = act[inds_mb,:]
                rew_mb = rew[inds_mb, :]

                # Run a train step
                self.sess.run(self.train_step, \
                    feed_dict={self.mlp_s_input: obs_mb, self.mlp_a_input: act_mb, self.mlp_sn_input: obs_n_mb, self.mlp_r_target: rew_mb})

        return self.sess.run(self.model_loss, feed_dict={self.mlp_s_input: obs, self.mlp_a_input: act, self.mlp_sn_input: obs_n, self.mlp_r_target: rew})
        #print('Ending Loss %f' % \
        #   self.sess.run(self.model_loss, feed_dict={self.mlp_s_input: obs, self.mlp_a_input: act, self.mlp_sn_target: obs_n}))

    def fit(self, obs, obs_n, act, rew):
        inds = np.arange(obs.shape[0]) # i.e. (0,..., N-1) for doing mini-batch SGD

        for i in range(self.iterations):
            np.random.shuffle(inds)

            for j in range(0, len(inds), self.batch_size):
                # Get next minibatch
                inds_mb = inds[j:j+self.batch_size]
                obs_mb = obs[inds_mb,:]
                obs_n_mb = obs_n[inds_mb,:]
                act_mb = act[inds_mb,:]
                rew_mb = rew[inds_mb, :]

                # Run a train step
                self.sess.run(self.train_step, \
                    feed_dict={self.mlp_s_input: obs_mb, self.mlp_a_input: act_mb, self.mlp_sn_input: obs_n_mb, self.mlp_r_target: rew_mb})

        return self.sess.run(self.model_loss, feed_dict={self.mlp_s_input: obs, self.mlp_a_input: act, self.mlp_sn_input: obs_n, self.mlp_r_target: rew})


    def predict(self, states, actions, states_n):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        return self.sess.run(self.model_output, feed_dict={self.mlp_s_input: states, self.mlp_a_input: actions, self.mlp_sn_input: states_n})

    def get_loss(self, data):
        obs = np.concatenate([path['observations'] for path in data])         # s_t, N by 20
        obs_n = np.concatenate([path['next_observations'] for path in data])  # s_t+1, N by 20
        act = np.concatenate([path['actions'] for path in data])              # a_t, N by 6
        rew = np.concatenate([path['reward'] for path in data])         # r_t, N by 20


        return self.sess.run(self.model_loss, feed_dict={self.mlp_s_input: obs, self.mlp_a_input: act, self.mlp_sn_input: obs_n, self.mlp_r_target: rew})

    def get_loss(self, obs, obs_n, act, rew):
        return self.sess.run(self.model_loss, feed_dict={self.mlp_s_input: obs, self.mlp_a_input: act, self.mlp_sn_input: obs_n, self.mlp_r_target: rew})
