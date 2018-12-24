import gym
import tensorflow as tf
import numpy as np
import pdb
import time
import dynamics as dyn
import controllers as ctrl
import matplotlib.pyplot as plt
import cheetah_env as cenv
import pickle
from cost_functions import cheetah_cost_fn

def plot_model_fit(env, model, data):
    path = data[0]
    '''obs = np.concatenate([path['observations'] for path in data])         # s_t, N by 20
    obs_n = np.concatenate([path['next_observations'] for path in data])  # s_t+1, N by 20
    act = np.concatenate([path['actions'] for path in data])              # a_t, N by 6
    '''

    obs = path['observations']
    obs_n = path['next_observations']
    act = path['actions']

    test_horizon = obs.shape[0]

    obs_n_predicted = model.predict(obs, act)

    fig = plt.figure()
    ax = fig.gca()

    plt.xlabel('Time step')
    plt.ylabel('State Value')

    errors = (obs_n - obs_n_predicted)**2
    error_vec = np.sum(errors, axis=0)

    print(error_vec)

    error_order = np.argpartition(-error_vec, 5)
    top_five_errors_ind = error_order[:5]

    for i in top_five_errors_ind:
        plt.title('State %d'  % i)
        plt.plot(np.arange(test_horizon), obs_n[:,i], c='k', ls='solid')
        plt.plot(np.arange(test_horizon), obs_n_predicted[:,i], c='r', ls='dashed')
        plt.show()

def make_rollouts(env, controller, num_paths=10, horizon=1000, render=False):
    paths = []

    for i in range(num_paths):
        last_ob = env.reset() # initial state, s
        obs, obs_next, rew, act = [], [], [], [] #(s,s',r,a)

        for j in range(horizon):
            if(render):
                env.render()
                time.sleep(0.01)

            action = controller.get_action(last_ob)
            ob, reward, done, info = env.step(action)

            # check if done
            if(done):
                break

            obs.append(last_ob)  # s
            act.append(np.squeeze(action))   # a
            rew.append(reward)   # R(s,a)
            obs_next.append(ob)  # s'

            last_ob = ob # s <- s'

        current_path = {'observations'      : np.array(obs),
            'reward'            : np.array(rew),
            'actions'           : np.array(act),
            'next_observations' : np.array(obs_next) }

        paths.append(current_path)

    return paths

def main_get_data():
    env = gym.make('Humanoid-v1')
    policy_file = 'experts/Humanoid-v1.pkl'
    eps = 0.2

    random_ctrl = ctrl.RandomController(env)
    policy_ctrl = ctrl.ExpertController(env, policy_file, eps)


    sess = tf.Session()

    sess.__enter__()
    tf.global_variables_initializer().run()

    data_train = make_rollouts(env, policy_ctrl, num_paths=300, horizon=1000, render=False)
    data_test = make_rollouts(env, policy_ctrl, num_paths=200, horizon=1000, render=False)

    train_env_name = "%s_rollouts_%s.pkl" % ('humanoid_v1', 'train')
    test_env_name = "%s_rollouts_%s.pkl" % ('humanoid_v1', 'test')
    pickle.dump(data_train, open(train_env_name, "wb"))
    pickle.dump(data_test, open(test_env_name, "wb"))

def main_fit_model():

    env = gym.make('Humanoid-v1')

    sess = tf.Session()

    dyn_model = dyn.NNDynamicsModel(env=env,
                            n_layers=2,
                            size=500,
                            activation=tf.nn.relu,
                            output_activation=None,
                            normalization=None,
                            batch_size=50,
                            iterations=10,
                            learning_rate=1e-3,
                            sess=sess)

    sess.__enter__()
    tf.global_variables_initializer().run()

    train_env_name = "%s_rollouts_%s.pkl" % ('humanoid_v1', 'train')
    test_env_name = "%s_rollouts_%s.pkl" % ('humanoid_v1', 'test')

    data_train = pickle.load(open(train_env_name, "rb"))
    data_test = pickle.load(open(test_env_name, "rb"))

    obs_train = np.concatenate([path['observations'] for path in data_train])         # s_t, N by 20
    obs_n_train = np.concatenate([path['next_observations'] for path in data_train])  # s_t+1, N by 20
    act_train = np.concatenate([path['actions'] for path in data_train])              # a_t, N by 6

    obs_test = np.concatenate([path['observations'] for path in data_test])         # s_t, N by 20
    obs_n_test = np.concatenate([path['next_observations'] for path in data_test])  # s_t+1, N by 20
    act_test = np.concatenate([path['actions'] for path in data_test])              # a_t, N by 6

    iter_arr = []
    train_loss_arr = []
    test_loss_arr = []

    train_iters = 100
    #init_train_loss = dyn_model.get_loss(data_train)
    #init_test_loss = dyn_model.get_loss(data_test)
    init_train_loss = dyn_model.get_loss(obs_train, obs_n_train, act_train)
    init_test_loss = dyn_model.get_loss(obs_test, obs_n_test, act_test)
    print 'Iter 0, Train Loss %f, Test Loss %f' % (init_train_loss, init_test_loss)
    iter_arr.append(0)
    train_loss_arr.append(init_train_loss)
    test_loss_arr.append(init_test_loss)

    pdb.set_trace()
    for i in range(0, train_iters, 10):
        train_loss = dyn_model.fit(obs_train, obs_n_train, act_train)
        test_loss = dyn_model.get_loss(obs_test, obs_n_test, act_test)
        print 'Iter %i, Train Loss %f, Test Loss %f' % (10+i, train_loss, test_loss)
        iter_arr.append(10 + i)
        train_loss_arr.append(train_loss)
        test_loss_arr.append(test_loss)

    training_results = {'iters': np.array(iter_arr), \
                        'train_loss': np.array(train_loss_arr), \
                        'test_loss': np.array(test_loss_arr)}

    pkl_results_name = "%s_mdl_results.pkl" % ('humanoid_v1')
    pickle.dump(training_results, open(pkl_results_name, "wb"))

def plot_results():
    pkl_results_name = "%s_mdl_results.pkl" % ('humanoid_v1')
    results = pickle.load(open(pkl_results_name, "rb"))
    fig = plt.figure()
    test_loss = results['test_loss']
    train_loss = results['train_loss']
    iters = np.array([x for x in range(0,101,10)])
    plt.plot(iters, train_loss/31289, 'r', label='train',  marker='o', ls = 'solid')
    plt.plot(iters, test_loss/20570,  'b', label='test', marker='o', ls = 'solid')
    plt.xlabel('Iteration')
    plt.ylabel('Error (MSE)')
    plt.yscale('log')
    plt.title('Training Curve for Humanoid-v1')
    plt.legend()
    plt.show()


if(__name__=="__main__"):
    #main_get_data()
    #main_fit_model()
    plot_results()
