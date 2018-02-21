
import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions              
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 10000. # timesteps to observe before training
EXPLORE = 200000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.01  # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
FRAME_PER_ACTION = 1



def buildDQN():

    #input
    s = tf.placeholder("float",[None, 80, 80, 4])

    #hidden layer1
    W_conv1 = weight_variable([8,8,4,32])
    b_conv1 = bias_variable([32])
    hidden_layer1 = tf.nn.relu(conv2d(s,W_conv1,4) + b_conv1)
    h_pool1 = max_pool_2x2(hidden_layer1)


    #hidden layer2
    W_conv2 = weight_variable([4,4,32,64])
    b_conv2 = bias_variable([64])
    hidden_layer2 = tf.nn.relu(conv2d(h_pool1,W_conv2,2) + b_conv2)

    #hidden layer3
    W_conv3 = weight_variable([3,3,64,64])
    b_conv3 = bias_variable([64])
    hidden_layer3 = tf.nn.relu(conv2d(hidden_layer2,W_conv3,1)+b_conv3)

    conv3_flat = tf.reshape(hidden_layer3,[-1,1600])
    W_fc1 = weight_variable([1600,512])
    b_fc1 = bias_variable([512])
    fc1 = tf.nn.relu(tf.matmul(conv3_flat,W_fc1)+b_fc1)

    #DQN
    W_fc2 = weight_variable([512,ACTIONS])
    b_fc2 = bias_variable([ACTIONS])
    dqn = tf.matmul(fc1, W_fc2) + b_fc2

    return s, dqn


def train(s,dqn_out,sess):
    #define the cost function
    a = tf.placeholder("float",[None,ACTIONS])
    y = tf.placeholder("float",[None])
    dqn_action = tf.reduce_sum(tf.multiply(dqn_out,a),reduction_indices=1)
    loss = tf.reduce_mean(tf.square(y-dqn_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

    #game emulator
    game_state = game.GameState()

    #replay memory
    D = deque()

    action0 = np.array([1,0])
    x_t, reward0,terminal = game_state.frame_step(action0)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)),cv2.COLOR_BGR2GRAY)
    ret,x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading network
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_network")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess,checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start train
    epsilon = INITIAL_EPSILON
    t = 0
    while True:
        # choose action with epsilon greedy
        dqn_t = dqn_out.eval(feed_dict={s:[s_t]})[0]
        action_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                action_t[action_index] = 1
            else:
                action_index = np.argmax(dqn_t)
                action_t[action_index] = 1
        else:
            action_t[0] = 1

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        # run the selected action and observe next state and reward
        x_t1_colored, reward_t, terminal = game_state.frame_step(action_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t,action_t,reward_t,s_t1,terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done obseving
        if t > OBSERVE:
            #sample a minibatch to train on
            batch = random.sample(D,BATCH_SIZE)
            # get batch variable
            s_j_batch = [d[0] for d in batch]
            a_batch = [d[1] for d in batch]
            r_batch = [d[2] for d in batch]
            s_j1_batch = [d[3] for d in batch]

            y_batch = []

            dqn_j1_batch = dqn_out.eval(feed_dict={s:s_j1_batch})
            for i in range(0,len(batch)):
                terminal = batch[i][4]
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA*np.max(dqn_j1_batch[i]))
            # sgd
            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch
            })

        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_network/' + GAME + '-dqn', global_step=t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP=", t, " STATE=", state, \
              " EPSILON=", epsilon, " ACTION=", action_index, " REWARD=", reward_t, \
              " Q_MAX=%e" % np.max(dqn_t))



def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W,stride):
    return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')




def playGame():
    sess = tf.InteractiveSession()
    s,dqn = buildDQN()
    train(s,dqn,sess)


def main():
    playGame()



if __name__ == '__main__':
    main()