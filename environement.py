import numpy as np
import rl.policy as pol
import tensorflow as tf
import gym
from gym import Env
from gym.spaces import Discrete, Box
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from gym import Wrapper


class ConstrainedActionsWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        if self.env.backtest.position is not None:
            action = 1

        return self.env.step(action)


class DqnTradingEnv(Env):
    def __init__(self, MARGE, backtest):
        self.action_space = Discrete(3, )
        # self.observation_shape = (300, 3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]), dtype=np.float32)

        self.MARGE = MARGE
        self.backtest = backtest

        self.state = self.backtest.Get_information()

        self.balance = self.backtest.balance
        self.win = 0
        self.loose = 0
        self.total_positions = 0

    def step(self, action):
        reward = 0

        if action == 0:
            self.total_positions += 1
            self.backtest.Buy(2, 2)
        elif action == 2:
            self.total_positions += 1
            self.backtest.Sell(2, 2)

        done, status = self.backtest.Step()
        self.state = self.backtest.Get_information()

        if status == 'l':
            self.loose += 1
            reward = -3
        elif status == 'w':
            self.win += 1
            reward = 2

        infos = {"balance": self.balance, "wins": self.win, "looses": self.loose, "positions": self.total_positions}
        return self.state, reward, done, infos

    def render(self):
        pass

    def reset(self):
        self.backtest.Reset()
        self.win = 0
        self.loose = 0
        self.total_positions = 0
        self.state = self.backtest.Get_information()
        return self.state


def build_agent(model, actions):
    policy = pol.BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=300, target_model_update=1e-2)
    return dqn


def build_model(states, actions, lookback):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(1, lookback, 8)))
    model.add(tf.keras.layers.MaxPooling2D((1, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(actions))

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['acc'])

    return model
