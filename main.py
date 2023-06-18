import numpy as np
import tensorflow as tf
import environement as enviro
from backtest import Backtest, Load_quotes
from environement import build_model, build_agent

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

CANDLE_PRED_MARGE = 1000
MARGE = 1000
LOOKBACK = 168

LEARN_DQN = True

if __name__ == '__main__':
    path = "BTC/Gemini_BTCUSD_1h.csv"

    quotes, quotes_list = Load_quotes(path, MARGE)
    backtest = Backtest(quotes, quotes_list, MARGE, CANDLE_PRED_MARGE, LOOKBACK, LEARN_DQN)

    if LEARN_DQN:
        env = enviro.DqnTradingEnv(MARGE, backtest)

        states = env.observation_space.shape
        actions = env.action_space.n

        model = build_model(states, actions, LOOKBACK)
        dqn = build_agent(model, actions)

        dqn.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])
        dqn.fit(env, nb_steps=20000, visualize=False, verbose=1)

        scores = dqn.test(env, nb_episodes=20, visualize=False)
        print(np.mean(scores.history['episode_reward']))

        dqn.save_weights('weigth/dqn_weights.h5f', overwrite=True)

