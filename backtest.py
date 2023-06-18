from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
from stock_indicators.indicators.common.quote import Quote
from stock_indicators import indicators, CandlePart
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    Red = "\033[31m"


def Load_quotes(path, row):
    df = pd.read_csv(path, nrows=row * 2)
    quotes = [Quote(datetime.strptime(date, "%Y-%m-%d %H:%M:%S"), int(open), int(high), int(low), int(close), 0)
              for date, open, high, low, close, vol in
              zip(df['date'], df['open'], df['high'], df['low'], df['close'], df['Volume BTC'])]

    quotes_list = [[date, open, high, low, close] for date, open, high, low, close, vol in
                   zip(df['date'], df['open'], df['high'], df['low'], df['close'], df['Volume BTC'])]

    quotes.reverse()
    quotes_list.reverse()
    return quotes, quotes_list


class Position:
    def __init__(self, price, tp, sl, type):
        self.price = float(price)
        self.tp = float(tp)
        self.sl = float(sl)
        self.type = type


class Backtest:
    def __init__(self, quotes, quotes_list, marge, candlePred_marge, lookback, DQN):

        self.quotes = quotes
        self.quotes_list = np.array(quotes_list)
        self.candlePred_marge = candlePred_marge

        self.index = marge
        self.last_index = marge

        self.marge = marge
        self.lookback = lookback
        self.position = None
        self.DQN = DQN

        self.balance = 100
        self.max_balance = self.balance

        self.done = False

        self.model = tf.keras.models.load_model("./Weights/v5.h5")

    def Step(self):
        status = 'n'
        if self.position is not None:
            if self.position.type == 'b':
                slPrice = self.position.price * (1 - self.position.sl / 100)
                tpPrice = self.position.price * (1 + self.position.tp / 100)

                if self.quotes[self.index].low <= slPrice:
                    self.Loose(self.position)
                    status = 'l'

                elif self.quotes[self.index].high >= tpPrice:
                    self.Win(self.position)
                    status = 'w'
            else:
                slPrice = self.position.price * (1 + self.position.sl / 100)
                tpPrice = self.position.price * (1 - self.position.tp / 100)

                if self.quotes[self.index].high >= slPrice:
                    self.Loose(self.position)
                    status = 'l'
                elif self.quotes[self.index].high <= tpPrice:
                    self.Win(self.position)
                    status = 'w'

        if self.index - self.last_index > 720 or self.balance < self.max_balance * 0.9:
            self.done = True
            self.last_index = self.index

        if self.index >= len(self.quotes) - 1:
            self.Add_candles()

        if self.done is False:
            if self.balance > self.max_balance:
                self.max_balance = self.balance
            self.index += 1

        return self.done, status

    def Add_candles(self):
        scaler = MinMaxScaler()
        q = self.quotes_list[:, 2:]
        q = np.array(q, dtype=np.float32)
        x = q[len(q) - self.candlePred_marge:]

        scaler.fit(x)
        max_ = float(max(max(row) for row in x))
        min_ = float(min(min(row) for row in x))

        trans_x = scaler.transform(x)

        n = np.array(trans_x)

        n = np.expand_dims(n, axis=0)
        n = np.expand_dims(n, axis=3)
        pred = self.model.predict(n)

        new_quote = pred[0] * (max_ - min_) + min_

        new_date = self.quotes[-1].date
        new_date = new_date + timedelta(hours=1)

        new_quote = list(np.around(new_quote, 2))

        new_quote.insert(0, float(self.quotes[-1].close))
        new_quote.insert(0, str(new_date))

        if new_quote[1] > new_quote[4]:
            if new_quote[2] < new_quote[1]:
                v = (new_quote[1] - new_quote[2])
                new_quote[2] = new_quote[1] + v
            if new_quote[3] > new_quote[4]:
                v = (new_quote[3] - new_quote[4])
                new_quote[3] = new_quote[4] - v
        else:
            if new_quote[2] < new_quote[4]:
                v = (new_quote[4] - new_quote[2])
                new_quote[2] = new_quote[4] + v
            if new_quote[3] > new_quote[1]:
                v = (new_quote[3] - new_quote[1])
                new_quote[3] = new_quote[1] - v

        self.quotes_list = np.vstack((self.quotes_list, new_quote))
        QUOTE = Quote(date=new_date, open=int(new_quote[1]), high=int(new_quote[2]),
                      low=int(new_quote[3]), close=int(new_quote[4]))
        self.quotes.append(QUOTE)

    def Get_information(self):  # je minmax le tout, peut etre faire par indicateur ?

        nb = self.index - self.marge
        ema50 = indicators.get_ema(self.quotes[nb:self.index], 50, candle_part=CandlePart.CLOSE)
        ema200 = indicators.get_ema(self.quotes[nb:self.index], 200, candle_part=CandlePart.CLOSE)
        rsi = indicators.get_rsi(self.quotes[nb:self.index], 14)
        macd = indicators.get_macd(self.quotes[nb:self.index], candle_part=CandlePart.CLOSE)

        ema50 = [float(x.ema) for x in ema50[50:]]
        ema200 = [float(x.ema) for x in ema200[200:]]
        rsi = [float(x.rsi) for x in rsi[14:]]
        macd_val = [float(x.macd) for x in macd[200:]]
        signal = [float(x.signal) for x in macd[200:]]
        histogram = [float(x.histogram) for x in macd[200:]]
        fast_ema = [float(x.fast_ema) for x in macd[200:]]
        slow_ema = [float(x.slow_ema) for x in macd[200:]]

        scaler = MinMaxScaler((0, 100))

        x = [rsi[-self.lookback:], ema50[-self.lookback:], ema200[-self.lookback:], macd_val[-self.lookback:],
             signal[-self.lookback:], histogram[-self.lookback:], fast_ema[-self.lookback:], slow_ema[-self.lookback:]]

        x = np.array(x)
        x = x.reshape(-1, 1)

        x = scaler.fit_transform(x)

        x = np.transpose(x)

        x = np.reshape(x, (self.lookback, 8))

        # x = np.reshape(x, (800, 3))

        return x

    def Reset(self):
        self.balance = 100
        self.done = False
        self.position = None

    def Buy(self, sl, tp):
        if self.position is None:
            self.position = Position(self.quotes[self.index].close, sl, tp, 'b')

    def Sell(self, sl, tp):
        if self.position is None:
            self.position = Position(self.quotes[self.index].close, sl, tp, 's')

    def Loose(self, position):
        self.balance *= (1 - position.tp / 100)
        print(f"\n{bcolors.Red}{self.position.type} - {self.quotes[self.index].date} - {self.balance}${bcolors.ENDC}\n")
        self.position = None

    def Win(self, position):
        self.balance *= (1 + position.tp / 100)
        print(
            bcolors.OKGREEN + f"\n{bcolors.OKGREEN}{self.position.type} - {self.quotes[self.index].date} - {self.balance}${bcolors.ENDC}\n")
        self.position = None
