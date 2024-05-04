###############################################################################
#
# https://mp.weixin.qq.com/s/FV4uB_r2ov-jOj3_OzGaqg
#
###############################################################################

import numpy as np
import pandas as pd

import tushare as ts
import backtrader as bt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def get_data():
    def get_stock_data(code, start_date, end_date, token):
        ts.set_token(token)
        pro = ts.pro_api()
        df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
        df = df.sort_values(by="trade_date", ascending=True)
        df.set_index("trade_date", inplace=True)
        return df

    stock_code = "600036.SH"
    start_date = "20200101"
    end_date = "20220101"
    api_token = "4a397fdb369f7993d5b3d00580068425dfef20694af83cf9743d0741"

    data = get_stock_data(stock_code, start_date, end_date, api_token)
    data.to_csv('../../data/stock/600036SH.csv')


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.bn(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


def train_model():

    def create_dataset(stock_data, window_size):

        X = []
        y = []
        scaler = MinMaxScaler()
        stock_data_normalized = scaler.fit_transform(stock_data.values.reshape(-1, 1))

        for i in range(len(stock_data) - window_size - 2):
            X.append(stock_data_normalized[i:i + window_size])
            if stock_data.iloc[i + window_size + 2] > stock_data.iloc[i + window_size - 1]:
                y.append(1)
            else:
                y.append(0)

        X, y = np.array(X), np.array(y)
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()
        return X, y, scaler

    stock_data = pd.read_csv('../../query/stock/600036SH.csv', index_col=0, parse_dates=True)
    stock_data = stock_data['close']

    window_size = 21
    X, y, scaler = create_dataset(stock_data, window_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    input_size = 1
    hidden_size = 10
    num_layers = 3
    num_classes = 2
    model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20

    for epoch in range(num_epochs):
        for i, (batch_X, batch_y) in enumerate(train_loader):
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Finished Training')
    torch.save(model.state_dict(), '../tmp/lstm_model.pth')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        test_data = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    print(f'Accuracy of the model on the test data: {100 * correct / total}%')

    return scaler, window_size, input_size, hidden_size, num_layers, num_classes


def eval_strategy(scaler, window_size, input_size, hidden_size, num_layers, num_classes):

    class LSTMStrategy(bt.Strategy):

        def __init__(self):
            self.data_close = self.datas[0].close
            self.model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)
            self.model.load_state_dict(torch.load('../tmp/lstm_model.pth'))
            self.model.eval()
            self.scaler = scaler
            self.counter = 1

        def next(self):
            if self.counter < window_size:
                self.counter += 1
                return
            previous_close_prices = [self.data_close[-i] for i in range(0, window_size)]
            X = torch.tensor(previous_close_prices).view(1, window_size, -1).float()
            X = self.scaler.transform(X.numpy().reshape(-1, 1)).reshape(1, window_size, -1)

            prediction = self.model(torch.tensor(X).float())

            max_vals, max_idxs = torch.max(prediction, dim=1)
            predicted_prob, predicted_trend = max_vals.item(), max_idxs.item()

            if predicted_trend == 1 and not self.position:  # 上涨趋势
                self.order = self.buy()  # 买入股票
            elif predicted_trend == 0 and self.position:  # 如果预测不是上涨趋势且持有股票，卖出股票
                self.order = self.sell()


    cerebro = bt.Cerebro(runonce=False)

    data = pd.read_csv('../../query/stock/600036SH.csv', index_col=0, parse_dates=True)
    data = bt.feeds.PandasData(dataname=data, datetime=None, open=1, high=2,
                               low=3, close=4, volume=8, openinterest=-1)
    cerebro.adddata(data)

    cerebro.addstrategy(LSTMStrategy)

    cerebro.broker.setcash(10000)  # 本金10000，每次交易100股
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    cerebro.broker.setcommission(commission=0.0005)  # 万五佣金

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.plot()


if __name__ == '__main__':
    scaler, window_size, input_size, hidden_size, num_layers, num_classes = train_model()
    eval_strategy(scaler, window_size, input_size, hidden_size, num_layers, num_classes)