import torch
import torch.nn as nn
import pandas_datareader as webreader
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yahooFinance
import datetime
import time
import argparse
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import math
from sklearn.metrics import mean_squared_error
import numpy as np
# Set the API 
def make_parse():
    parser=argparse.ArgumentParser("Stock analysis")
    parser.add_argument('-m',"--mode", type=str, default='realtime',
                        choices=['realtime','csv'],
                        help="Regular grid size",)

    parser.add_argument(
        "--csv_path",
        default='/home/src/yolo/trimvideo.mp4',
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default='./checkpoints/cotracker2.pth',
        help="CoTracker model parameters",
    )
    parser.add_argument(
        "--save",
        default=False,
        help="Save frame and track points",
    )    
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")

    return parser
def split_data(stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to('cuda')
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to('cuda')
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
def main(args):
    device='cuda'
    print('args.mode',args.mode)
    data_source = "yahoo"

    # Set the API parameters
    date_today = "2020-01-01" # period start date
    date_start = "2010-01-01" # period end date
    company = "META" # asset symbol - For more symbols check yahoo.finance.com


    # Here We are getting Facebook financial information
    # We need to pass FB as argument for that
    GetInformation = yahooFinance.Ticker(company)


    
    # in order to specify start date and 
    # end date we need datetime package

    
    # startDate , as per our convenience we can modify
    startDate = datetime.datetime(2019, 5, 31)
    
    # endDate , as per our convenience we can modify
    endDate = datetime.datetime(2021, 1, 30)
    #data=GetInformation.history(start=startDate,end=endDate)
    data=GetInformation.history(period="2y")
    print("data",data)
    data.to_csv("price_quotes.csv", index=True)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.plot(data.index, data.Close)
    plt.show()
    price = data[['Close']]
    price.info()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
    #data.to_csv("price_quotes.csv", index=False)

    lookback = 20 # choose sequence length
    x_train, y_train, x_test, y_test = split_data(price, lookback)
    print('x_train.shape = ',x_train.shape)
    print('y_train.shape = ',y_train.shape)
    print('x_test.shape = ',x_test.shape)
    print('y_test.shape = ',y_test.shape)
    x_train = torch.from_numpy(x_train).type(torch.Tensor).to(device)
    x_test = torch.from_numpy(x_test).type(torch.Tensor).to(device)
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor).to(device)
    y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor).to(device)
    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor).to(device)
    y_test_gru = torch.from_numpy(y_test).type(torch.Tensor).to(device)

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 100
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []

    for t in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))
    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.cpu().detach().numpy()))
    original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.cpu().detach().numpy()))
    sns.set_style("darkgrid")    

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 2, 1)
    ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
    ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (LSTM)", color='tomato')
    ax.set_title('Stock price', size = 14, fontweight='bold')
    ax.set_xlabel("Days", size = 14)
    ax.set_ylabel("Cost (USD)", size = 14)
    ax.set_xticklabels('', size=10)


    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=hist, color='royalblue')
    ax.set_xlabel("Epoch", size = 14)
    ax.set_ylabel("Loss", size = 14)
    ax.set_title("Training Loss", size = 14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)

    # make predictions
    y_test_pred = model(x_test)
    print("x test",x_test.shape)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.cpu().detach().numpy())
    y_train = scaler.inverse_transform(y_train_lstm.cpu().detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.cpu().detach().numpy())
    y_test = scaler.inverse_transform(y_test_lstm.cpu().detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    lstm.append(trainScore)
    lstm.append(testScore)
    lstm.append(training_time)


    # Visualising the results
    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()

    axes.plot(data[len(data)-len(y_test):].index, y_test, color = 'red', label = 'Real Stock Price')
    axes.plot(data[len(data)-len(y_test):].index, y_test_pred, color = 'blue', label = 'Predicted Stock Price')
    #axes.xticks(np.arange(0,394,50))
    plt.title('IBM Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('IBM Stock Price')
    plt.legend()
    plt.savefig('ibm_pred.png')
    plt.show()
if __name__ == "__main__":
    args=make_parse().parse_args()
    #start_time = time.time()
    main(args)
    #end_time = time.time()
    #print("Total time",end_time-start_time)