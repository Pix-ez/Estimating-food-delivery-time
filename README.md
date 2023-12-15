
# Estimating Food delivery time using machine learning

Using Machine Learning model to estimate time will take to deliver a food item to customer.

Specifically Sequence model which are good at problem with time data like we have here time series prediction.

I have used this dataset from kaggel - [https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset?select=train.csv],
this is the dataset which contains data like food delivery pickup and drop location, food item type, vehicle used by delivery agent.

I did some EDA on this dataset, made new data column for distance, which we'll need to feed into our machine learning model.

This is the model - 
```class Sequntial(pl.LightningModule):
    def __init__(self):
        super(Sequntial, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size = 3, hidden_size= 128,
                             batch_first = True, num_layers = 1)
        
        self.lstm2 = nn.LSTM(input_size = 128, hidden_size= 64,
                             batch_first = True, num_layers = 1)
        self.linear1 = nn.Linear(in_features=64 , out_features=25 )
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features= 25, out_features=1)
        
    def forward(self,x):
       
        output, _ = self.lstm1(x)
        output, _ = self.lstm2(output)
        res = self.linear1(output)
        res = self.relu(res)
        return self.linear2(res)
```

Adam optimizer is used with learning rate- 3e-4, trained for 50 epochs achiving around MSE loss - 60.


