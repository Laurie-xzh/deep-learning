import torch 
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

print(torch.cuda.is_available())

# Create RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        '''
        the shape of input x is (batch_size, seq_length, input_dim)
        '''
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        # show the use of Variable
        # print(h0)


        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) # here we only use the last time step of the sequence
        print(hn.shape)
        return out

# 定义模型参数
input_dim = 10
hidden_dim = 20
layer_dim = 2
output_dim = 5

# 创建一个随机输入张量
batch_size = 3
seq_length = 4
x = torch.randn(batch_size, seq_length, input_dim)

# 创建RNN模型
model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

# 进行前向传播
output = model(x)

# 打印输出张量
print(output.shape)
