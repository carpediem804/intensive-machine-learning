import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,
                 vocab_size, embed_dim=300, # embed_dim은 내가원하는 차원으로 
		         hidden_dim=512, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim) #(input chnanel, output channel)# embedding  = h->1 e->0이런식으로 바꿔진거를 vector형태로 바꿔줌 
        self.encoder = nn.LSTM(embed_dim, hidden_dim, 
                               num_layers=num_layers) #(input channel, output channel, layer num 2이상쓰면성능안좋음) #hidden dim = hidden channel 갯수

        self.decoder = nn.Linear(hidden_dim, vocab_size) #fully connected layer 
    
    def forward(self, x, hidden=None): #(현재state ,이전 state, hidden state = default ==첫 스텝에서는 값이없다 )
        batch_size = x.size(0) # x = [batchsize, sequence , vocavory size ] char rnn에서는 sequence 가 1이다 

        embed = self.embedding(x) #vector로바꾸는거  embed = [n,s,embed_dim]
        out, hidden = self.encoder(embed.view(1, batch_size, -1), hidden) #embed.view(1, batch_size, -1) =  [1,N,embed_dim]으로 바꾼다 hidden = 다음 step에 넣을거 
        out = self.decoder(out.view(out.size(0)*out.size(1), -1)) # 2차원으로 바꾸어서 넣어줌 
        
        return out, hidden
