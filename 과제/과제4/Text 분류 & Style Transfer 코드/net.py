import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, TEXT,
		         hidden_dim=512, num_layers=1,
                 num_class=4):
        super().__init__()

        vocab_size = TEXT.vocab.vectors.size(0) #TEXT의 사전에 학습된 vector를 가져옴 
        embed_dim = TEXT.vocab.vectors.size(1) #300차원 

        self.embedding = nn.Embedding(vocab_size, embed_dim) #embedding 하기 
        self.encoder = nn.GRU(embed_dim, hidden_dim, 
                              num_layers=num_layers, 
                              dropout=0.5, bidirectional=True) # bidirectional왼쪽에서 오른쪽으로 오른쪽에서 왼쪽으로 하는걸 다 합쳐서 vector 하나로표현 # hidden state에서 다른 hidden state로 넘어갈때 drop out 쓰면 안댐 

        self.embedding.weight.data.copy_(TEXT.vocab.vectors) # text의 vector를 copy
        self.embedding.weight.requires_grad=False # 

        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim*2, num_class) #이번데이터는 class가 5개 # bidirection임으로 *2해서 쓴당
        )
    
    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.encoder(embed)
       
        out = self.linear(out[-1])  #out으로 나온 것 중 가장 오른쪽에 나온것만 fully connectivy 해서 하기로하기위해 -1로 함 
        return out
