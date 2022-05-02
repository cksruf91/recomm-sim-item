import torch


class Item2Vector(torch.nn.Module):
    
    def __init__(self, nitem: int, emb_dim: int, device=torch.device('cpu')):
        super().__init__()
        self.embedding_1 = torch.nn.Embedding(nitem, emb_dim)
        # self.embedding_2 = torch.nn.Embedding(nitem, emb_dim)
        self.fc_1 = torch.nn.Linear(emb_dim*2, emb_dim)
        self.activation = torch.nn.ReLU()
        self.fc_2 = torch.nn.Linear(emb_dim, 1)
        
    def forward(self, center, context):
        emb_1 = self.embedding_1(center)
        emb_2 = self.embedding_1(context)
        
        x = torch.concat([emb_1, emb_2], dim=-1)
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.fc_2(x)
        
        return x.squeeze()
    
    def get_vector(self):
        for name, param in self.embedding_1.named_parameters():
            with torch.no_grad():
                param = param.cpu().numpy()
        return param
        
        