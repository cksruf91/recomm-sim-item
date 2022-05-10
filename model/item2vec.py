import torch


class Item2Vector(torch.nn.Module):
    
    def __init__(self, nitem: int, emb_dim: int, device=torch.device('cpu')):
        super().__init__()
        self.embedding_1 = torch.nn.Embedding(nitem, emb_dim)
        self.embedding_2 = torch.nn.Embedding(nitem, emb_dim)
        self.sigmoid= torch.nn.Sigmoid()
        self.to(device)
        
    def forward(self, center, context):
        emb_1 = self.embedding_1(center)
        emb_2 = self.embedding_2(context)
        x = torch.sum(torch.mul(emb_1, emb_2), dim=-1)
        return self.sigmoid(x)
    
    def get_vector(self):
        for name, param in self.embedding_1.named_parameters():
            with torch.no_grad():
                param = param.cpu().numpy()
        return param
