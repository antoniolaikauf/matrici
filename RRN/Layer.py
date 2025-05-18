import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, tokensEmbedding, head = 8):
        super(MultiHeadAttention, self).__init__()
        self.head = head
        self.d_model = d_model
        self.tokensEmbedding = tokensEmbedding
        self.dk = d_model // head
        self.dv = d_model // head
        self.w_q = nn.ParameterList([nn.Parameter(torch.randn(self.d_model, self.dk)) for _ in range(head)])
        self.w_k = nn.ParameterList([nn.Parameter(torch.randn(self.d_model, self.dk)) for _ in range(head)])
        self.w_v = nn.ParameterList([nn.Parameter(torch.randn(self.d_model, self.dk)) for _ in range(head)])

    def attention(self, q, k, v):
        pass

    def forward(self):
        output = torch.zeros((len(self.tokensEmbedding), self.d_model))
        print(self.tokensEmbedding.size())
        for head in  range(self.head):
            print(self.w_q[head].size())
            q_w =  torch.matmul(self.tokensEmbedding, self.w_q[head])
            k_w = torch.matmul(self.tokensEmbedding, self.w_k[head])
            v_k = torch.matmul(self.tokensEmbedding, self.w_v[head])
            head = self.attention(q_w, k_w, v_k)
            # output = np.concatenate(head, axis=-1)
        
        return output

class FFN:
    def __init__(self):
        pass

class Norm:
    def __init__(self):
        pass

if __name__ == '__main__':
    test = torch.tensor(torch.ones((2, 512)))
    heads = MultiHeadAttention(512, test)
    heads.forward()
