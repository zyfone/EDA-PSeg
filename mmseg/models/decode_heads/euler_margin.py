import torch
from torch import nn
import math
from torch import Tensor


#https://github.com/ermongroup/neuralsort/blob/master/pytorch/neuralsort.py 
class NeuralSort(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False):
        super(NeuralSort, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n x 1
        """
        scores = scores.unsqueeze(-1)
        bsize = scores.size()[0]
        dim = scores.size()[1]
        # one = torch.cuda.FloatTensor(dim, 1).fill_(1)
        device = scores.device
        one = torch.ones(dim, 1, device=device)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(
            one, torch.transpose(one, 0, 1)))
        # scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)
        #            ).type(torch.cuda.FloatTensor)
        scaling = (dim + 1 - 2 * (torch.arange(dim, device=device) + 1)).float()
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C-B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        if self.hard:
            P = torch.zeros_like(P_hat, device='cuda')
            b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(
                dim0=1, dim1=0).flatten().type(torch.cuda.LongTensor)
            r_idx = torch.arange(dim).repeat(
                [bsize, 1]).flatten().type(torch.cuda.LongTensor)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P-P_hat).detach() + P_hat
        return P_hat

class EulerFormer(nn.Module):
    def __init__(self, hidden_size=None, vector_wise=True, tau=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.vector_wise = vector_wise
        self.tau = tau 
        self.delta = nn.Parameter(torch.ones(1,1,self.hidden_size//2)*0.1)
        self.b = nn.Parameter(torch.zeros(1,1,self.hidden_size//2))
        self.log_scale = nn.Parameter(torch.zeros(1,1,self.hidden_size//2))
        self.soft_sort=NeuralSort()
    
    def forward(self, v, type="ro"):
        B, L, D = v.shape

        scores = v.mean(dim=1)  # (B, D)
        P = self.soft_sort(scores)  # (B, D, D)
        v_sorted = torch.matmul(P, v.transpose(1, 2)).transpose(1, 2)  # (B, L, D)

        r, p = v_sorted[..., ::2], v_sorted[..., 1::2]

        # lam = torch.sqrt(r ** 2 + p ** 2 + 1e-9)
        epsilon = 1e-6
        lam = torch.sqrt(r ** 2 + p ** 2 + epsilon)
        # theta = torch.atan2(p, r)
        r_norm = r / lam.detach()
        p_norm = p / lam.detach()
        theta = torch.atan2(p_norm, r_norm)

        if "ro" in type:
            theta = theta * self.delta
            if "query" in type:
                theta = theta + self.b

        # lam = lam * torch.exp(self.log_scale)
        lam = lam * torch.exp(self.log_scale.clamp(-5, 5))
        
        r, p = lam * torch.cos(theta), lam * torch.sin(theta)
        embeddings = torch.stack([r, p], dim=-1).reshape(B, L, D)

        return embeddings
    

class Euler_Attention(nn.Module):

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob=0.1,
        attn_dropout_prob=0.1,
        layer_norm_eps=1e-12,
    ):
        super(Euler_Attention, self).__init__()

        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size")

        self.num_attention_heads = n_heads
        self.attention_head_size = hidden_size // n_heads
        self.all_head_size = hidden_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dense = nn.Linear(hidden_size, hidden_size)

        self.euler = EulerFormer(hidden_size)

    def transpose_for_scores(self, x):
        B, L, _ = x.size()
        x = x.view(B, L, self.num_attention_heads, self.attention_head_size)
        return x

    def forward(self, input_tensor):

        q = self.query(input_tensor)
        k = self.key(input_tensor)
        v = self.value(input_tensor)

        # Euler positional rotation
        q = self.euler(q, "queryro")
        k = self.euler(k, "ro")

        q = self.transpose_for_scores(q).permute(0, 2, 1, 3)
        k = self.transpose_for_scores(k).permute(0, 2, 1, 3)
        v = self.transpose_for_scores(v).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.sqrt_attention_head_size
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size(0), -1, self.all_head_size)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states