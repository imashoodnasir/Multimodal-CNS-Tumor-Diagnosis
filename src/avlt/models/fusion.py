
import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 2),
            nn.Sigmoid()
        )

    def forward(self, f_v, f_t):
        q = self.q_proj(f_t).unsqueeze(1)
        k = self.k_proj(f_v).unsqueeze(1)
        v = self.v_proj(f_v).unsqueeze(1)
        fused, _ = self.attn(q, k, v)   # (B,1,D)
        fused = self.norm(fused.squeeze(1))
        gate_in = torch.cat([f_v, f_t], dim=1)
        alpha_beta = self.gate(gate_in)
        alpha, beta = alpha_beta[:,0:1], alpha_beta[:,1:2]
        f_fused = alpha * f_v + beta * f_t + fused
        return f_fused, alpha.squeeze(1), beta.squeeze(1)
