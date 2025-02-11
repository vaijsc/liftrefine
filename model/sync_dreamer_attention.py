import torch
import torch.nn as nn

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class DepthAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads, dim_head, output_bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Conv2d(query_dim, inner_dim, 1, 1, bias=False)
        self.to_k = nn.Conv3d(context_dim, inner_dim, 1, 1, bias=False)
        self.to_v = nn.Conv3d(context_dim, inner_dim, 1, 1, bias=False)
        if output_bias:
            self.to_out = nn.Conv2d(inner_dim, query_dim, 1, 1)
        else:
            self.to_out = nn.Conv2d(inner_dim, query_dim, 1, 1, bias=False)

    def forward(self, x, context):
        """

        @param x:        b,f0,h,w
        @param context:  b,f1,d,h,w
        @return:
        """
        hn, hd = self.heads, self.dim_head
        b, _, h, w = x.shape
        b, _, d, h, w = context.shape

        q = self.to_q(x).reshape(b,hn,hd,h,w) # b,t,h,w
        k = self.to_k(context).reshape(b,hn,hd,d,h,w) # b,t,d,h,w
        v = self.to_v(context).reshape(b,hn,hd,d,h,w) # b,t,d,h,w

        sim = torch.sum(q.unsqueeze(3) * k, 2) * self.scale # b,hn,d,h,w
        attn = sim.softmax(dim=2)

        # b,hn,hd,d,h,w * b,hn,1,d,h,w
        out = torch.sum(v * attn.unsqueeze(2), 3) # b,hn,hd,h,w
        out = out.reshape(b,hn*hd,h,w)
        return self.to_out(out)


class DepthTransformer(nn.Module):
    def __init__(self, dim, n_heads, d_head, context_dim=None):
        super().__init__()
        inner_dim = n_heads * d_head
        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1, 1),
            nn.GroupNorm(8, inner_dim),
            nn.SiLU(True),
        )
        self.proj_context = nn.Sequential(
            nn.Conv3d(context_dim, context_dim, 1, 1, bias=False), # no bias
            nn.GroupNorm(8, context_dim),
            nn.ReLU(True), # only relu, because we want input is 0, output is 0
        )
        self.depth_attn = DepthAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, context_dim=context_dim, output_bias=False)  # is a self-attention if not self.disable_self_attn
        self.proj_out = nn.Sequential(
            nn.GroupNorm(8, inner_dim),
            nn.ReLU(True),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(8, inner_dim),
            nn.ReLU(True),
            zero_module(nn.Conv2d(inner_dim, dim, 3, 1, 1, bias=False)),
        )


    def forward(self, x, context):
        x_in = x
        x = self.proj_in(x)
        context = self.proj_context(context)
        x = self.depth_attn(x, context)
        x = self.proj_out(x) + x_in
        return x
<<<<<<< HEAD

=======
>>>>>>> dev_aggregate_zero123
