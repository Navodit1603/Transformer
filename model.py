from torch import nn
import torch
import math

class SinusoidalPositions(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        
        position = torch.arange(max_seq_len).unsqueeze(-1) # S, 1
        # inside sine / cosine we have pos * (10_000**-2m/d)
        # for stability, calculate instead exp(-2m/d * log(10_000))
        # multiplier shape D/2, then S, 1 * D/2 -> S, D/2
        multiplier = torch.exp((torch.arange(0, d_model, 2) / d_model) * -math.log(10_000))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * multiplier) # S, D/2
        pe[:, 1::2] = torch.cos(position * multiplier)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has shape B, S, D
        batch_seq_len = x.shape[1]
        return x + self.pe[:batch_seq_len, :]


"""
TODO define your transformer model here. 
this will include: 
    - embed tokens (nn.Embedding)
    - add position encoding (provided)
    - n repetitions of 
        - *masked* self attention (can be single or multi-headed)
        - feedforward (MLP)
        - remember that the layer outputs are added to a residual connection
    - final linear layer with out_features equal to your vocabulary size
"""

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layer=6, n_head=8, 
                 d_ff=2048, max_seq_len=256, dropout=0.1):
        super().__init__()
        
        self.emb = nn.Embedding(vocab_size, d_model) # create token embedding 
        self.pos_emb = SinusoidalPositions(max_seq_len, d_model) # create vectors that represent the position of each token

        # Layers of the transformer
        self.layers = nn.ModuleList([
            Block(d_model, n_head, d_ff, dropout)
            for _ in range(n_layer)
        ])

        self.norm_f = nn.LayerNorm(d_model) # normalization
        self.out = nn.Linear(d_model, vocab_size) # output layer, pass to loss function

    def forward(self, x, padding_mask):
        # x has shape B,S
        x = self.emb(x)    # (B, S) -> (B, S, D)
        x = self.pos_emb(x) # (B, S, D) -> (B, S, D)
        
        # Pass through all N blocks
        for layer in self.layers:
            x = layer(x, padding_mask)
            
        # Final norm and output
        x = self.norm_f(x)
        x = self.out(x)    # (B, S, D) -> (B, S, V)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, max_seq_len=256):
        super().__init__()
        
        # --- 1. Save dimensions ---
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head # Dimension of each head
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
        # --- 2. Causal Mask (Look-ahead mask) ---
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        self.register_buffer('mask', mask.bool())

    def forward(self, x, padding_mask):
        # x shape: (B, S, D)
        # padding_mask shape: (B, S)
        B, S, D = x.shape
        
        # --- 3. Project and Split into Heads ---
        # (B, S, D) -> (B, S, D)
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # (B, S, D) -> (B, S, H, Dk) -> (B, H, S, Dk)
        q = q.view(B, S, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, S, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, S, self.n_head, self.d_k).transpose(1, 2)
        
        # --- 4. Scaled Dot-Product Attention ---
        # (B, H, S, Dk) @ (B, H, Dk, S) -> (B, H, S, S)
        dots = q @ k.transpose(-2, -1) 
        
        scaled_dot = dots / math.sqrt(self.d_k)

        # --- 5. Apply Masks ---
        if padding_mask is not None:
            # Reshape mask from (B, S) to (B, 1, 1, S) for broadcasting
            pad_mask = padding_mask.unsqueeze(1).unsqueeze(1) # (B, 1, 1, S)
            # Fill with -infinity where the mask is 0 (padding)
            scaled_dot = scaled_dot.masked_fill(pad_mask == 0, -1e9)
            
        # (B, H, S, S)
        scaled_dot = scaled_dot.masked_fill(
            self.mask[:S, :S], -1e9 # Using -infinity for future tokens
        )

        scores = torch.softmax(scaled_dot, dim=-1)
        scores = self.dropout(scores)

        # (B, H, S, S) @ (B, H, S, Dk) -> (B, H, S, Dk)
        x = scores @ v

        # --- 6. Combine Heads ---
        # (B, H, S, Dk) -> (B, S, H, Dk) -> (B, S, D)
        x = x.transpose(1, 2).contiguous().view(B, S, D)

        return self.out(x)

class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        
        self.sa = MultiHeadAttention(d_model, n_head, dropout)
        self.norm_sa = nn.LayerNorm(d_model)
        self.drop_sa = nn.Dropout(dropout)
        
        # Feed-Forwarding
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), # size: d_model -> d_ff
            nn.GELU(), # activation
            nn.Linear(d_ff, d_model), # size: d_ff -> d_model
            nn.Dropout(dropout)
        )
        
        self.norm_ff = nn.LayerNorm(d_model)
        self.drop_ff = nn.Dropout(dropout) # Dropout neurons to prevent overfitting

    def forward(self, x, padding_mask):
        
        sa_x = self.norm_sa(x) # Layer normalization
        sa_x = self.sa(sa_x, padding_mask) #Multihead attention
        sa_x = self.drop_sa(sa_x) #drop out
        x = x + sa_x #add back to original input for residual connection
        
        # Same process for feed-forward
        ffn_x = self.norm_ff(x)
        ffn_x = self.ff(ffn_x) 
        ffn_x = self.drop_ff(ffn_x)
        x = x + ffn_x
        
        return x
        



def get_best_model_definition(vocab_size):
    """
    This is the model that will be used in the evaluation script
    Ensure it matches the .pt file provided there
    """
    return Transformer(vocab_size=vocab_size, 
                       d_model=128, # Model dimention 
                       n_layer=6, #number of layers 
                       n_head=8, # number of attention heads
                       d_ff=512, # feed forward dimetions (read somewhere that these are 4x bigger than d)          
                       max_seq_len=256,
                       dropout=0.15) # ignore about 15% of the neurons to avoid overfitting

