import torch
import torch.nn as nn

class GatedResidualNetwork(nn.Module):
    """Standard TFT component to suppress non-relevant features."""
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Projection if input size doesn't match hidden size for the residual connection
        self.project = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

    def forward(self, x):
        residual = self.project(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x) * x
        return self.layer_norm(residual + x)

class TFTClassifier(nn.Module):
    def __init__(self, num_static_feats=3, num_dyn_cont_feats=4, num_classes=14, 
                 passband_vocab_size=6, d_model=128, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Feature Embeddings
        # Categorical embedding for passband (0-5)
        self.passband_emb = nn.Embedding(passband_vocab_size, d_model)
        
        # VSN/Projection for continuous features
        self.static_encoder = GatedResidualNetwork(num_static_feats, d_model, dropout)
        self.dyn_cont_encoder = GatedResidualNetwork(num_dyn_cont_feats, d_model, dropout)
        
        # Combine dynamic features (Continuous + Categorical Passband)
        self.dyn_combiner = GatedResidualNetwork(d_model * 2, d_model, dropout)
        
        # 2. Sequence Processing (Temporal Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model), # *2 because we concat static context and sequence output
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, static, dyn_cont, dyn_cat, mask):
        # 1. Encode Static Features: [batch_size, d_model]
        static_context = self.static_encoder(static)
        
        # 2. Encode Dynamic Features: [batch_size, seq_len, d_model]
        dyn_cont_emb = self.dyn_cont_encoder(dyn_cont)
        dyn_cat_emb = self.passband_emb(dyn_cat)
        
        # Concatenate continuous and categorical along feature dimension, then mix
        dyn_combined = torch.cat([dyn_cont_emb, dyn_cat_emb], dim=-1)
        sequence = self.dyn_combiner(dyn_combined)
        
        # Add static context to every time step in the sequence
        # static_context.unsqueeze(1) creates [batch, 1, d_model] -> broadcasts across seq_len
        sequence = sequence + static_context.unsqueeze(1)
        
        # 3. Transformer Forward Pass
        # PyTorch Transformer expects src_key_padding_mask to be True for padding
        # Our mask is 1 (True) for real, 0 (False) for pad. So we invert it (~).
        pad_mask = ~mask 
        
        # sequence shape: [batch, seq_len, d_model]
        encoded_seq = self.transformer(sequence, src_key_padding_mask=pad_mask)
        
        # 4. Pooling & Classification
        # We cannot just take the last element due to varying sequence lengths.
        # Max-pooling across the time dimension, ignoring padded values
        
        # Set padded values to a large negative number so they don't affect max pooling
        encoded_seq = encoded_seq.masked_fill(pad_mask.unsqueeze(-1), -1e4)
        seq_representation = encoded_seq.max(dim=1)[0] # Shape: [batch, d_model]
        
        # Concatenate the original static context with the temporal representation
        final_representation = torch.cat([seq_representation, static_context], dim=-1)
        
        logits = self.classifier(final_representation)
        return logits