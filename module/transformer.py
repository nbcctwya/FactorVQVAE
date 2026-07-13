import torch
import torch.nn as nn
import torch.nn.functional as F
from module.mingpt import GPT
import copy
import numpy as np
from pathlib import Path
from module.vqvae import FactorEncoder, FactorDecoder, FeatureExtractor
from vqtorch.nn import VectorQuant
#from vector_quantize_pytorch import VectorQuantizer
import os
import sys
import math 
from utils import freeze, get_root_dir, load_pretrained_tok_emb

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

class AutoRegressiveTransformer(nn.Module):
    """
    #! For the most part, you should follow the minGPT implementation.
    #! However, be careful to slightly modify the tok_emb (token embedding).
    """
    def __init__(self,
                 temperature,
                 config):
        super().__init__()

        self.sos_token_ids = config['vqvae']['num_factors'] # sos token 

        self.config = config
        self.num_factors = config['vqvae']['num_factors']
        self.label_delay = config['transformer'].get('label_delay', 2)
        if self.label_delay < 1:
            raise ValueError("transformer.label_delay must be at least 1")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # define models
        self.dim           = config['vqvae']['hidden_size']
        self.input_channel = config['vqvae']['input_channel']
        self.dropout       = config['vqvae']['dropout'] # 0.1
        self.num_heads     = config['vqvae']['num_heads']
        self.num_features  = config['vqvae']['num_features']

        self.feature_extractor = FeatureExtractor(
            num_latent  = self.num_features,
            hidden_size = self.dim)

        self.encoder = FactorEncoder(
            input_size  = self.input_channel, 
            hidden_size = self.dim, 
            num_heads   = self.num_heads,
            use_attn    = True,
            dropout     = self.dropout)

        self.decoder = FactorDecoder(
            input_size  = self.dim, hidden_size = self.dim,
            num_elements = config['vqvae']['num_elements']) # num_factors = num_elements
        
        self.quantizer = VectorQuant(
            feature_size = self.dim,                                 # feature dimension corresponding to the vectors
            num_codes    = self.num_factors,                         # number of codebook vectors
            beta         = self.config['quantizer']['beta'],         # (default: 0.9) commitment trade-off
            kmeans_init  = self.config['quantizer']['kmeans_init'],  # (default: False) whether to use kmeans++ init
            norm         = None,                                     # (default: None) normalization for the input vectors
            cb_norm      = None,                                     # (default: None) normalization for codebookc vectors
            affine_lr    = self.config['quantizer']['affine_lr'],    # (default: 0.0) lr scale for affine parameters
            sync_nu      = self.config['quantizer']['sync_nu'],      # (default: 0.0) codebook synchronization contribution
            replace_freq = self.config['quantizer']['replace_freq'], # (default: None) frequency to replace dead codes
            dim=-1,                                                  # (default: -1) dimension to be quantized
            )
        
        # load trained models for encoder, decoder, and quantizer
        self.checkpoint_folder = config['paths']['checkpoint_dir']
        self.load_pretrained_model(config)

        # Initialize transformer
        self.vocab_size = self.config['vqvae']['num_factors']
        self.pkeep = self.config['transformer']['pkeep']
        
        self.transformer = GPT(
            vocab_size = self.vocab_size,
            block_size = self.config['transformer']['num_tokens'] + 1,
            n_layer    = self.config['transformer']['n_layers'],
            n_head     = self.config['transformer']['heads'],
            n_embd     = self.config['transformer']['hidden_size'],
            market_dim = self.config['transformer']['hidden_size'], 
            attn_pdrop = self.config['transformer']['attn_pdrop'],
        )

        # Use Market and MarketAttention
        self.use_market = config['transformer']['use_market']
        self.market_extractor = FeatureExtractor(num_latent = config['vqvae']['market_features'],
                                                 hidden_size = config['transformer']['hidden_size'])
    def load_pretrained_model(self, config):
        saved_model = config['transformer']['saved_model']
        saved_model = f"{saved_model}.ckpt" if saved_model and not saved_model.endswith('.ckpt') else saved_model
        checkpoint_path = Path(self.checkpoint_folder).joinpath(saved_model)
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )['state_dict']
        except FileNotFoundError:
            print(f"Checkpoint not found at {checkpoint_path}.")
            sys.exit(1)

        def load_state_dict(module, prefix):
            state_dict = {k.replace(f'{prefix}.', ''): v for k, v in checkpoint.items() if k.startswith(prefix)}
            module.load_state_dict(state_dict)

        load_state_dict(self.feature_extractor, 'feature_extractor')
        load_state_dict(self.encoder, 'encoder')
        load_state_dict(self.decoder, 'decoder')
        load_state_dict(self.quantizer, 'quantizer')

        freeze(self.encoder)
        freeze(self.quantizer)
        freeze(self.feature_extractor)

        self.encoder.eval()
        self.quantizer.eval()
        self.feature_extractor.eval()

    @torch.no_grad()
    def encode_to_z_q(self, y):
        """
        Encodes input `y` into quantized representation `z_q`.
        """
        z_e = self.encoder(y)
        z_q, vq_dict = self.quantizer(z_e)
        return z_q, vq_dict['q'].squeeze()

    @torch.no_grad()
    def prepare_transformer_inputs(self, known_indices, sequence_length, device):
        """
        Prepares input indices for the transformer, including SOS tokens and masked indices.
        """
        sos_tokens = torch.full((known_indices.shape[0], 1), self.sos_token_ids, dtype=torch.long, device=device)
        
        # Apply masking for denoising training (optional)
        assert 0.0 <= self.pkeep <= 1.0, "pkeep must be in the range [0, 1]"
        if self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep * torch.ones(known_indices.shape, device=device)).round().to(torch.int64)
            random_indices = torch.randint_like(known_indices, self.vocab_size, device=device)
            masked_indices = mask * known_indices + (1 - mask) * random_indices
        else:
            masked_indices = known_indices

        # At prediction day t, labels t-1 ... t-label_delay+1 are not yet
        # realized.  Represent those unavailable positions with the SOS/MASK
        # embedding; never manufacture them by encoding future labels.
        unavailable = sequence_length - 1 - known_indices.shape[1]
        if unavailable < 0:
            raise ValueError("known label sequence is longer than the model window")
        mask_tokens = torch.full(
            (known_indices.shape[0], unavailable), self.sos_token_ids,
            dtype=torch.long, device=device,
        )
        input_indices = torch.cat((sos_tokens, masked_indices, mask_tokens), dim=1)
        return input_indices
    
    def decode_quantized_embeddings(self, firm_features, predicted_indices):
        """
        Passes quantized embeddings through the decoder.
        """
        # Retrieve quantized embeddings from the codebook
        codebook = self.quantizer.get_codebook().to(predicted_indices.device)
        quantized_embeddings = F.embedding(predicted_indices, codebook)

        # Decode the quantized embeddings
        y_hat, _ = self.decoder(firm_char=firm_features, inputs=quantized_embeddings)
        return y_hat
    
    def predict(self, firm_char, known_y, market):
        """Predict y(t) using only labels that are realized by prediction day t."""
        sequence_length = firm_char.shape[1]
        expected_known = sequence_length - self.label_delay
        if known_y.shape[1] != expected_known:
            raise ValueError(
                f"expected {expected_known} known labels for a window of "
                f"{sequence_length}, got {known_y.shape[1]}"
            )

        # A suspended/missing historical return is not observable.  Zero is a
        # neutral, leakage-safe placeholder; never backfill it from a later day.
        known_y = torch.nan_to_num(known_y, nan=0.0, posinf=0.0, neginf=0.0)

        firm_char = torch.nan_to_num(firm_char, nan=0.0, posinf=0.0, neginf=0.0)
        market = torch.nan_to_num(market, nan=0.0, posinf=0.0, neginf=0.0)
        firm_features = self.feature_extractor(firm_char[:, -1:, :])
        market_features = self.market_extractor(market)
        _, known_indices = self.encode_to_z_q(known_y)
        known_indices = known_indices.reshape(known_y.shape[0], -1).long()
        input_indices = self.prepare_transformer_inputs(
            known_indices, sequence_length, firm_char.device
        )

        if self.use_market:
            logits = self.transformer(input_indices, market_features)
        else:
            logits = self.transformer(input_indices)
        target_logits = logits[:, -1:, :]
        predicted_indices = torch.argmax(target_logits, dim=-1)
        y_hat = self.decode_quantized_embeddings(firm_features, predicted_indices)
        return target_logits, y_hat

    def forward(self, firm_char, y, market):
        """
        Forward pass through the model.
        """
        if y.shape[1] != firm_char.shape[1]:
            raise ValueError("firm_char and y must have the same sequence length")

        known_y = y[:, :-self.label_delay, :]
        logits, y_hat = self.predict(firm_char, known_y, market)

        # The target is supervision, not a model input.  Generate it exactly as
        # Stage1 does (on the complete window), then retain only s(t).  The
        # historical tokens from this call are deliberately discarded so they
        # can never carry target/future information into the GPT context.
        safe_y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        _, all_target_indices = self.encode_to_z_q(safe_y)
        all_target_indices = all_target_indices.reshape(y.shape[0], y.shape[1])
        target_indices = all_target_indices[:, -1:].long()
        return logits, target_indices, y_hat
        
    # def top_k_logits(self, logits, k):
    #     v, ix = torch.topk(logits, k)
    #     out = logits.clone()
    #     out[out < v[..., [-1]]] = -float("inf")
    #     return out
    
    # @torch.no_grad()
    # def sample(self, x, c, steps, temperature=1.0, top_k=3):
    #     self.transformer.eval()
    #     x = torch.cat((c, x), dim=1)
    #     for k in range(steps):
    #         logits = self.transformer(x)
    #         logits = logits[:, -1, :] / temperature

    #         if top_k is not None:
    #             logits = self.top_k_logits(logits, top_k)

    #         probs = F.softmax(logits, dim=-1)

    #         ix = torch.multinomial(probs, num_samples=1)

    #         x = torch.cat((x, ix), dim=1)

    #     x = x[:, c.shape[1]:]
    #     self.transformer.train()
    #     return x
