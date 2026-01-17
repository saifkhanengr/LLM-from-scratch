import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

__all__ = [
    'Rope',
    'DeepSeek_MLA',
    'DeepSeek_MoE',
    'DeepSeek_MTP',
    'DeepSeek_V3_Block',
    'DeepSeek_V3_Encoder',
    'DeepSeek_V3_Model',
    'generate_text',
    'clean_response'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model configurations

class Config:
    hidden_size = 128                # Embedding dimension (D)
    latent_dim = hidden_size // 2    # Latent dimension, half of D (a random choice)
    num_heads = 16                   # Number of attention heads (should divide hidden_size)
    pos_dim = 24                     # Positional encoding dimension
    pad_token_id = 50256             # Padding token ID (matches <|endoftext|> in GPT-2 vocab)
    num_shared_experts = 4
    num_routed_experts = 8
    top_k = 8                        # Kr, number of experts selected per token
    bias_update_speed = 0.01
    balance_alpha = 0.01
    lambda_mtp = 0.5                 # λ, weighting
    num_depths = 3                   # D, number of prediction depths
    vocab_size= tiktoken.get_encoding("gpt2").n_vocab  # Vocab size of tiktoken’s GPT-2 vocab (50257)
    layer_norm_eps = 1e-5            # Small epsilon value for numerical stability in layer normalization
    num_blocks = 12                  # Number of transformer blocks to stack in the model
    batch_size = 64                  # Number of sequences per batch
    context_length = 60              # Number of tokens per sequence


class Rope(nn.Module):

    """
    Rotary Position Embedding (RoPE) module.
    Applies rotary position encoding to an input tensor of shape (B, H, S, D),
    """

    def __init__(self, dim, max_seq_len = 4096):
        super().__init__()

        # Safety check: RoPE requires even dimensionality (for splitting into pairs)
        assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"

        self.dim = dim
        self.max_seq_len = max_seq_len

        # Step 2: Compute rotation frequencies for sinusoidal positions
        # inv_freq[i] = 1 / (10000^(2i/dim)), where i = 0, 1, ..., dim/2 - 1
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        # Store as non-trainable buffer
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute and cache cos/sin values up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        """
        Precompute cosine and sine embeddings for all positions up to seq_len.
        This avoids recomputing trig functions during every forward pass.
        """

        # Positions: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)

        # Step 3: Compute rotation angles (per position and dimension pair)
        # Each row is t * inv_freq[i], giving angular frequency per dimension
        freqs = torch.outer(t, self.inv_freq)

        # Duplicate for concatenation of sin and cos values, shape: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Step 4: Construct rotation matrix elements (cos and sin)
        # Register as buffers with shape (1, 1, seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

        # Track how many positions we have cached
        self.max_seq_len = seq_len

    def forward(self, x, seq_len, position_offset = 0):

        """
        Apply RoPE to input tensor.

        Args:
            x: Input tensor of shape (B, H, S, D)
            seq_len: Actual sequence length to encode
            position_offset: Offset for decoding continuation (default = 0)

        Returns:
            Tensor with RoPE applied, same shape as x.
        """

        device = x.device

        # Ensure input matches expected dimensionality
        assert x.shape[-1] == self.dim, (
            f"RoPE input dim mismatch: expected {self.dim}, got {x.shape[-1]}"
        )

        seq_len_x = x.size(-2)  # sequence length from input tensor

        if (position_offset + seq_len) > self.max_seq_len:
            # Rebuild cache with doubled size for efficiency
            self._build_cache(max(position_offset + seq_len, self.max_seq_len * 2))

        # Select only the needed positions
        cos = self.cos_cached[:, :, position_offset:position_offset + seq_len, :].to(device)
        sin = self.sin_cached[:, :, position_offset:position_offset + seq_len, :].to(device)

        # Ensure cache slice matches actual input sequence length
        assert cos.shape[2] == seq_len_x, (
            f"RoPE seq_len mismatch: expected {seq_len_x}, got {cos.shape[2]}"
        )

        # Step 1: Split Q/K into 2D subspaces (pairs of dimensions)
        # Split last dimension into pairs: (x1, x2)
        x1, x2 = x.chunk(2, dim=-1)

        # Step 5 and Step 6: Apply rotation to each 2D subspace
        # Rotate pairs: (x1, x2) → (-x2, x1)
        rotated = torch.cat((-x2, x1), dim=-1)
        # Apply rotary transformation: elementwise (x*cos + rotated*sin)
        result = x*cos + rotated*sin

        return result
        

class DeepSeek_MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size # Embedding dimension
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.latent_dim = config.latent_dim
        self.pos_dim = config.pos_dim
        self.max_seq_len = getattr(config, 'max_seq_len', 512)  # Add max sequence length
        self.pad_token_id = getattr(config, 'pad_token_id', 50256)  # Default pad token ID to 50256 of tiktoken’s GPT-2 vocab, same as <|endoftext|> in the tiktoken’s GPT-2 vocab


        assert self.hidden_size % self.num_heads == 0, f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        # Ensure pos_dim is even for RoPE
        assert self.pos_dim % 2 == 0, f"pos_dim ({self.pos_dim}) must be even for RoPE"

        # Latent compression projections
        self.W_DKV = nn.Linear(self.hidden_size, self.latent_dim, bias=False)  # KV compression
        self.W_DQ = nn.Linear(self.hidden_size, self.latent_dim, bias=False)   # Q compression

        # Content projection from latent to multi-head space
        self.W_UK = nn.Linear(self.latent_dim, self.hidden_size, bias=False)  # K content
        self.W_UV = nn.Linear(self.latent_dim, self.hidden_size, bias=False)  # V content
        self.W_UQ = nn.Linear(self.latent_dim, self.hidden_size, bias=False)  # Q content

        # Positional projections (RoPE pathway)
        self.W_KR = nn.Linear(self.hidden_size, self.pos_dim, bias=False)     # K positional
        self.W_QR = nn.Linear(self.latent_dim, self.num_heads * self.pos_dim, bias=False)  # Q positional

        # Output projection
        self.W_O = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # RoPE initialization
        self.rope_k = Rope(self.pos_dim)
        self.rope_q = Rope(self.pos_dim)

        # ---- Precomputed causal mask ----
        # Create upper triangular mask with ones above diagonal and convert to boolean
        self.register_buffer("causal_mask", torch.triu(torch.ones(self.max_seq_len, self.max_seq_len), diagonal=1).bool())

    def forward(self, hidden_states, input_tokens=None, mode="train", use_cache=False, past_key_values=None, attention_mask=False):
        batch_size, seq_len, hidden_size = hidden_states.shape
        assert hidden_size == self.hidden_size, f"hidden_size mismatch: got {hidden_size}, expected {self.hidden_size}"

        # ---- Latent compressions ----
        c_KV = self.W_DKV(hidden_states)   # (batch_size, seq_len, latent_dim)
        c_Q  = self.W_DQ(hidden_states)    # (batch_size, seq_len, latent_dim)

        # ---- Content projections (per-head) ----
        k_C = self.W_UK(c_KV).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, H, seq_len, head_dim)
        v_C = self.W_UV(c_KV).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, H, seq_len, head_dim)
        q_C = self.W_UQ(c_Q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)   # (batch_size, H, seq_len, head_dim)

        # ---- Positional projections ----
        k_R = self.W_KR(hidden_states)   # (batch_size, seq_len, pos_dim)
        q_R = self.W_QR(c_Q).view(batch_size, seq_len, self.num_heads, self.pos_dim).transpose(1, 2)  # (batch_size, H, seq_len, pos_dim)

        # ---- Determine past length for RoPE position_offset ----
        past_len = 0 if past_key_values is None else past_key_values[0].size(2)

        # ---- Apply RoPE (position offset = past_len) ----
        k_R = self.rope_k(k_R.unsqueeze(1).expand(-1, self.num_heads, -1, -1), seq_len=seq_len, position_offset=past_len)  # (batch_size, H, seq_len, pos_dim)
        q_R = self.rope_q(q_R, seq_len=seq_len, position_offset=past_len)  # (batch_size, H, seq_len, pos_dim)

        ######### TRAINING MODE #########

        if mode == "train":
            k = torch.cat([k_C, k_R], dim=-1)  # (batch_size, H, seq_len, head_dim + pos_dim)
            q = torch.cat([q_C, q_R], dim=-1)  # (batch_size, H, seq_len, head_dim + pos_dim)

            scale = 1.0 / math.sqrt(q.shape[-1]) # same as scale = 1.0 / math.sqrt(head_dim + pos_dim)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (batch_size, H, seq_len, seq_len)

            # ---- Apply mask (causal + padding) ----
            if attention_mask:
                # Mask truncated to the number of tokens and converted to boolean
                mask_bool = self.causal_mask[:seq_len, :seq_len]

                # Convert boolean mask to -inf format for attention
                causal_mask = mask_bool.float().masked_fill(mask_bool, float('-inf'))

                # Create padding mask from hidden states
                padding_mask = (input_tokens == self.pad_token_id) #.all(dim=-1)  # (B, S) - True where all features are 50256

                # Expand padding mask to match attention scores shape
                padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
                padding_mask = padding_mask.expand(-1, self.num_heads, seq_len, -1)  # (B, H, S, S)
                padding_mask = padding_mask.float().masked_fill(padding_mask, float('-inf'))

                # Combine causal and padding masks
                full_mask = causal_mask.unsqueeze(0).unsqueeze(0) + padding_mask
                attn_scores = attn_scores + full_mask


            attn_probs = F.softmax(attn_scores, dim=-1)
            o_heads = torch.matmul(attn_probs, v_C)  # (batch_size, H, seq_len, head_dim)

            kv_cache = None  # training returns no cache

        ######### INFERENCE MODE #########

        elif mode == "inference":
            # Concatenate past and current per-head keys/values/pos if provided
            if past_key_values is None:
                k_C_total = k_C  # (batch_size, H, seq_len, head_dim)
                v_C_total = v_C
                k_R_total = k_R
                q_R_total = q_R
                c_KV_total = c_KV.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (batch_size, H, seq_len, latent_dim)
                total_len = seq_len
            else:
                # past_key_values: (past_k_cache, past_v_cache, past_kR_cache, past_qR_cache)
                past_k_cache, past_v_cache, past_k_R_cache, past_q_R_cache, past_c_KV_total = past_key_values
                # Append along sequence dim (dim=2 for per-head)
                k_C_total = torch.cat([past_k_cache, k_C], dim=2)   # (batch_size, H, past_len+seq_len, head_dim)
                v_C_total = torch.cat([past_v_cache, v_C], dim=2)
                k_R_total = torch.cat([past_k_R_cache, k_R], dim=2)  # (batch_size, H, past_len+seq_len, pos_dim)
                q_R_total = torch.cat([past_q_R_cache, q_R], dim=2)
                c_KV_total = torch.cat([past_c_KV_total, c_KV.unsqueeze(1).expand(-1, self.num_heads, -1, -1)], dim=2) # (batch_size, H, total_len, latent_dim)
                total_len = k_C_total.size(2)


            # q_latent computation
            W_UK_heads = self.W_UK.weight.view(self.num_heads, self.head_dim, self.latent_dim)
            q_latent = torch.matmul(q_C, W_UK_heads)  # (batch, heads, seq_len, latent_dim)

            k_hat = torch.cat([c_KV_total, k_R_total], dim=-1)
            q_hat = torch.cat([q_latent, q_R], dim=-1)  # (batch_size, H, seq_len, head_dim+pos_dim)

            # Attention
            scale = 1.0 / math.sqrt(k_hat.shape[-1])
            attn_scores = torch.matmul(q_hat, k_hat.transpose(-2, -1)) * scale  # (batch_size, H, seq_len, total_len)

            # ---- Apply mask (causal + padding, cache-aware) ----
            if attention_mask:

                mask_bool = self.causal_mask[:total_len, :total_len]
                causal_mask_base = mask_bool.float().masked_fill(mask_bool, float('-inf'))

                offset = total_len - seq_len
                causal_mask = causal_mask_base[offset:offset+seq_len, :total_len].unsqueeze(0).unsqueeze(0)

                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, total_len)

                # Create padding mask from hidden states
                padding_mask = (input_tokens == self.pad_token_id) #.all(dim=-1)  # (B, S) - True where all features are 50256

                # For inference with cache, we need to handle the full sequence length
                padding_mask_full = torch.zeros(batch_size, total_len, device=hidden_states.device, dtype=torch.bool)
                padding_mask_full[:, -seq_len:] = padding_mask  # Only the current tokens have padding

                padding_mask_expanded = padding_mask_full.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, seq_len, -1)
                padding_mask_expanded = padding_mask_expanded.float().masked_fill(padding_mask_expanded, float('-inf'))

                full_mask = causal_mask + padding_mask_expanded
                attn_scores = attn_scores + full_mask


            attn_probs = F.softmax(attn_scores, dim=-1)
            o_hat = torch.matmul(attn_probs, c_KV_total)  # (batch_size, H, seq_len, latent_dim)

            # 2. Apply per-head W_UV projection (Absorb step)
            W_UV_heads = self.W_UV.weight.view(self.num_heads, self.head_dim, self.latent_dim) # [H, head_dim, latent_dim]
            o_heads = torch.matmul(o_hat, W_UV_heads.transpose(1, 2))  # [batch_size, H, seq_len, head]

            # Prepare kv_cache tuple to return (present caches covering full sequence)
            if use_cache:
                kv_cache = (
                    k_C_total.detach(),
                    v_C_total.detach(),
                    k_R_total.detach(),
                    q_R_total.detach(),
                    c_KV_total.detach()
                )
            else:
                kv_cache = None

        else:
            raise ValueError("mode must be 'train' or 'inference'")

        # ---- Final projection ----
        o = o_heads.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)  # (batch_size, seq_len, hidden_size)
        attn_output = self.W_O(o)  # (batch_size, seq_len, hidden_size)

        return attn_output, kv_cache
        

class DeepSeek_MoE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size # Embedding dimension
        self.latent_dim = config.latent_dim
        self.num_shared_experts = config.num_shared_experts
        self.num_routed_experts = config.num_routed_experts
        self.top_k = config.top_k  # Kr
        self.bias_update_speed = config.bias_update_speed
        self.balance_alpha = config.balance_alpha

        assert self.top_k <= self.num_routed_experts, f"top_k: ({self.top_k}) exceeds available experts: ({self.num_routed_experts})"

        # Expert centroids for affinity scores
        self.expert_centroids = nn.Parameter(
            torch.empty(self.num_routed_experts, self.hidden_size)
        )

        # Bias terms for load balancing
        self.register_buffer("expert_biases", torch.zeros(self.num_routed_experts))

        # Shared experts
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.latent_dim),
                nn.SiLU(),
                nn.Linear(self.latent_dim, self.hidden_size)
            ) for _ in range(self.num_shared_experts)
        ])

        # Routed experts
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.latent_dim),
                nn.SiLU(),
                nn.Linear(self.latent_dim, self.hidden_size)
            ) for _ in range(self.num_routed_experts)
        ])

        # Initialize centroids
        nn.init.xavier_uniform_(self.expert_centroids)

    def forward(self, hidden_states, training=True):

        batch_size, seq_len, hidden_dim = hidden_states.shape
        assert hidden_dim == self.hidden_size, f"Input hidden size mismatch: got {hidden_dim}, expected {self.hidden_size}."
        total_tokens = batch_size * seq_len

        # ========== Compute affinity scores ==========
        # Equation: s_i,t = Sigmoid(u_t^T e_i)
        flat_input = hidden_states.view(-1, hidden_dim)
        affinity_scores = torch.sigmoid(
            F.linear(flat_input, self.expert_centroids)  # u_t^T e_i
        ).view(batch_size, seq_len, self.num_routed_experts)

        # ========== Top-K routing with bias ==========
        # Equation: Use biased scores s_i,t + b_i for routing selection
        biased_scores = affinity_scores + self.expert_biases

        # Get top-K experts using biased scores
        topk_values, topk_indices = torch.topk(biased_scores, self.top_k, dim=-1)

        # Create mask for selected experts
        expert_mask = torch.zeros_like(affinity_scores)
        expert_mask.scatter_(-1, topk_indices, 1.0)

        # ========== Compute gating values ==========
        # Equation: g'_i,t = s_i,t if selected, 0 otherwise
        selected_scores = affinity_scores * expert_mask

        # Equation: g_i,t = g'_i,t / sum_j(g'_j,t) - normalization
        gating_values = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # ========== Shared experts computation ==========
        # Equation: ∑_{i=1}^{N_s} FFN_i^{(s)}(u_t)
        shared_output = sum(expert(hidden_states) for expert in self.shared_experts)

        # ========== Routed experts computation ==========
        # Equation: ∑_{i=1}^{N_r} g_i,t FFN_i^{(r)}(u_t)
        flat_gating = gating_values.view(-1, self.num_routed_experts)
        flat_indices = topk_indices.view(-1, self.top_k)

        # Precompute all expert outputs: FFN_i^{(r)}(u_t) for all experts
        all_expert_outputs = torch.stack([
            expert(flat_input) for expert in self.routed_experts
        ], dim=1)  # [total_tokens, num_routed_experts, hidden_size]

        # Gather outputs for selected experts and apply gating
        expanded_indices = flat_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        selected_outputs = all_expert_outputs.gather(1, expanded_indices)  # Get FFN outputs for top-k experts

        gating_weights = flat_gating.gather(1, flat_indices).unsqueeze(-1)  # Get g_i,t for selected experts
        routed_output_flat = (selected_outputs * gating_weights).sum(dim=1)  # ∑ g_i,t * FFN_i^{(r)}(u_t)
        routed_output = routed_output_flat.view(batch_size, seq_len, hidden_dim)

        # ========== Load balancing updates ==========
        aux_loss = torch.tensor(0.0, device=hidden_states.device)

        if training:
            # ========== Bias Update ==========
            # Count how many times each expert is selected (or the number of tokens routed to that expert)
            expert_counts = torch.bincount(
                topk_indices.view(-1),
                minlength=self.num_routed_experts
            ).float()
            expert_loads = expert_counts / total_tokens  # Load proportion for each expert

            target_load = torch.ones_like(expert_loads) / self.num_routed_experts  # Ideal balanced load
            load_diff = expert_loads - target_load  # Positive = overloaded, Negative = underloaded
            # Update: decrease bias for overloaded experts, increase for underloaded
            self.expert_biases -= self.bias_update_speed * load_diff

            # ========== Sequence-wise Auxiliary Loss ==========
            # Equation: f_i = (N_r / (K_r * T)) * ∑_t 𝟙(s_i,t ∈ TopK)
            f_i = expert_mask.view(-1, self.num_routed_experts).sum(dim=0)  # Count selections per expert
            f_i = f_i * (self.num_routed_experts / (self.top_k * seq_len))  # Normalize by sequence length
            f_i = f_i / batch_size  # Average over batch

            # Equation: P_i = (1/T) ∑_t s'_i,t where s'_i,t = s_i,t / ∑_j s_j,t
            s_prime = affinity_scores / (affinity_scores.sum(dim=-1, keepdim=True) + 1e-8)  # Normalized affinities
            P_i = s_prime.view(-1, self.num_routed_experts).mean(dim=0)  # Average over all tokens

            # Equation: ℒ_Bal = α * ∑_{i=1}^{N_r} f_i * P_i
            aux_loss = self.balance_alpha * (f_i * P_i).sum()

        # ========== Final output ==========
        # Equation: O_t = X_t + shared_experts +  routed_experts
        output = hidden_states + shared_output + routed_output

        return output, aux_loss
        

class DeepSeek_MTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size # Embedding dimension
        self.vocab_size = config.vocab_size
        self.num_depths = config.num_depths    # D (Please note that this D is different from Embedding dimension D; feel free to replace it with another notation)
        self.lambda_mtp = config.lambda_mtp    # λ
        self.max_seq_len = getattr(config, 'max_seq_len', 512)  # Add max sequence length
        self.pad_token_id = getattr(config, 'pad_token_id', 50256)  # Default pad token ID to 50256 of tiktoken’s GPT-2 vocab, same as <|endoftext|> in the tiktoken’s GPT-2 vocab

        assert self.hidden_size % config.num_heads == 0,f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({config.num_heads})"

        # ===== Shared layers =====
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)   # shared Emb(·)
        self.output_head = nn.Linear(self.hidden_size, self.vocab_size)    # shared OutHead(·)

        # ---- Create D Transformer blocks TRM_k ----
        self.trm_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.latent_dim,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(self.num_depths)
        ])

        # ---- Projection matrices M_k ∈ ℝ^{d×2d} ----
        self.proj_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(self.hidden_size, 2 * self.hidden_size))
            for _ in range(self.num_depths)
        ])

        # ---- RMSNorm layers ----
        self.rmsnorm_h = nn.RMSNorm(self.hidden_size)
        self.rmsnorm_e = nn.RMSNorm(self.hidden_size)

        # ---- Precomputed causal mask ----
        # Create upper triangular mask with ones above diagonal and convert to boolean
        #self.register_buffer("causal_mask", torch.triu(torch.ones(self.max_seq_len, self.max_seq_len), diagonal=1).bool())
        self.register_buffer("causal_mask", torch.triu(torch.ones(self.max_seq_len, self.max_seq_len, device=device)).bool())


    def forward(self, hidden_states, input_tokens=None, mode="train", attention_mask=True):
        batch_size, seq_len, hidden_size = hidden_states.shape

        assert hidden_size == self.hidden_size, f"hidden_states last dim {hidden_size} != expected hidden_size {self.hidden_size}"

        if mode == "train":
            assert input_tokens is not None, "input_tokens required in training mode"
            assert input_tokens.shape == (batch_size, seq_len), f"input_tokens {(input_tokens.shape)} must match batch & seq length of hidden_states= {[batch_size, seq_len]}"

            mtp_losses = []

            # Use separate variable to prevent in-place overwriting
            h_current = hidden_states

            # ===== MTP depths loop =====
            for k in range(1, self.num_depths + 1):
                current_seq_len = h_current.shape[1]  # Use current sequence length
                if current_seq_len - k <= 0:
                    break  # nothing left to predict


                # ---- h'_i^k = M_k [RMSNorm(h_i^{k−1}); RMSNorm(Emb(t_{i+k}))] ----
                h_prev = h_current[:, :current_seq_len - k, :]            # h_i^{k−1}
                emb_shifted = self.embedding(input_tokens[:, k:])         # Emb(t_{i+k})
                h_prev_norm = self.rmsnorm_h(h_prev)
                emb_norm = self.rmsnorm_e(emb_shifted)
                concat = torch.cat([h_prev_norm, emb_norm], dim=-1)       # concat [h; e]
                h_prime_k = torch.matmul(concat, self.proj_matrices[k - 1].T)

                # ---- causal + padding attention mask ----
                causal_mask = None
                padding_mask = None

                if attention_mask:
                    # Get the actual sequence length for this depth
                    L = current_seq_len - k

                    # Original mask truncated to the number of tokens and converted to boolean
                    causal_mask = self.causal_mask[:L, :L]

                    # Create padding mask from input tokens (also boolean)
                    padding_mask = (input_tokens[:, k:current_seq_len] == self.pad_token_id)  # (B, L)

                # ---- Transformer block TRM_k(h'_i^k) ----
                h_k = self.trm_blocks[k - 1](h_prime_k, src_mask=causal_mask, src_key_padding_mask=padding_mask)

                # ---- logits = OutHead(h_i^k) ----
                mtp_logits = self.output_head(h_k)

                # ---- Cross-entropy loss ----
                target_k = input_tokens[:, k:current_seq_len]  # shift targets by +k, match current length
                loss_k = F.cross_entropy(
                    mtp_logits.reshape(-1, self.vocab_size),
                    target_k.reshape(-1),
                    reduction="mean",
                    ignore_index=self.pad_token_id
                )
                mtp_losses.append(loss_k)

                # Update h_current for next depth (maintain causal chain)
                h_current = torch.cat([h_k, h_current[:, current_seq_len - k:, :]], dim=1)

            assert mtp_losses, "No valid MTP losses computed"
            mtp_loss = self.lambda_mtp * torch.stack(mtp_losses).mean()
            return mtp_loss, mtp_logits

        elif mode == "inference":
            # completely skip MTP path — just run the shared output head
            logits = self.output_head(hidden_states)     # [B, S, V]
            predicted_ids = torch.argmax(logits, dim=-1) # [B, S]
            return predicted_ids, logits

        else:
            raise ValueError(f"Invalid mode '{mode}', must be 'train' or 'inference'")

            
class DeepSeek_V3_Block(nn.Module):
    """
    Single-Block Transformer.
    """
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps # Small epsilon value for numerical stability in layer normalization

        # --- Layers ---

        # Input normalization
        self.rms_norm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # DeepSeek_MLA
        self.attention = DeepSeek_MLA(config)

        # Post-attention normalization
        self.rms_norm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # DeepSeek_MoE
        self.moe = DeepSeek_MoE(config)

        # Final normalization
        self.rms_norm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # Linear Output
        self.linear_output = nn.Linear(self.hidden_size, self.hidden_size)


    def forward(self, hidden_states,input_tokens=None, mode="train", use_cache=False, past_key_values=None, attention_mask=False):


        assert hidden_states.dim() == 3, (f"hidden_states must have shape [batch, seq_len, hidden_size], got {hidden_states.shape}.")
        assert hidden_states.size(-1) == self.hidden_size, (f"Last dim mismatch: expected {self.hidden_size}, got {hidden_states.size(-1)}.")

        # Input normalization
        normed_states = self.rms_norm1(hidden_states)

        # Multi-Head Latent Attention
        attn_output, kv_cache = self.attention(
            hidden_states = normed_states,
            input_tokens = input_tokens,
            mode= mode,
            use_cache= use_cache,
            past_key_values=past_key_values,
            attention_mask=attention_mask
        )

        assert attn_output.shape == hidden_states.shape, (f"attn_output shape {attn_output.shape} != hidden_states {hidden_states.shape}.")

        # Residual connection
        hidden_states = hidden_states + attn_output

        # Post-attention normalization
        normed_states = self.rms_norm2(hidden_states)

        # DeepSeekMoE
        moe_output, aux_loss = self.moe(normed_states)

        # Residual connection
        hidden_states = hidden_states + moe_output

        # Final normalization
        hidden_states = self.rms_norm3(hidden_states)

        # Final Output
        hidden_states = self.linear_output(hidden_states)

        return hidden_states, kv_cache, aux_loss
        

class DeepSeek_V3_Encoder(nn.Module):
    """
    Multi-Block Transformer.
    """

    def __init__(self, config):
        super().__init__()

        self.num_blocks = config.num_blocks # Number of transformer blocks to stack in the model
        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps
        self.vocab_size = config.vocab_size

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            DeepSeek_V3_Block(config)
            for _ in range(self.num_blocks)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # ---- Final output ----
        self.output = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # MTP head (Multi-Token Prediction)
        self.mtp = DeepSeek_MTP(config)


    def forward(self,hidden_states, input_tokens=None, mode="train", past_key_values=None, use_cache=False, attention_mask=False):

        assert hidden_states.dim() == 3, (f"hidden_states must have shape [batch, seq_len, hidden_size], got {hidden_states.shape}.")

        if past_key_values is None:
            past_key_values = [None] * self.num_blocks

        new_past_key_values = [] if use_cache else None

        # Forward through stacked transformer blocks
        for i, block in enumerate(self.blocks):
            hidden_states, kv_cache, aux_loss = block(
                hidden_states=hidden_states,
                input_tokens=input_tokens,
                mode=mode,
                use_cache=use_cache,
                past_key_values=past_key_values[i],
                attention_mask=attention_mask,
            )
            if use_cache:
              new_past_key_values.append(kv_cache)

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        # Output
        logits = self.output(hidden_states)  # [B, S, V]

        # MTP output handling
        if mode == "train" and input_tokens is not None:
            mtp_loss, mtp_logits = self.mtp(
                hidden_states=hidden_states,
                input_tokens=input_tokens,
                mode = "train",
                attention_mask=attention_mask
            )
            return logits, mtp_loss, mtp_logits, aux_loss
        else: # mode == "inference"
            predicted_ids, mtp_logits = self.mtp(
                hidden_states,
                input_tokens=input_tokens,
                mode = "inference",
                attention_mask=attention_mask
            )
            return predicted_ids, logits, new_past_key_values


class DeepSeek_V3_Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding Layer
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Core model
        self.model = DeepSeek_V3_Encoder(config)

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)


    def forward(self, input_tokens = None, mode="train", use_cache=False, past_key_values=None,attention_mask=False):

        # Generate embeddings from input tokens
        hidden_states = self.embedding(input_tokens)
        batch_size, seq_len = input_tokens.shape

        # Core model forward
        outputs = self.model(
            hidden_states,
            mode=mode,
            input_tokens=input_tokens,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        if mode == "train":
            logits, mtp_loss, mtp_logits, aux_loss = outputs

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_tokens[..., 1:].contiguous()

            main_loss = self.ce_loss(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

            # Combine losses
            total_loss = main_loss + mtp_loss
            return total_loss, main_loss, mtp_loss, aux_loss, logits

        else: # mode == "inference"
            predicted_ids, logits, new_cache = outputs
            return predicted_ids, logits, new_cache

            

# Code adapted from  Sebastian Raschka
def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=20, eos_id=None, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = input_ids.clone()

    past_key_values = None  # cache for inference

    for _ in range(max_length):

        if past_key_values is None:
            idx_cond = generated          # full prompt (first step)
        else:
            idx_cond = generated[:, -1:]  # only last token

        with torch.no_grad():
            # Use mode="inference" and cache past keys/values
            predicted_ids, logits, past_key_values = model(
                input_tokens=idx_cond.to(device),
                mode="inference",
                use_cache=True,
                past_key_values=past_key_values,
                attention_mask=True
            )

            logits = logits[:, -1, :]  # last token logits

        # Top-k filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_val, torch.tensor(float("-inf"), device=device), logits)

        # Temperature + sampling or greedy
        if temperature > 0:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        # Stop if EOS token generated
        if eos_id is not None and next_token.item() == eos_id:
            break

        # Append generated token
        generated = torch.cat((generated, next_token.to(device)), dim=1)

    # Decode full sequence back to text
    return tokenizer.decode(generated[0].tolist())

def clean_response(generated_text):
    if not generated_text:
        return "Sorry, I couldn't generate a response."
    
    text = str(generated_text)
    
    # Print the prompt part
    if "Response:" in text:
        prompt_part = text.split("Response:", 1)[0] + "Response:"
    else:
        prompt_part = ""
    
    print("=======================================")
    print(f"{prompt_part.strip()}")
    
    # Extract response
    if "Response:" in text:
        text = text.split("Response:", 1)[1]
    
    # Truncate at <|endoftext|>
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>", 1)[0]
    
    # Remove non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c.isspace())
    
    # If text is empty after cleaning, return a default message
    if not text.strip():
        return "I'm not sure how to answer that. Could you ask in a different way?"
    
    return text.strip()

if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be executed directly.")
