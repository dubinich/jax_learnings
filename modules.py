import dataclasses
import jax
import jax.numpy as jnp

# TODO: add JAX typing with expected shapes and tensor types.
# TODO: use flax nn.Module?

@dataclasses.dataclass
class TransformerConfig:
    n_layers: int
    d_vocab: int
    d_model: int
    d_head: int
    d_mlp: int
    n_head: int
    n_ctx: int
    init_range: float = 0.02
    ln_eps: float = 1e-5


class LayerNorm:
    def __init__(self, key, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.params = self.initialize_params(key)

    def initialize_params(self, key):
        w_key, b_key = jax.random.split(key)
        w = jax.nn.initializers.ones(w_key, self.d_model)
        b = jax.nn.initializers.zeros(b_key, self.d_model)
        return {"w": w, "b": b}

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        std = jnp.sqrt(var + self.eps)
        x_normalized = (x - mean) / std

        w, b = self.params["w"], self.params["b"]
        output = x_normalized * w + b
        return output
    

class Embed:
    def __init__(self, key, d_vocab: int, d_model: int, init_range: float=0.02):
        self.d_vocab = d_vocab
        self.d_model = d_model
        self.init_range = init_range
        self.params = self.initialize_params(key)

    def initialize_params(self, key):
        init = jax.nn.initializers.normal(self.init_range)
        return {
            "W_E": init(key, (self.d_vocab, self.d_model))
            }

    def __call__(self, x):
        W_E = self.params["W_E"]
        return W_E[x]


class PosEmbed:
    def __init__(self, key, n_ctx: int, d_model: int, init_range: float=0.02):
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.init_range = init_range
        self.params = self.initialize_params(key)

    def initialize_params(self, key):
        init = jax.nn.initializers.normal(self.init_range)
        return {
            "W_P": init(key, (self.n_ctx, self.d_model))
            }

    def __call__(self, x):
        batch, seqlen = x.shape
        W_P = self.params["W_P"]
        pos_embed = W_P[:seqlen]
        pos_embed = jnp.expand_dims(pos_embed, 0)
        pos_embed = jnp.tile(pos_embed, (batch, 1, 1))
        return pos_embed


class Attention:
    def __init__(self, key, n_head: int, d_head: int, d_model: int, init_range: float=0.02):
        self.n_head = n_head
        self.d_head = d_head
        self.d_model = d_model
        self.init_range = init_range
        self.params = self.initialize_params(key)

    def initialize_params(self, key):
        w_init = jax.nn.initializers.normal(self.init_range)
        return {
                "W_Q": w_init(key, (self.n_head, self.d_model, self.d_head)),
                "W_K": w_init(key, (self.n_head, self.d_model, self.d_head)),
                "W_V": w_init(key, (self.n_head, self.d_model, self.d_head)),
                "W_O": w_init(key, (self.n_head, self.d_head, self.d_model)),
                "b_Q": jax.nn.initializers.zeros(key, (self.n_head, self.d_head)),
                "b_K": jax.nn.initializers.zeros(key, (self.n_head, self.d_head)),
                "b_V": jax.nn.initializers.zeros(key, (self.n_head, self.d_head)),
                "b_O": jax.nn.initializers.zeros(key, self.d_model)
        }

    def _get_causal_mask(self, attn_scores):
        _,_,query_pos,key_pos = attn_scores.shape
        tril = jnp.tril(jnp.ones((query_pos,key_pos)))
        tril = jnp.expand_dims(tril, (0,1))
        return tril

    def __call__(self, x):
        q = jnp.einsum(
            "bse,ned->bsnd", x, self.params["W_Q"]
            ) + self.params["b_Q"]
        k = jnp.einsum(
            "bse,ned->bsnd", x, self.params["W_K"]
            ) + self.params["b_K"]
        v = jnp.einsum(
            "bse,ned->bsnd", x, self.params["W_V"]
            ) + self.params["b_V"]

        attn_scores = jnp.einsum(
            "bqnd,bknd->bnqk", q, k
            ) / jnp.sqrt(self.d_head)
        mask = self._get_causal_mask(attn_scores)
        attn_scores_masked = jax.nn.softmax(
            jnp.where(mask, attn_scores, -jnp.inf), axis=-1)

        attn_out = jnp.einsum(
            "bknd,bnqk->bqnd", v, attn_scores_masked
            )
        attn_out = jnp.einsum(
            "bsnd,nde->bse", attn_out, self.params["W_O"]
            ) + self.params["b_O"]
        return attn_out


class NewGELU:
    def __call__(self, x):
        tanh_output = jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.pow(x, 3.0)))
        return 0.5 * x * (1.0 + tanh_output)
    

class MLP:
    def __init__(self, key, d_model, d_mlp, init_range:float=0.02):
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.init_range = init_range
        self.gelu = NewGELU()
        self.params = self.initialize_params(key)

    def initialize_params(self, key):
        w_init = jax.nn.initializers.normal(self.init_range)
        return {
                "W_in": w_init(key, (self.d_model, self.d_mlp)),
                "W_out": w_init(key, (self.d_mlp, self.d_model)),
                "b_in": jax.nn.initializers.zeros(key, self.d_mlp),
                "b_out": jax.nn.initializers.zeros(key, self.d_model),
        }

    def __call__(self, x):
        w_in = jnp.einsum(
            "bse,ed->bsd", x, self.params["W_in"]
            ) + self.params["b_in"]
        activation = self.gelu(w_in)
        w_out = jnp.einsum(
            "bsd,de->bse", activation, self.params["W_out"]
            ) + self.params["b_out"]
        return w_out

class TransformerBlock:
    def __init__(self, key, d_model, n_head, d_head, d_mlp, ln_eps:float=1e-5, init_range:float=0.02):
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_mlp = d_mlp
        self.ln_eps = ln_eps
        self.init_range = init_range
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.ln1 = LayerNorm(key1, self.d_model, self.ln_eps)
        self.ln2 = LayerNorm(key2, self.d_model, self.ln_eps)
        self.attn = Attention(key3, self.n_head, self.d_head, self.d_model, self.init_range)
        self.mlp = MLP(key4, self.d_model, self.d_mlp)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    

class Unembed:
    def __init__(self, key, d_vocab: int, d_model: int, init_range:float = 0.02):
        self.d_vocab = d_vocab
        self.d_model = d_model
        self.init_range = init_range
        self.params = self.initialize_params(key)
    
    def initialize_params(self, key):
        w_init = jax.nn.initializers.normal(self.init_range)
        return {
            "W_U": w_init(key, (self.d_model, self.d_vocab)),
            "b_U": jax.nn.initializers.zeros(key, self.d_vocab)
        }
    
    def __call__(self, x):
        unembed = jnp.einsum(
            "bse,ev->bsv", x, self.params["W_U"]
        ) + self.params["b_U"]
        return unembed


class TransformerModel:
    def __init__(self, key, cfg):
        self.cfg = cfg
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

        self.embed = Embed(key1, cfg.d_vocab, cfg.d_model, cfg.init_range)
        self.pos_embed = PosEmbed(key2, cfg.n_ctx, cfg.d_model, cfg.init_range)
        self.transformer_blocks = [
            TransformerBlock(
                key3, cfg.d_model, cfg.n_head, cfg.d_head, cfg.d_mlp, cfg.ln_eps, cfg.init_range)
                for _ in range(cfg.n_layers)
                ]
        self.ln = LayerNorm(key4, cfg.d_model, cfg.ln_eps)
        self.unembed = Unembed(key5, cfg.d_vocab, cfg.d_model, cfg.init_range)
    
    def __call__(self, x):
        embed = self.embed(x)
        pos_embed = self.pos_embed(x)
        res = embed + pos_embed

        for block in self.transformer_blocks:
            res = block(res)
        
        logits = self.unembed(self.ln(res))
    
        return logits