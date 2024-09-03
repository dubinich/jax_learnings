import jax
import jax.numpy as jnp

# TODO: add JAX typing


class LayerNorm:
    def __init__(self, key, d_model:int, eps:float=1e-5):
        self.d_model = d_model
        self.eps = eps
        self.params = self.initialize_params(key)

    def initialize_params(self, key):
        w_key, b_key = jax.random.split(key)
        w = jax.nn.initializers.ones(key, self.d_model)
        b = jax.nn.initializers.zeros(key, self.d_model)
        return {"w": w, "b": b}

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(x, axis=-1, keepdims=True)
        std = jnp.sqrt(var + self.eps)

        w,b = self.params["w"], self.params["b"]
        return (x - mean) / std * w + b
    

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
        W_P = self.params["W_P"]
        return W_P[x]


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