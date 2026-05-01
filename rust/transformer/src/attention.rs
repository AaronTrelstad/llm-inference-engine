use crate::config::LlamaConfig;
use crate::rope::RoPE;
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

pub struct KVCache {
    pub k: Tensor,
    pub v: Tensor,
}

pub struct GroupedQueryAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rope: RoPE,
    n_heads: usize,
    n_kv_heads: usize,
    n_rep: usize,
    head_dim: usize,
    scale: f64,
}

impl GroupedQueryAttention {
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        rope: RoPE,
        config: &LlamaConfig,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            n_rep: config.n_rep,
            head_dim: config.head_dim,
            scale: (config.head_dim as f64).sqrt(),
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KVCache>,
        pos: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.rope.apply(&q, pos)?;
        let k = self.rope.apply(&k, pos)?;

        let (k, v) = match kv_cache {
            None => {
                *kv_cache = Some(KVCache {
                    k: k.clone(),
                    v: v.clone(),
                });
                (k, v)
            }
            Some(cache) => {
                let k = Tensor::cat(&[&cache.k, &k], 2)?;
                let v = Tensor::cat(&[&cache.v, &v], 2)?;
                *cache = KVCache {
                    k: k.clone(),
                    v: v.clone(),
                };
                (k, v)
            }
        };

        let k = Self::repeat_kv(&k, self.n_rep)?;
        let v = Self::repeat_kv(&v, self.n_rep)?;

        let scores = (q.matmul(&k.transpose(2, 3)?)? / self.scale)?;
        let scores = Self::apply_causal_mask(&scores)?;

        let weights = candle_nn::ops::softmax_last_dim(&scores)?;

        let attn_out = weights.matmul(&v)?;

        let attn_out =
            attn_out
                .transpose(1, 2)?
                .reshape((batch, seq_len, self.n_heads * self.head_dim))?;

        self.o_proj.forward(&attn_out)
    }

    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let (batch, n_kv_heads, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((batch, n_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((batch, n_kv_heads * n_rep, seq_len, head_dim))
    }

    fn apply_causal_mask(scores: &Tensor) -> Result<Tensor> {
        let (_, _, q_len, k_len) = scores.dims4()?;
        let device = scores.device();

        let mask: Vec<f32> = (0..q_len)
            .flat_map(|i| (0..k_len).map(move |j| if j <= i { 0.0f32 } else { f32::NEG_INFINITY }))
            .collect();

        let mask = Tensor::from_vec(mask, (q_len, k_len), device)?;
        scores.broadcast_add(&mask)
    }
}
