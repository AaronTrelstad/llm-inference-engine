use crate::attention::{GroupedQueryAttention, KVCache};
use crate::ffn::SwiGLU;
use crate::rmsnorm::RMSNorm;
use candle_core::{Result, Tensor};

pub struct LlamaBlock {
    attn_norm: RMSNorm,
    attention: GroupedQueryAttention,
    ffn_norm: RMSNorm,
    ffn: SwiGLU,
}

impl LlamaBlock {
    pub fn new(
        attn_norm: RMSNorm,
        attention: GroupedQueryAttention,
        ffn_norm: RMSNorm,
        ffn: SwiGLU,
    ) -> Self {
        Self {
            attn_norm,
            attention,
            ffn_norm,
            ffn,
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: &mut Option<KVCache>,
        pos: usize,
    ) -> Result<Tensor> {
        let attn_out = self
            .attention
            .forward(&self.attn_norm.forward(x)?, kv_cache, pos)?;
        let x = (x + &attn_out)?;

        let ffn_out = self.ffn.forward(&self.ffn_norm.forward(&x)?)?;
        let x = (&x + &ffn_out)?;

        Ok(x)
    }
}
