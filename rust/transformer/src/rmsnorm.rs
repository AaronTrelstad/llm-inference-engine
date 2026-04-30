use candle_core::{Result, Tensor};

pub struct RMSNorm {
    weight: Tensor,
    eps: f64
}

impl RMSNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            eps
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dims().len() - 1;

        let rms = (x.sqr()?
                    .mean_keepdim(last_dim)?
                    + self.eps)?
                    .sqrt()?;

        x.broadcast_div(&rms)?
         .broadcast_mul(&self.weight)
    }
}
