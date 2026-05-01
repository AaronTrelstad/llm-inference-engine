use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

pub struct SwiGLU {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGLU {
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        let out = (gate * up)?;
        self.down_proj.forward(&out)
    }
}
