use transformer::block::LlamaBlock;
use transformer::attention::{GroupedQueryAttention};
use transformer::rmsnorm::RMSNorm;
use transformer::ffn::SwiGLU;
use transformer::rope::RoPE;
use transformer::config::LlamaConfig;
use candle_core::{Device, Tensor};
use candle_nn::Linear;

fn small_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size:                 128256,
        hidden_size:                256,
        n_layers:                   2,
        n_heads:                    4,
        n_kv_heads:                 2,
        intermediate_size:          512,
        max_seq_len:                128,
        rope_theta:                 500000.0,
        rope_scaling_factor:        8.0,
        rope_low_freq_factor:       1.0,
        rope_high_freq_factor:      4.0,
        rope_original_max_seq_len:  8192,
        rms_norm_eps:               1e-5,
        bos_token_id:               128000,
        eos_token_id:               128001,
        head_dim:                   256 / 4,
        n_rep:                      4 / 2,
    }
}

fn make_block(config: &LlamaConfig, device: &Device) -> LlamaBlock {
    let h   = config.hidden_size;
    let qd  = config.n_heads    * config.head_dim;
    let kd  = config.n_kv_heads * config.head_dim;
    let int = config.intermediate_size;

    let attn_norm = RMSNorm::new(
        Tensor::ones(h, candle_core::DType::F32, device).unwrap(),
        config.rms_norm_eps,
    );
    let ffn_norm = RMSNorm::new(
        Tensor::ones(h, candle_core::DType::F32, device).unwrap(),
        config.rms_norm_eps,
    );

    let q_proj = Linear::new(Tensor::randn(0f32, 0.02, (qd, h),  device).unwrap(), None);
    let k_proj = Linear::new(Tensor::randn(0f32, 0.02, (kd, h),  device).unwrap(), None);
    let v_proj = Linear::new(Tensor::randn(0f32, 0.02, (kd, h),  device).unwrap(), None);
    let o_proj = Linear::new(Tensor::randn(0f32, 0.02, (h,  qd), device).unwrap(), None);
    let rope   = RoPE::new(config, device).unwrap();
    let attn   = GroupedQueryAttention::new(q_proj, k_proj, v_proj, o_proj, rope, config);

    let gate_proj = Linear::new(Tensor::randn(0f32, 0.02, (int, h),   device).unwrap(), None);
    let up_proj   = Linear::new(Tensor::randn(0f32, 0.02, (int, h),   device).unwrap(), None);
    let down_proj = Linear::new(Tensor::randn(0f32, 0.02, (h,   int), device).unwrap(), None);
    let ffn       = SwiGLU::new(gate_proj, up_proj, down_proj);

    LlamaBlock::new(attn_norm, attn, ffn_norm, ffn)
}

#[test]
fn test_block_output_shape() {
    let device = Device::Cpu;
    let config = small_config();
    let block  = make_block(&config, &device);

    let x   = Tensor::randn(0f32, 1f32, (1, 4, config.hidden_size), &device).unwrap();
    let out = block.forward(&x, &mut None, 0).unwrap();

    assert_eq!(out.dims(), x.dims(), "block should preserve input shape");
}

#[test]
fn test_block_output_finite() {
    let device = Device::Cpu;
    let config = small_config();
    let block  = make_block(&config, &device);

    let x    = Tensor::randn(0f32, 1f32, (1, 4, config.hidden_size), &device).unwrap();
    let out  = block.forward(&x, &mut None, 0).unwrap();
    let data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert!(data.iter().all(|v| v.is_finite()), "output contains NaN or inf");
}

#[test]
fn test_block_kv_cache_populated() {
    let device = Device::Cpu;
    let config = small_config();
    let block  = make_block(&config, &device);

    let x         = Tensor::randn(0f32, 1f32, (1, 4, config.hidden_size), &device).unwrap();
    let mut cache = None;
    block.forward(&x, &mut cache, 0).unwrap();

    assert!(cache.is_some(), "KV cache should be populated after forward");
    assert_eq!(cache.unwrap().k.dim(2).unwrap(), 4, "KV cache should have 4 entries");
}

#[test]
fn test_block_decode_step() {
    let device = Device::Cpu;
    let config = small_config();
    let block  = make_block(&config, &device);

    let x_prefill = Tensor::randn(0f32, 1f32, (1, 4, config.hidden_size), &device).unwrap();
    let mut cache = None;
    block.forward(&x_prefill, &mut cache, 0).unwrap();

    let x_decode = Tensor::randn(0f32, 1f32, (1, 1, config.hidden_size), &device).unwrap();
    let out      = block.forward(&x_decode, &mut cache, 4).unwrap();

    assert_eq!(out.dims(), &[1, 1, config.hidden_size]);
    assert_eq!(cache.unwrap().k.dim(2).unwrap(), 5, "KV cache should grow to 5");
}

#[test]
fn test_block_residual_connection() {
    let device = Device::Cpu;
    let config = small_config();
    let block  = make_block(&config, &device);

    let x   = Tensor::zeros((1, 1, config.hidden_size), candle_core::DType::F32, &device).unwrap();
    let out = block.forward(&x, &mut None, 0).unwrap();

    let out_data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let max_val  = out_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    assert!(max_val.abs() < 1.0, "zero input should give near-zero output, got max={}", max_val);
}
