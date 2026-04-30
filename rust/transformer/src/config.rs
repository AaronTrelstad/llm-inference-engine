#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,       
    pub intermediate_size: usize, 
    pub max_seq_len: usize, 
    pub rope_theta: f64,  
    pub rope_scaling_factor: f64,         
    pub rope_low_freq_factor: f64,       
    pub rope_high_freq_factor: f64,      
    pub rope_original_max_seq_len: usize, 
    pub rms_norm_eps: f64,
    pub bos_token_id: u32, 
    pub eos_token_id: u32, 
    pub head_dim: usize, 
    pub n_rep: usize,  
}

impl LlamaConfig {
    pub fn llama3_1_8b() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 4096,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8,
            intermediate_size: 14336,
            max_seq_len: 131072,
            rope_theta: 500000.0,
            rope_scaling_factor: 8.0,
            rope_low_freq_factor: 1.0,
            rope_high_freq_factor: 4.0,
            rope_original_max_seq_len: 8192,
            rms_norm_eps: 1e-5,
            bos_token_id: 128000,
            eos_token_id: 128001,
            head_dim: 4096 / 32, 
            n_rep: 32 / 8,      
        }
    }

    pub fn from_json(path: &std::path::Path) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&s)?;

        let hidden_size = v["hidden_size"].as_u64().unwrap() as usize;
        let n_heads = v["num_attention_heads"].as_u64().unwrap() as usize;
        let n_kv_heads = v["num_key_value_heads"].as_u64().unwrap() as usize;
        let scaling = &v["rope_scaling"];

        Ok(Self {
            vocab_size: v["vocab_size"].as_u64().unwrap() as usize,
            hidden_size,
            n_layers: v["num_hidden_layers"].as_u64().unwrap() as usize,
            n_heads,
            n_kv_heads,
            intermediate_size: v["intermediate_size"].as_u64().unwrap() as usize,
            max_seq_len: v["max_position_embeddings"].as_u64().unwrap() as usize,
            rope_theta: v["rope_theta"].as_f64().unwrap(),
            rope_scaling_factor: scaling["factor"].as_f64().unwrap_or(1.0),
            rope_low_freq_factor: scaling["low_freq_factor"].as_f64().unwrap_or(1.0),
            rope_high_freq_factor: scaling["high_freq_factor"].as_f64().unwrap_or(4.0),
            rope_original_max_seq_len: scaling["original_max_position_embeddings"]
                .as_u64()
                .unwrap_or(8192) as usize,
            rms_norm_eps: v["rms_norm_eps"].as_f64().unwrap(),
            bos_token_id: v["bos_token_id"].as_u64().unwrap() as u32,
            eos_token_id: v["eos_token_id"].as_u64().unwrap() as u32,
            head_dim: hidden_size / n_heads,
            n_rep: n_heads / n_kv_heads,
        })
    }
}
