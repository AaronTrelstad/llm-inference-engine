use candle_core::{DType, Device, Result, Tensor};

pub struct Block {
    pub block_idx: usize,
    pub n_tokens: usize,
    pub ref_count: usize,
    pub last_access: u64,
}

pub struct BlockPool {
    pub n_blocks:   usize,
    pub block_size: usize,
    pub blocks:     Vec<Block>,
    pub free_list:  Vec<usize>,
    pub k_data:     Vec<Vec<Tensor>>,
    pub v_data:     Vec<Vec<Tensor>>,
}

impl BlockPool {
    pub fn new(
        n_blocks:   usize,
        block_size: usize,
        n_layers:   usize,
        n_kv_heads: usize,
        head_dim:   usize,
        device:     &Device,
    ) -> Result<Self> {
        let mut k_data = Vec::with_capacity(n_blocks);
        let mut v_data = Vec::with_capacity(n_blocks);

        for _ in 0..n_blocks {
            let mut k_layers = Vec::with_capacity(n_layers);
            let mut v_layers = Vec::with_capacity(n_layers);

            for _ in 0..n_layers {
                k_layers.push(Tensor::zeros(
                    (n_kv_heads, block_size, head_dim),
                    DType::F32,
                    device,
                )?);
                v_layers.push(Tensor::zeros(
                    (n_kv_heads, block_size, head_dim),
                    DType::F32,
                    device,
                )?);
            }

            k_data.push(k_layers);
            v_data.push(v_layers);
        }

        let blocks    = (0..n_blocks).map(|i| Block {
            block_idx:   i,
            n_tokens:    0,
            ref_count:   0,
            last_access: 0,
        }).collect();

        let free_list = (0..n_blocks).collect();

        Ok(Self { n_blocks, block_size, blocks, free_list, k_data, v_data })
    }

    pub fn write_k(
        &mut self,
        block_idx: usize,
        layer_idx: usize,
        slot:      usize,
        k_token:   &Tensor, 
    ) -> Result<()> {
        let current = &self.k_data[block_idx][layer_idx];
        let k_token = k_token.unsqueeze(1)?; 

        let new_block = if slot == 0 {
            let zeros = Tensor::zeros(
                (k_token.dim(0)?, self.block_size - 1, k_token.dim(2)?),
                k_token.dtype(),
                k_token.device(),
            )?;
            Tensor::cat(&[&k_token, &zeros], 1)?
        } else {
            let before = current.narrow(1, 0, slot)?;
            let after  = current.narrow(1, slot + 1, self.block_size - slot - 1)
                .unwrap_or(Tensor::zeros((k_token.dim(0)?, 0, k_token.dim(2)?),
                    k_token.dtype(), k_token.device())?);
            Tensor::cat(&[&before, &k_token, &after], 1)?
        };

        self.k_data[block_idx][layer_idx] = new_block;
        Ok(())
    }

    pub fn write_v(
        &mut self,
        block_idx: usize,
        layer_idx: usize,
        slot:      usize,
        v_token:   &Tensor,
    ) -> Result<()> {
        let current = &self.v_data[block_idx][layer_idx];
        let v_token = v_token.unsqueeze(1)?;

        let new_block = if slot == 0 {
            let zeros = Tensor::zeros(
                (v_token.dim(0)?, self.block_size - 1, v_token.dim(2)?),
                v_token.dtype(),
                v_token.device(),
            )?;
            Tensor::cat(&[&v_token, &zeros], 1)?
        } else {
            let before = current.narrow(1, 0, slot)?;
            let after  = current.narrow(1, slot + 1, self.block_size - slot - 1)
                .unwrap_or(Tensor::zeros((v_token.dim(0)?, 0, v_token.dim(2)?),
                    v_token.dtype(), v_token.device())?);
            Tensor::cat(&[&before, &v_token, &after], 1)?
        };

        self.v_data[block_idx][layer_idx] = new_block;
        Ok(())
    }

    pub fn read_k(
        &self,
        block_table: &[usize],
        layer_idx:   usize,
        n_tokens:    usize,
        block_size:  usize,
    ) -> Result<Tensor> {
        let mut chunks = Vec::new();
        let mut remaining = n_tokens;

        for &block_idx in block_table {
            let tokens_in_block = remaining.min(block_size);
            let block_k = self.k_data[block_idx][layer_idx]
                .narrow(1, 0, tokens_in_block)?; 
            chunks.push(block_k);
            remaining -= tokens_in_block;
            if remaining == 0 { break; }
        }

        Tensor::cat(&chunks, 1) 
    }

    pub fn read_v(
        &self,
        block_table: &[usize],
        layer_idx:   usize,
        n_tokens:    usize,
        block_size:  usize,
    ) -> Result<Tensor> {
        let mut chunks = Vec::new();
        let mut remaining = n_tokens;

        for &block_idx in block_table {
            let tokens_in_block = remaining.min(block_size);
            let block_v = self.v_data[block_idx][layer_idx]
                .narrow(1, 0, tokens_in_block)?;
            chunks.push(block_v);
            remaining -= tokens_in_block;
            if remaining == 0 { break; }
        }

        Tensor::cat(&chunks, 1)
    }

    pub fn alloc(&mut self) -> Option<usize> {
        self.free_list.pop()
    }

    pub fn free(&mut self, block_idx: usize) {
        self.blocks[block_idx].n_tokens = 0;
        self.blocks[block_idx].ref_count = 0;
        self.free_list.push(block_idx);
    }

    pub fn n_free(&self) -> usize {
        self.free_list.len()
    }
}

