use crate::block::BlockPool;
use crate::table::BlockTable;
use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;

pub struct KVCacheConfig {
    pub n_blocks: usize,
    pub block_size: usize,
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

pub struct KVCacheManager {
    pool: BlockPool,
    block_tables: HashMap<String, BlockTable>,
    config: KVCacheConfig,
}

impl KVCacheManager {
    pub fn new(config: KVCacheConfig, device: &Device) -> Result<Self> {
        let pool = BlockPool::new(
            config.n_blocks,
            config.block_size,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            device,
        )?;

        Ok(Self {
            pool,
            block_tables: HashMap::new(),
            config,
        })
    }

    pub fn allocate(&mut self, job_id: &str, n_tokens: usize) -> Result<()> {
        let mut table    = BlockTable::new(job_id.to_string(), self.config.block_size);
        let n_blocks     = BlockTable::blocks_needed(n_tokens, self.config.block_size);
    
        for _ in 0..n_blocks {
            let block_idx = match self.pool.alloc() {
                Some(idx) => idx,
                None      => {
                    self.evict_lru()?; 
                    self.pool.alloc()
                        .ok_or_else(|| anyhow::anyhow!("out of KV blocks"))?
                }
            };
            table.append_block(block_idx);
        }
    
        self.block_tables.insert(job_id.to_string(), table);
        Ok(())
    }

    pub fn get_kv(&self, job_id: &str, layer_idx: usize) -> Result<(Tensor, Tensor)> {
        let table = self
            .block_tables
            .get(job_id)
            .ok_or_else(|| anyhow::anyhow!("job not found"))?;

        let k = self.pool.read_k(
            &table.blocks,
            layer_idx,
            table.n_tokens,
            self.config.block_size,
        )?;
        let v = self.pool.read_v(
            &table.blocks,
            layer_idx,
            table.n_tokens,
            self.config.block_size,
        )?;

        Ok((k, v))
    }

    pub fn free(&mut self, job_id: &str) -> Result<()> {
        let table = self
            .block_tables
            .remove(job_id)
            .ok_or_else(|| anyhow::anyhow!("job not found"))?;

        for block_idx in table.blocks {
            self.pool.free(block_idx);
        }

        Ok(())
    }

    pub fn evict_lru(&mut self) -> Result<()> {
        let lru_job = self.block_tables
            .iter()
            .min_by_key(|(_, table)| {
                table.blocks.iter()
                    .map(|&idx| self.pool.blocks[idx].last_access)
                    .min()
                    .unwrap_or(u64::MAX)
            })
            .map(|(id, _)| id.clone())
            .ok_or_else(|| anyhow::anyhow!("no requests to evict"))?;
    
        self.free(&lru_job)
    }

    pub fn append_token(
        &mut self,
        job_id:  &str,
        layer:   usize,
        k_token: &Tensor,
        v_token: &Tensor,
    ) -> Result<()> {
        let needs_block = self.block_tables
            .get(job_id)
            .ok_or_else(|| anyhow::anyhow!("job not found: {}", job_id))?
            .last_block_full();
    
        if needs_block {
            let block_idx = match self.pool.alloc() {
                Some(idx) => idx,
                None      => {
                    self.evict_lru()?;
                    self.pool.alloc()
                        .ok_or_else(|| anyhow::anyhow!("out of KV blocks"))?
                }
            };
            self.block_tables.get_mut(job_id).unwrap().append_block(block_idx);
        }
    
        let (physical_block, slot) = {
            let table = self.block_tables.get(job_id).unwrap();
            let pos   = table.n_tokens;
            (
                table.physical_block(pos)
                    .ok_or_else(|| anyhow::anyhow!("no physical block for pos {}", pos))?,
                table.slot_in_block(pos),
            )
        };
    
        self.pool.write_k(physical_block, layer, slot, k_token)?;
        self.pool.write_v(physical_block, layer, slot, v_token)?;
    
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.pool.blocks[physical_block].last_access = ts;
    
        if layer == self.config.n_layers - 1 {
            self.block_tables
                .get_mut(job_id)
                .unwrap()
                .increment_tokens();
        }
    
        Ok(())
    }
}
