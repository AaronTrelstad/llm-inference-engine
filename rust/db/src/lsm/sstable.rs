use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Result, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use super::bloom::BloomFilter;
use super::memtable::MemTable;

pub struct SSTable {
    path: PathBuf,
    file: File,
    index: BTreeMap<Vec<u8>, u64>,
    bloom_filter: BloomFilter,
}

impl SSTable {
    pub fn write(path: PathBuf, memtable: &MemTable) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&path)?;

        let mut index = BTreeMap::new();
        let mut bloom_filter = BloomFilter::new(1_000_000, 0.01);
    
        {
            let mut writer = std::io::BufWriter::new(&file);        
            let mut offset = 0;

            for (key, value) in memtable.iter() {
                index.insert(key.clone(), offset);
                bloom_filter.insert(&key);

                let key_len = key.len() as u64;
                writer.write_all(&key_len.to_le_bytes())?;
                writer.write_all(&key)?;

                let value_len = value.len() as u64;
                writer.write_all(&value_len.to_le_bytes())?;
                writer.write_all(&value)?;

                offset += 8 + key.len() as u64 + 8 + value.len() as u64;
            }

            let index_offset = offset;
            let index_bytes = bincode::serialize(&index)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            writer.write_all(&(index_bytes.len() as u64).to_le_bytes())?;
            writer.write_all(&index_bytes)?;
            offset += 8 + index_bytes.len() as u64;

            let bloom_offset = offset; 
            let bloom_bytes = bincode::serialize(&bloom_filter)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            writer.write_all(&(bloom_bytes.len() as u64).to_le_bytes())?;
            writer.write_all(&bloom_bytes)?;

            writer.write_all(&index_offset.to_le_bytes())?; 
            writer.write_all(&bloom_offset.to_le_bytes())?;  
            writer.write_all(&(memtable.size() as u64).to_le_bytes())?;  

            writer.flush()?;
        }

        Ok(Self {
            path,
            file,
            index,
            bloom_filter,
        })
    }

    pub fn open(path: PathBuf) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .open(&path)?;

        file.seek(SeekFrom::End(-24))?;
        let mut footer = [0u8; 24];
        file.read_exact(&mut footer)?;

        let index_offset = u64::from_le_bytes(footer[0..8].try_into().unwrap());
        let bloom_offset = u64::from_le_bytes(footer[8..16].try_into().unwrap());

        file.seek(SeekFrom::Start(index_offset))?;
        let mut index_len_buf = [0u8; 8];
        file.read_exact(&mut index_len_buf)?;
        let index_len = u64::from_le_bytes(index_len_buf) as usize;
        let mut index_buf = vec![0u8; index_len];
        file.read_exact(&mut index_buf)?;
        let index: BTreeMap<Vec<u8>, u64> = bincode::deserialize(&index_buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        file.seek(SeekFrom::Start(bloom_offset))?;
        let mut bloom_len_buf = [0u8; 8];
        file.read_exact(&mut bloom_len_buf)?;
        let bloom_len = u64::from_le_bytes(bloom_len_buf) as usize;
        let mut bloom_buf = vec![0u8; bloom_len];
        file.read_exact(&mut bloom_buf)?;
        let bloom_filter: BloomFilter = bincode::deserialize(&bloom_buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        Ok(Self {
            path,
            file,
            index,
            bloom_filter,
        })
    }

    pub fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if !self.bloom_filter.contains(key) {
            return Ok(None);
        }
    
        let offset = match self.index.get(key) {
            Some(o) => *o,
            None    => return Ok(None), 
        };
    
        self.file.seek(SeekFrom::Start(offset))?;
    
        let mut key_len_buf = [0u8; 8];
        self.file.read_exact(&mut key_len_buf)?;
        let key_len = u64::from_le_bytes(key_len_buf) as usize;
        self.file.seek(SeekFrom::Current(key_len as i64))?;
    
        let mut val_len_buf = [0u8; 8];
        self.file.read_exact(&mut val_len_buf)?;
        let val_len = u64::from_le_bytes(val_len_buf) as usize;
        let mut value = vec![0u8; val_len];
        self.file.read_exact(&mut value)?;
    
        Ok(Some(value))
    }
}


