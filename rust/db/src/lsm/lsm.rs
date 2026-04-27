use std::path::PathBuf;
use crate::lsm::compaction::Compaction;
use crate::wal::wal::{WAL, WALRecordType, WALRecord};
use super::memtable::MemTable;
use super::sstable::SSTable;
use std::io::Result;

pub struct LSMTree {
    wal: WAL,
    memtable: MemTable,
    sstables: Vec<SSTable>,
    data_dir: PathBuf,
}

impl LSMTree {
    const COMPACTION_THRESHOLD: usize = 4; 

    pub fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if let Some(value) = self.memtable.get(key) {
            return Ok(Some(value));
        }

        for table in self.sstables.iter_mut() {
            if let Some(value) = table.get(key)? {
                return Ok(Some(value));
            }
        }
        
        Ok(None)
    }

    pub fn open(data_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&data_dir)?;
        let wal_path = data_dir.join("wal.log");
        let mut wal = WAL::open(wal_path)?;

        let memtable = MemTable::new();
        for record in wal.recover()? {
            memtable.insert(record.key, record.value)?;
        }

        let mut sst_paths: Vec<PathBuf> = std::fs::read_dir(&data_dir)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path  = entry.path();
                if path.extension()?.to_str()? == "sst" {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        sst_paths.sort_by(|a, b| b.cmp(a));

        let sstables = sst_paths
            .into_iter()
            .map(|path| SSTable::open(path))
            .collect::<Result<Vec<SSTable>>>()?;

        Ok(Self {
            wal,
            memtable,
            sstables,
            data_dir,
        })
    }

    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>, record_type: WALRecordType) -> Result<()> {
        if self.memtable.is_full() {
            self.flush()?;
        }
    
        self.wal.append(&WALRecord {
            sequence:    0,
            timestamp:   std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            record_type, 
            key:      key.clone(),
            value:    value.clone()
        })?;
    
        self.memtable.insert(key, value)?;
    
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        let path = self.data_dir.join(format!("sstable_{:03}.sst", self.sstables.len()));
        let new_sstable = SSTable::write(path, &self.memtable)?;
        self.sstables.insert(0, new_sstable);  
        self.memtable = MemTable::new();       
        self.wal.truncate()?;  

        if self.should_compact() {
            let compaction = Compaction::new();
            let old = std::mem::take(&mut self.sstables);
            self.sstables = compaction.compact(old, &self.data_dir)?;
        }               
        Ok(())
    }

    fn should_compact(&self) -> bool {
        self.sstables.len() >= Self::COMPACTION_THRESHOLD
    }
}
