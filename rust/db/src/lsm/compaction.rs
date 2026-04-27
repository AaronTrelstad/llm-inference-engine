use std::collections::BTreeMap;
use std::path::PathBuf;
use std::io::Result;
use crate::lsm::memtable::MemTable;
use crate::wal::wal::WALRecordType;
use super::sstable::SSTable;

pub struct RetentionPolicy {
    pub record_type: WALRecordType,
    pub ttl_seconds: Option<u64>,
}

pub struct Compaction {
    retention_policies: Vec<RetentionPolicy>,
}

impl Compaction {
    pub fn new() -> Self {
        Self {
            retention_policies: vec![
                RetentionPolicy {
                    record_type: WALRecordType::GpuMetric,
                    ttl_seconds: Some(7 * 24 * 3600), 
                },
                RetentionPolicy {
                    record_type: WALRecordType::Job,
                    ttl_seconds: Some(7 * 24 * 3600),                 
                },
                RetentionPolicy {
                    record_type: WALRecordType::ChatMessages,
                    ttl_seconds: Some(7 * 24 * 3600),
                },
            ],
        }
    }

    pub fn compact(
        &self,
        mut sstables: Vec<SSTable>,
        data_dir: &PathBuf,
    ) -> Result<Vec<SSTable>> {
        let mut all_records: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    
        for table in sstables.iter_mut().rev() {
            for (key, value) in table.iter()? {
                all_records.insert(key, value);
            }
        }
    
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    
        let surviving: Vec<(Vec<u8>, Vec<u8>)> = all_records
            .into_iter()
            .filter(|(key, _)| {
                if key.len() < 9 {
                    return false; 
                }
    
                let record_type = match key[0] {
                    0 => WALRecordType::Job,
                    1 => WALRecordType::GpuMetric,
                    2 => WALRecordType::ChatMessages,
                    _ => return false,  
                };
    
                let timestamp = u64::from_be_bytes(key[1..9].try_into().unwrap());
    
                self.should_keep(&record_type, timestamp, now)
            })
            .collect();
    
        let new_path = data_dir.join(format!(
            "sstable_{:03}.sst",
            sstables.len()
        ));
    
        let memtable = MemTable::new();
        for (key, value) in surviving {
            memtable.insert(key, value)?;
        }
        let new_sst = SSTable::write(new_path, &memtable)?;
    
        for table in &sstables {
            if let Ok(_) = std::fs::remove_file(&table.path) {}
        }
    
        Ok(vec![new_sst])
    }
    
    fn should_keep(&self, record_type: &WALRecordType, timestamp: u64, now: u64) -> bool {
        match self.retention_policies.iter().find(|p| {
            std::mem::discriminant(&p.record_type) == std::mem::discriminant(record_type)
        }) {
            Some(policy) => match policy.ttl_seconds {
                Some(ttl) => now - timestamp < ttl,
                None      => true,
            }
            None => true,
        }
    }
}
