use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::io::{Read, Result, Seek, SeekFrom, Write};
use std::cmp;

pub struct WAL {
    file: File,
    sequence: u64
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub enum WALRecordType {
    Job,
    GpuMetric,
    ChatMessages
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct WALRecord {
    pub sequence: u64,
    pub timestamp: u64,
    pub record_type: WALRecordType,
    pub key: Vec<u8>,
    pub value: Vec<u8>
}

impl WAL {
    pub fn open(path: PathBuf) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;

        let mut wal = Self { file, sequence: 0 };
        wal.recover()?;
        Ok(wal)
    }

    pub fn append(&mut self, record: &WALRecord) -> Result<()> {
        self.sequence += 1;
        let record = WALRecord {
            sequence:    self.sequence,
            timestamp:   record.timestamp,
            record_type: record.record_type.clone(), 
            key:         record.key.clone(),
            value:       record.value.clone(),
        };

        let bytes = bincode::serialize(&record)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let len = bytes.len() as u64;
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&bytes)?;
        self.file.sync_all()?;
        Ok(())
    }

    pub fn recover(&mut self) -> Result<Vec<WALRecord>> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut records = Vec::new();

        loop {
            let mut len_buf = [0u8; 8];
            match self.file.read_exact(&mut len_buf) {
                Ok(_) => {},
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            let len = u64::from_le_bytes(len_buf) as usize;
            
            let mut buf = vec![0u8; len];
            match self.file.read_exact(&mut buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            let record: WALRecord = bincode::deserialize(&buf)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

            self.sequence = self.sequence.max(record.sequence);
            records.push(record);
        }

        Ok(records)
    }

    pub fn truncate(&mut self) -> Result<()> {
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.sync_all()?;
        Ok(())
    }
}
