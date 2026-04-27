use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crate::skiplist::skiplist::SkipList;
use std::io::Result;

const MAX_SIZE: usize = 4 * 1024 * 1024;

pub struct MemTable {
    data: SkipList,
    size: AtomicUsize,
    frozen: AtomicBool
}

impl MemTable {
    pub fn new() -> Self {
        Self {
            data: SkipList::new(),
            size: AtomicUsize::new(0),
            frozen: AtomicBool::new(false),
        }
    }

    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.data.get(key)
    }

    pub fn insert(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        if self.frozen.load(Ordering::Acquire) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "memtable is frozen"
            ));
        }

        let entry_size = key.len() + value.len();
        self.data.insert(key, value);
        self.size.fetch_add(entry_size, Ordering::Relaxed);

        Ok(())
    }
    
    pub fn scan(&self, start: &[u8], end: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.data.scan(start, end)
    }

    pub fn is_full(&self) -> bool {
        self.size.load(Ordering::Relaxed) >= MAX_SIZE
    }

    pub fn freeze(&self) {
        self.frozen.store(true, Ordering::Release);
    }

    pub fn size(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    pub fn iter(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.data.scan(&[], &vec![0xFF; 32])
    }
}
