use crossbeam_epoch::{Atomic, Owned, Shared};
use std::sync::atomic::Ordering;
use rand;

const MAX_LEVEL: usize = 16;
const PROBABILITY: f64 = 0.5;

pub struct Node {
    key:   Vec<u8>,
    value: Vec<u8>,
    next:  Vec<Atomic<Node>>,
}

impl Node {
    pub fn new(key: Vec<u8>, value: Vec<u8>, level: usize) -> Self {
        Self {
            key,
            value,
            next: (0..level).map(|_| Atomic::null()).collect(),
        }
    }

    fn head() -> Self {
        Self::new(vec![], vec![], MAX_LEVEL)  
    }
}

pub struct SkipList {
    head: Atomic<Node>,
    size: std::sync::atomic::AtomicUsize,
}

impl SkipList {
    pub fn new() -> Self {
        Self {
            head: Atomic::new(Node::head()),
            size: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let guard = crossbeam_epoch::pin();

        unsafe {
            let mut current = self.head.load(Ordering::Acquire, &guard);

            for level in (0..MAX_LEVEL).rev() {
                loop {
                    let next = (*current.deref())
                        .next[level]
                        .load(Ordering::Acquire, &guard);

                    if next.is_null() {
                        break;
                    }

                    match (*next.deref()).key.as_slice().cmp(key) {
                        std::cmp::Ordering::Less => current = next,
                        std::cmp::Ordering::Equal => return Some((*next.deref()).value.clone()),
                        std::cmp::Ordering::Greater => break
                    }
                }
            }
        }

        None
    }

    pub fn insert(&self, key: Vec<u8>, value: Vec<u8>) {
        let guard = crossbeam_epoch::pin();
        let level = Self::random_level();

        unsafe {
            let mut update = vec![Shared::<Node>::null(); MAX_LEVEL];
            let mut current = self.head.load(Ordering::Acquire, &guard);

            for i in (0..MAX_LEVEL).rev() {
                loop {
                    let next = current.deref().next[i].load(Ordering::Acquire, &guard);

                    if next.is_null() {
                        break;
                    }

                    match next.deref().key.as_slice().cmp(key.as_slice()) {
                        std::cmp::Ordering::Less => current = next,
                        std::cmp::Ordering::Equal => return,
                        std::cmp::Ordering::Greater => break
                    }
                }

                update[i] = current;
            }

            let new_node = Owned::new(Node::new(key, value, level)).into_shared(&guard);

            for i in 0..level {
                loop {
                    let next = update[i].deref().next[i].load(Ordering::Acquire, &guard);

                    new_node.deref().next[i].store(next, Ordering::Relaxed);

                    match update[i].deref().next[i].compare_exchange(
                        next, 
                        new_node,
                        Ordering::Release,
                        Ordering::Relaxed,
                        &guard,
                    ) {
                        Ok(_) => break,
                        Err(_) => continue
                    }
                }
            }

        }

        self.size.fetch_add(1, Ordering::Relaxed);
    }

    pub fn scan(&self, start: &[u8], end: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        let guard = crossbeam_epoch::pin();
        let mut results = Vec::new();

        unsafe {
            let mut current = self.head.load(Ordering::Acquire, &guard);

            for level in (0..MAX_LEVEL).rev() {
                loop {
                    let next = current.deref().next[level].load(Ordering::Acquire, &guard);

                    if next.is_null() {
                        break;
                    }

                    match next.deref().key.as_slice().cmp(start) {
                        std::cmp::Ordering::Less => current = next,
                        std::cmp::Ordering::Greater => break,
                        std::cmp::Ordering::Equal => {
                            current = next;
                            break;
                        }
                    }
                }
            }

            loop {
                let next = current.deref().next[0].load(Ordering::Acquire, &guard);

                if next.is_null() {
                    break;
                }

                let node = next.deref();

                if node.key.as_slice() > end {
                    break;
                }

                results.push((node.key.clone(), node.value.clone()));
                current = next;
            }
        }

        results
    }

    fn random_level() -> usize {
        let mut level = 1;
        while rand::random::<f64>() < PROBABILITY && level < MAX_LEVEL {
            level += 1;
        }
        level
    }
}
