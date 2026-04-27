use crate::btree::btree::BTree;
use crate::lsm::lsm::LSMTree;
use crate::wal::wal::WALRecordType;
use std::io::Result;
use std::path::PathBuf;
use std::collections::HashMap;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct JobRecord {
    pub job_id:            String,
    pub status:            String,
    pub prompt:            String,
    pub output:            String,
    pub prefill_worker_id: Option<String>,
    pub decode_worker_id:  Option<String>,
    pub latency:           u64,
    pub timestamp:         u64,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct MetricRecord {
    pub worker_id:      String,
    pub gpu_memory:     f32,
    pub gpu_util:       f32,
    pub active_jobs:    u32,
    pub tokens_per_sec: u32,
    pub timestamp:      u64,
}

pub struct JobFilter {
    pub status:      Option<String>,
    pub worker_id:   Option<String>,
    pub time_range:  Option<(u64, u64)>,
    pub aggregate:   bool,
}

pub struct MetricFilter {
    pub worker_id:   Option<String>,
    pub time_range:  Option<(u64, u64)>,
    pub aggregation: Option<GPUAggregation>,
}

pub enum GPUAggregation { Avg, Max, Min, P99 }

pub struct JobAggregateResult {
    pub avg_latency:     f64,
    pub p99_latency:     f64,
    pub max_latency:     u64,
    pub min_latency:     u64,
    pub total_count:     usize,
    pub count_by_status: HashMap<String, usize>,
}

pub enum JobQueryResult {
    Records(Vec<JobRecord>),
    Aggregate(JobAggregateResult),
}

pub struct Series {
    lsm:                 LSMTree,
    inference_job_btree: BTree,
    gpu_metrics_btree:   BTree,
}

impl Series {
    pub fn new(data_dir: PathBuf) -> Result<Self> {
        Ok(Self {
            lsm:                 LSMTree::open(data_dir)?,
            inference_job_btree: BTree::new(4),
            gpu_metrics_btree:   BTree::new(4),
        })
    }

    pub fn insert_job(&mut self, job: JobRecord) -> Result<()> {
        let key = job.job_id.as_bytes().to_vec();
        let value = bincode::serialize(&job)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        self.lsm.put(key, value, WALRecordType::InferenceJob)?;
        self.inference_job_btree.insert(job.timestamp, job.job_id.clone());

        Ok(())
    }

    pub fn insert_metrics(&mut self, metrics: MetricRecord) -> Result<()> {
        let mut key = metrics.worker_id.as_bytes().to_vec();
        key.extend_from_slice(&metrics.timestamp.to_be_bytes());

        let value = bincode::serialize(&metrics)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        self.lsm.put(key, value, WALRecordType::GPUMetric)?;
        let btree_value = format!("{}:{}", metrics.worker_id, metrics.timestamp);
        self.gpu_metrics_btree.insert(metrics.timestamp, btree_value);

        Ok(())
    }

    pub fn get_job(&mut self, job_id: &str) -> Result<Option<JobRecord>> {
        let key = job_id.as_bytes().to_vec();

        match self.lsm.get(&key)? {
            Some(bytes) => {
                let job: JobRecord = bincode::deserialize(&bytes)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                Ok(Some(job))
            }
            None => Ok(None),
        }
    }

    pub fn query_jobs(&mut self, filter: JobFilter) -> Result<JobQueryResult> {
        let (start, end) = filter.time_range.unwrap_or((0, u64::MAX));
        let job_ids = self.inference_job_btree.range(start, end);
        let mut jobs = Vec::new();

        for job_id in job_ids {
            let key = job_id.as_bytes().to_vec();
            match self.lsm.get(&key)? {
                Some(bytes) => {
                    let job: JobRecord = bincode::deserialize(&bytes)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

                    if let Some(ref status) = filter.status {
                        if &job.status != status { continue; }
                    }
                    if let Some(ref worker_id) = filter.worker_id {
                        if job.prefill_worker_id.as_deref() != Some(worker_id)
                            && job.decode_worker_id.as_deref() != Some(worker_id) {
                            continue;
                        }
                    }

                    jobs.push(job);
                }
                None => continue,
            }
        }

        match filter.aggregate {
            true  => Ok(JobQueryResult::Aggregate(Self::aggregate_inference_jobs(jobs))),
            false => Ok(JobQueryResult::Records(jobs)),
        }
    }

    pub fn query_metrics(&mut self, filter: MetricFilter) -> Result<Vec<MetricRecord>> {
        let (start, end) = filter.time_range.unwrap_or((0, u64::MAX));
        let entries = self.gpu_metrics_btree.range(start, end);
        let mut metrics = Vec::new();

        for entry in entries {
            let parts: Vec<&str> = entry.splitn(2, ':').collect();
            if parts.len() != 2 { continue; }
            let worker_id = parts[0];
            let timestamp: u64 = parts[1].parse().unwrap_or(0);

            let mut key = worker_id.as_bytes().to_vec();
            key.extend_from_slice(&timestamp.to_be_bytes());

            match self.lsm.get(&key)? {
                Some(bytes) => {
                    let metric: MetricRecord = bincode::deserialize(&bytes)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

                    if let Some(ref wid) = filter.worker_id {
                        if &metric.worker_id != wid { continue; }
                    }

                    metrics.push(metric);
                }
                None => continue,
            }
        }

        match filter.aggregation {
            Some(agg) => Ok(vec![Self::aggregate_gpu_metrics(metrics, agg)]),
            None      => Ok(metrics),
        }
    }

    fn aggregate_inference_jobs(jobs: Vec<JobRecord>) -> JobAggregateResult {
        let mut latencies: Vec<u64> = jobs.iter().map(|j| j.latency).collect();
        latencies.sort();

        let mut count_by_status = HashMap::new();
        for job in &jobs {
            *count_by_status.entry(job.status.clone()).or_insert(0) += 1;
        }

        JobAggregateResult {
            avg_latency:     latencies.iter().sum::<u64>() as f64 / latencies.len() as f64,
            p99_latency:     latencies[(latencies.len() as f64 * 0.99) as usize] as f64,
            max_latency:     *latencies.last().unwrap_or(&0),
            min_latency:     *latencies.first().unwrap_or(&0),
            total_count:     jobs.len(),
            count_by_status,
        }
    }

    fn aggregate_gpu_metrics(metrics: Vec<MetricRecord>, agg: GPUAggregation) -> MetricRecord {
        match agg {
            GPUAggregation::Avg => {
                let len = metrics.len() as f32;
                let gpu_memory     = metrics.iter().map(|m| m.gpu_memory).sum::<f32>() / len;
                let gpu_util       = metrics.iter().map(|m| m.gpu_util).sum::<f32>() / len;
                let active_jobs    = (metrics.iter().map(|m| m.active_jobs as f32).sum::<f32>() / len) as u32;
                let tokens_per_sec = (metrics.iter().map(|m| m.tokens_per_sec as f32).sum::<f32>() / len) as u32;

                MetricRecord {
                    worker_id:      "aggregate".to_string(),
                    timestamp:      metrics.last().map(|m| m.timestamp).unwrap_or(0),
                    gpu_memory,
                    gpu_util,
                    active_jobs,
                    tokens_per_sec,
                }
            }

            GPUAggregation::Max => MetricRecord {
                worker_id:      "aggregate".to_string(),
                timestamp:      metrics.last().map(|m| m.timestamp).unwrap_or(0),
                gpu_memory:     metrics.iter().map(|m| m.gpu_memory).fold(f32::MIN, f32::max),
                gpu_util:       metrics.iter().map(|m| m.gpu_util).fold(f32::MIN, f32::max),
                active_jobs:    metrics.iter().map(|m| m.active_jobs).max().unwrap_or(0),
                tokens_per_sec: metrics.iter().map(|m| m.tokens_per_sec).max().unwrap_or(0),
            },

            GPUAggregation::Min => MetricRecord {
                worker_id:      "aggregate".to_string(),
                timestamp:      metrics.last().map(|m| m.timestamp).unwrap_or(0),
                gpu_memory:     metrics.iter().map(|m| m.gpu_memory).fold(f32::MAX, f32::min),
                gpu_util:       metrics.iter().map(|m| m.gpu_util).fold(f32::MAX, f32::min),
                active_jobs:    metrics.iter().map(|m| m.active_jobs).min().unwrap_or(0),
                tokens_per_sec: metrics.iter().map(|m| m.tokens_per_sec).min().unwrap_or(0),
            },

            GPUAggregation::P99 => {
                let mut gpu_utils: Vec<f32> = metrics.iter().map(|m| m.gpu_util).collect();
                gpu_utils.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p99_util = gpu_utils[(gpu_utils.len() as f32 * 0.99) as usize];

                let mut gpu_mems: Vec<f32> = metrics.iter().map(|m| m.gpu_memory).collect();
                gpu_mems.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p99_mem = gpu_mems[(gpu_mems.len() as f32 * 0.99) as usize];

                MetricRecord {
                    worker_id:      "aggregate".to_string(),
                    timestamp:      metrics.last().map(|m| m.timestamp).unwrap_or(0),
                    gpu_memory:     p99_mem,
                    gpu_util:       p99_util,
                    active_jobs:    0,
                    tokens_per_sec: 0,
                }
            }
        }
    }
}
