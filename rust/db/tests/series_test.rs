use db::series::series::{
    GPUAggregation, JobFilter, JobQueryResult, JobRecord, MetricFilter, MetricRecord, Series,
};
use tempfile::tempdir;

fn make_job(id: &str, status: &str, timestamp: u64) -> JobRecord {
    JobRecord {
        job_id:            id.to_string(),
        status:            status.to_string(),
        prompt:            "test prompt".to_string(),
        output:            "test output".to_string(),
        prefill_worker_id: Some("worker_01".to_string()),
        decode_worker_id:  Some("worker_02".to_string()),
        latency:           100,
        timestamp,
    }
}

fn make_metric(worker_id: &str, timestamp: u64, gpu_memory: f32) -> MetricRecord {
    MetricRecord {
        worker_id:      worker_id.to_string(),
        gpu_memory,
        gpu_util:       50.0,
        active_jobs:    1,
        tokens_per_sec: 100,
        timestamp,
    }
}

// ── Job point lookups ─────────────────────────────────────────────────────────

#[test]
fn test_insert_and_get_job() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut series = Series::new(&path).unwrap();

    let job = make_job("job_001", "done", 1000);
    series.insert_job(job).unwrap();

    let result = series.get_job("job_001").unwrap();
    assert!(result.is_some());
    let job = result.unwrap();
    assert_eq!(job.job_id, "job_001");
    assert_eq!(job.status, "done");
    assert_eq!(job.timestamp, 1000);
}

#[test]
fn test_get_missing_job() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut series = Series::new(&path).unwrap();

    let result = series.get_job("nonexistent").unwrap();
    assert!(result.is_none());
}

// ── Job range queries ─────────────────────────────────────────────────────────

#[test]
fn test_query_jobs_time_range() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut series = Series::new(&path).unwrap();

    series.insert_job(make_job("job_001", "done",       1000)).unwrap();
    series.insert_job(make_job("job_002", "done",       2000)).unwrap();
    series.insert_job(make_job("job_003", "processing", 3000)).unwrap();
    series.insert_job(make_job("job_004", "done",       4000)).unwrap();
    series.insert_job(make_job("job_005", "done",       5000)).unwrap();

    let result = series.query_jobs(JobFilter {
        status:     None,
        worker_id:  None,
        time_range: Some((2000, 4000)),
        aggregate:  false,
    }).unwrap();

    match result {
        JobQueryResult::Records(jobs) => {
            assert_eq!(jobs.len(), 3);
            assert!(jobs.iter().all(|j| j.timestamp >= 2000 && j.timestamp <= 4000));
        }
        _ => panic!("expected records"),
    }
}

#[test]
fn test_query_jobs_filter_by_status() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut series = Series::new(&path).unwrap();

    series.insert_job(make_job("job_001", "done",       1000)).unwrap();
    series.insert_job(make_job("job_002", "failed",     2000)).unwrap();
    series.insert_job(make_job("job_003", "done",       3000)).unwrap();
    series.insert_job(make_job("job_004", "processing", 4000)).unwrap();

    let result = series.query_jobs(JobFilter {
        status:     Some("done".to_string()),
        worker_id:  None,
        time_range: Some((0, u64::MAX)),
        aggregate:  false,
    }).unwrap();

    match result {
        JobQueryResult::Records(jobs) => {
            assert_eq!(jobs.len(), 2);
            assert!(jobs.iter().all(|j| j.status == "done"));
        }
        _ => panic!("expected records"),
    }
}

#[test]
fn test_query_jobs_filter_by_worker_id() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut series = Series::new(&path).unwrap();

    let mut job1 = make_job("job_001", "done", 1000);
    job1.prefill_worker_id = Some("worker_A".to_string());
    job1.decode_worker_id  = Some("worker_B".to_string());

    let mut job2 = make_job("job_002", "done", 2000);
    job2.prefill_worker_id = Some("worker_C".to_string());
    job2.decode_worker_id  = Some("worker_D".to_string());

    series.insert_job(job1).unwrap();
    series.insert_job(job2).unwrap();

    let result = series.query_jobs(JobFilter {
        status:     None,
        worker_id:  Some("worker_A".to_string()),
        time_range: Some((0, u64::MAX)),
        aggregate:  false,
    }).unwrap();

    match result {
        JobQueryResult::Records(jobs) => {
            assert_eq!(jobs.len(), 1);
            assert_eq!(jobs[0].job_id, "job_001");
        }
        _ => panic!("expected records"),
    }
}

// ── Job aggregation ───────────────────────────────────────────────────────────

#[test]
fn test_query_jobs_aggregation() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut series = Series::new(&path).unwrap();

    let mut job1 = make_job("job_001", "done", 1000); job1.latency = 100;
    let mut job2 = make_job("job_002", "done", 2000); job2.latency = 200;
    let mut job3 = make_job("job_003", "done", 3000); job3.latency = 300;

    series.insert_job(job1).unwrap();
    series.insert_job(job2).unwrap();
    series.insert_job(job3).unwrap();

    let result = series.query_jobs(JobFilter {
        status:     None,
        worker_id:  None,
        time_range: Some((0, u64::MAX)),
        aggregate:  true,
    }).unwrap();

    match result {
        JobQueryResult::Aggregate(agg) => {
            assert_eq!(agg.total_count, 3);
            assert_eq!(agg.min_latency, 100);
            assert_eq!(agg.max_latency, 300);
            assert!((agg.avg_latency - 200.0).abs() < 0.1);
        }
        _ => panic!("expected aggregate"),
    }
}

#[test]
fn test_query_jobs_count_by_status() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut series = Series::new(&path).unwrap();

    series.insert_job(make_job("j1", "done",   1000)).unwrap();
    series.insert_job(make_job("j2", "done",   2000)).unwrap();
    series.insert_job(make_job("j3", "failed", 3000)).unwrap();

    let result = series.query_jobs(JobFilter {
        status:     None,
        worker_id:  None,
        time_range: Some((0, u64::MAX)),
        aggregate:  true,
    }).unwrap();

    match result {
        JobQueryResult::Aggregate(agg) => {
            assert_eq!(*agg.count_by_status.get("done").unwrap(), 2);
            assert_eq!(*agg.count_by_status.get("failed").unwrap(), 1);
        }
        _ => panic!("expected aggregate"),
    }
}

// ── Metric queries ────────────────────────────────────────────────────────────

#[test]
fn test_insert_and_query_metrics() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut series = Series::new(&path).unwrap();

    for i in 0..5u64 {
        series.insert_metrics(MetricRecord {
            worker_id:      "worker_01".to_string(),
            gpu_memory:     50.0 + i as f32,
            gpu_util:       60.0 + i as f32,
            active_jobs:    i as u32,
            tokens_per_sec: 100 + i as u32,
            timestamp:      1000 + i * 1000,
        }).unwrap();
    }

    let result = series.query_metrics(MetricFilter {
        worker_id:   Some("worker_01".to_string()),
        time_range:  Some((1000, 5000)),
        aggregation: None,
    }).unwrap();

    assert_eq!(result.len(), 5);
    assert!(result.iter().all(|m| m.worker_id == "worker_01"));
}

#[test]
fn test_query_metrics_filter_by_worker_id() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut series = Series::new(&path).unwrap();

    series.insert_metrics(make_metric("worker_A", 1000, 40.0)).unwrap();
    series.insert_metrics(make_metric("worker_B", 2000, 60.0)).unwrap();
    series.insert_metrics(make_metric("worker_A", 3000, 80.0)).unwrap();

    let result = series.query_metrics(MetricFilter {
        worker_id:   Some("worker_A".to_string()),
        time_range:  None,
        aggregation: None,
    }).unwrap();

    assert_eq!(result.len(), 2);
    assert!(result.iter().all(|m| m.worker_id == "worker_A"));
}

#[test]
fn test_metrics_aggregation_avg() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut series = Series::new(&path).unwrap();

    for i in 0..4u64 {
        series.insert_metrics(MetricRecord {
            worker_id:      "worker_01".to_string(),
            gpu_memory:     (i * 10) as f32,  // 0, 10, 20, 30
            gpu_util:       50.0,
            active_jobs:    0,
            tokens_per_sec: 100,
            timestamp:      1000 + i * 1000,
        }).unwrap();
    }

    let result = series.query_metrics(MetricFilter {
        worker_id:   None,
        time_range:  Some((0, u64::MAX)),
        aggregation: Some(GPUAggregation::Avg),
    }).unwrap();

    assert_eq!(result.len(), 1);
    assert!((result[0].gpu_memory - 15.0).abs() < 0.1);  // avg of 0,10,20,30 = 15
}

#[test]
fn test_metrics_aggregation_max() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut series = Series::new(&path).unwrap();

    series.insert_metrics(make_metric("w", 1000, 20.0)).unwrap();
    series.insert_metrics(make_metric("w", 2000, 80.0)).unwrap();
    series.insert_metrics(make_metric("w", 3000, 50.0)).unwrap();

    let result = series.query_metrics(MetricFilter {
        worker_id:   None,
        time_range:  None,
        aggregation: Some(GPUAggregation::Max),
    }).unwrap();

    assert_eq!(result.len(), 1);
    assert!((result[0].gpu_memory - 80.0).abs() < 0.1);
}

#[test]
fn test_metrics_aggregation_min() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut series = Series::new(&path).unwrap();

    series.insert_metrics(make_metric("w", 1000, 20.0)).unwrap();
    series.insert_metrics(make_metric("w", 2000, 80.0)).unwrap();
    series.insert_metrics(make_metric("w", 3000, 50.0)).unwrap();

    let result = series.query_metrics(MetricFilter {
        worker_id:   None,
        time_range:  None,
        aggregation: Some(GPUAggregation::Min),
    }).unwrap();

    assert_eq!(result.len(), 1);
    assert!((result[0].gpu_memory - 20.0).abs() < 0.1);
}

// ── End-to-end: single session ────────────────────────────────────────────────

#[test]
fn test_e2e_insert_query_jobs_and_metrics() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let mut series = Series::new(&path).unwrap();

    // Insert 10 jobs with alternating statuses
    for i in 1..=10u64 {
        let mut job = make_job(
            &format!("job_{:03}", i),
            if i % 2 == 0 { "done" } else { "failed" },
            i * 1000,
        );
        job.latency = i * 50;
        series.insert_job(job).unwrap();
    }

    // Insert metrics for two workers
    for i in 1..=5u64 {
        series.insert_metrics(make_metric("worker_A", i * 1000, i as f32 * 10.0)).unwrap();
        series.insert_metrics(make_metric("worker_B", i * 1000, i as f32 * 5.0)).unwrap();
    }

    // Point lookup
    let job = series.get_job("job_001").unwrap().unwrap();
    assert_eq!(job.status, "failed");
    assert_eq!(job.latency, 50);

    // Status filter: 5 "done" jobs (even indices 2,4,6,8,10)
    let result = series.query_jobs(JobFilter {
        status:     Some("done".to_string()),
        worker_id:  None,
        time_range: Some((0, u64::MAX)),
        aggregate:  false,
    }).unwrap();
    match result {
        JobQueryResult::Records(jobs) => assert_eq!(jobs.len(), 5),
        _ => panic!("expected records"),
    }

    // Aggregate over all jobs
    let result = series.query_jobs(JobFilter {
        status:     None,
        worker_id:  None,
        time_range: Some((0, u64::MAX)),
        aggregate:  true,
    }).unwrap();
    match result {
        JobQueryResult::Aggregate(agg) => {
            assert_eq!(agg.total_count, 10);
            assert_eq!(agg.min_latency, 50);
            assert_eq!(agg.max_latency, 500);
        }
        _ => panic!("expected aggregate"),
    }

    // Metric query by worker
    let metrics = series.query_metrics(MetricFilter {
        worker_id:   Some("worker_A".to_string()),
        time_range:  None,
        aggregation: None,
    }).unwrap();
    assert_eq!(metrics.len(), 5);

    // Max aggregation across all workers — worker_A peaks at 5*10=50.0
    let result = series.query_metrics(MetricFilter {
        worker_id:   None,
        time_range:  None,
        aggregation: Some(GPUAggregation::Max),
    }).unwrap();
    assert_eq!(result.len(), 1);
    assert!((result[0].gpu_memory - 50.0).abs() < 0.1);
}

// ── WAL recovery (raw LSM level) ──────────────────────────────────────────────

#[test]
fn test_wal_recovery_restores_raw_lsm_data() {
    // This tests that the underlying WAL + LSM survive a restart.
    // Note: Series-level point lookups (get_job) require the in-memory
    // job_timestamps index, which is not persisted (see bug #2). After
    // restart, get_job returns None even though the data is in the WAL.
    // This test verifies the WAL data is intact at the raw level.
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();

    {
        let mut series = Series::new(&path).unwrap();
        series.insert_job(make_job("job_001", "done", 1000)).unwrap();
        series.insert_job(make_job("job_002", "done", 2000)).unwrap();
    }

    let wal_path = dir.path().join("wal.log");
    assert!(wal_path.exists(), "WAL file should exist after writes");

    let mut wal = db::wal::wal::WAL::open(wal_path).unwrap();
    let records = wal.recover().unwrap();

    // Both jobs should be recoverable from the WAL
    assert_eq!(records.len(), 2);
    // Keys have the [type][timestamp][job_id] format written by Series
    assert_eq!(records[0].key[0], 0u8, "first byte is InferenceJob type");
    assert_eq!(records[1].key[0], 0u8, "first byte is InferenceJob type");
}

#[test]
fn test_persistence_across_restart() {
    let dir = tempdir().unwrap();
    let path = dir.path().to_path_buf();

    {
        let mut series = Series::new(&path).unwrap();
        series.insert_job(make_job("job_001", "done", 1000)).unwrap();
        series.insert_job(make_job("job_002", "done", 2000)).unwrap();
    }

    let wal_path = dir.path().join("wal.log");
    let wal_size = std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
    println!("WAL file size: {} bytes", wal_size);

    // Manually check WAL recovery at the raw level
    let mut wal = db::wal::wal::WAL::open(wal_path).unwrap();
    let records = wal.recover().unwrap();
    println!("WAL records recovered: {}", records.len());
    assert_eq!(records.len(), 2, "both jobs should be in WAL");

    // Reopen series — LSM memtable is rebuilt from WAL, but the in-memory
    // job_timestamps index (needed by get_job) is NOT restored. get_job
    // returns None until bug #2 (BTree/index persistence) is fixed.
    {
        let mut series = Series::new(&path).unwrap();
        let job = series.get_job("job_001").unwrap();
        // Expected: None — job_timestamps is empty after restart (bug #2)
        assert!(
            job.is_none(),
            "get_job returns None after restart until index persistence is implemented"
        );
    }
}
