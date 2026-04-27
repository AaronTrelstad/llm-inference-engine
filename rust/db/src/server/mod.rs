use async_trait::async_trait;
use tonic::{transport::Server, Request, Response, Status};
use std::sync::{Arc, Mutex};
use crate::series::series::{
    Series, JobRecord, MetricRecord, JobFilter, MetricFilter,
    GPUAggregation, JobQueryResult,
};

pub mod db {
    tonic::include_proto!("db");
}

use db::db_service_server::{DbService, DbServiceServer};
use db::*;

pub struct DbServer {
    series: Arc<Mutex<Series>>,
}

impl DbServer {
    pub fn new(series: Series) -> Self {
        Self {
            series: Arc::new(Mutex::new(series)),
        }
    }
}

#[async_trait]
impl DbService for DbServer {
    async fn insert_job(
        &self,
        req: Request<JobRequest>,
    ) -> Result<Response<Empty>, Status> {
        let r = req.into_inner();
        let job = JobRecord {
            job_id:            r.job_id,
            status:            r.status,
            prompt:            r.prompt,
            output:            r.output,
            prefill_worker_id: if r.prefill_worker_id.is_empty() { None } else { Some(r.prefill_worker_id) },
            decode_worker_id:  if r.decode_worker_id.is_empty()  { None } else { Some(r.decode_worker_id)  },
            latency:           r.latency,
            timestamp:         r.timestamp,
        };

        self.series
            .lock()
            .unwrap()
            .insert_job(job)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(Empty {}))
    }

    async fn insert_metric(
        &self,
        req: Request<MetricRequest>,
    ) -> Result<Response<Empty>, Status> {
        let r = req.into_inner();
        let metric = MetricRecord {
            worker_id:      r.worker_id,
            gpu_memory:     r.gpu_memory,
            gpu_util:       r.gpu_util,
            active_jobs:    r.active_jobs,
            tokens_per_sec: r.tokens_per_sec,
            timestamp:      r.timestamp,
        };

        self.series
            .lock()
            .unwrap()
            .insert_metrics(metric)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(Empty {}))
    }

    async fn get_job(
        &self,
        req: Request<GetJobRequest>,
    ) -> Result<Response<JobResponse>, Status> {
        let job_id = req.into_inner().job_id;

        let result = self.series
            .lock()
            .unwrap()
            .get_job(&job_id)
            .map_err(|e| Status::internal(e.to_string()))?;

        match result {
            Some(job) => Ok(Response::new(JobResponse {
                found: true,
                job:   Some(job_to_proto(job)),
            })),
            None => Ok(Response::new(JobResponse {
                found: false,
                job:   None,
            })),
        }
    }

    async fn query_jobs(
        &self,
        req: Request<JobFilterRequest>,
    ) -> Result<Response<JobQueryResponse>, Status> {
        let r = req.into_inner();

        let filter = JobFilter {
            status:     if r.status.is_empty()    { None } else { Some(r.status)    },
            worker_id:  if r.worker_id.is_empty() { None } else { Some(r.worker_id) },
            time_range: if r.time_start == 0 && r.time_end == 0 {
                None
            } else {
                Some((r.time_start, r.time_end))
            },
            aggregate: r.aggregate,
        };

        let result = self.series
            .lock()
            .unwrap()
            .query_jobs(filter)
            .map_err(|e| Status::internal(e.to_string()))?;

        match result {
            JobQueryResult::Records(jobs) => Ok(Response::new(JobQueryResponse {
                is_aggregate: false,
                jobs:         jobs.into_iter().map(job_to_proto).collect(),
                aggregate:    None,
            })),
            JobQueryResult::Aggregate(agg) => Ok(Response::new(JobQueryResponse {
                is_aggregate: true,
                jobs:         vec![],
                aggregate:    Some(JobAggregateResponse {
                    avg_latency: agg.avg_latency,
                    p99_latency: agg.p99_latency,
                    max_latency: agg.max_latency,
                    min_latency: agg.min_latency,
                    total_count: agg.total_count as u64,
                }),
            })),
        }
    }

    async fn query_metrics(
        &self,
        req: Request<MetricFilterRequest>,
    ) -> Result<Response<MetricQueryResponse>, Status> {
        let r = req.into_inner();

        let agg = match r.aggregation.as_str() {
            "avg" => Some(GPUAggregation::Avg),
            "max" => Some(GPUAggregation::Max),
            "min" => Some(GPUAggregation::Min),
            "p99" => Some(GPUAggregation::P99),
            _     => None,
        };

        let filter = MetricFilter {
            worker_id:   if r.worker_id.is_empty() { None } else { Some(r.worker_id) },
            time_range:  if r.time_start == 0 && r.time_end == 0 {
                None
            } else {
                Some((r.time_start, r.time_end))
            },
            aggregation: agg,
        };

        let metrics = self.series
            .lock()
            .unwrap()
            .query_metrics(filter)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(MetricQueryResponse {
            metrics: metrics.into_iter().map(metric_to_proto).collect(),
        }))
    }
}

fn job_to_proto(job: JobRecord) -> JobRequest {
    JobRequest {
        job_id:            job.job_id,
        status:            job.status,
        prompt:            job.prompt,
        output:            job.output,
        prefill_worker_id: job.prefill_worker_id.unwrap_or_default(),
        decode_worker_id:  job.decode_worker_id.unwrap_or_default(),
        latency:           job.latency,
        timestamp:         job.timestamp,
    }
}

fn metric_to_proto(metric: MetricRecord) -> MetricRequest {
    MetricRequest {
        worker_id:      metric.worker_id,
        gpu_memory:     metric.gpu_memory,
        gpu_util:       metric.gpu_util,
        active_jobs:    metric.active_jobs,
        tokens_per_sec: metric.tokens_per_sec,
        timestamp:      metric.timestamp,
    }
}

pub async fn serve(
    series: Series,
    port:   u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr   = format!("0.0.0.0:{}", port).parse()?;
    let server = DbServer::new(series);

    println!("DB gRPC server listening on {}", addr);

    Server::builder()
        .add_service(DbServiceServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}
