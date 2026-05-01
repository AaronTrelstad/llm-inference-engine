use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tonic::{Request, Response, Status, transport::Server};

use super::proto::{DecodeRequest, KvBlock, decode_service_client::DecodeServiceClient};
use super::proto::{
    Empty, HealthResponse, PrefillRequest, PrefillResponse,
    prefill_service_server::{PrefillService, PrefillServiceServer},
};
use crate::generate::prefill::PrefillWorker;

pub struct PrefillServer {
    worker: Arc<PrefillWorker>,
}

impl PrefillServer {
    pub fn new(worker: PrefillWorker) -> Self {
        Self {
            worker: Arc::new(worker),
        }
    }
}

#[async_trait]
impl PrefillService for PrefillServer {
    async fn prefill(
        &self,
        req: Request<PrefillRequest>,
    ) -> Result<Response<PrefillResponse>, Status> {
        let r = req.into_inner();

        let result = self
            .worker
            .prefill(&r.job_id, &r.prompt)
            .map_err(|e| Status::internal(e.to_string()))?;

        let kv_blocks: Vec<KvBlock> = result
            .serialized_kv
            .into_iter()
            .map(|s| KvBlock {
                layer_idx: s.layer_idx as u32,
                k_data: s.k_data,
                v_data: s.v_data,
                k_shape: s.k_shape.iter().map(|&d| d as u32).collect(),
                v_shape: s.v_shape.iter().map(|&d| d as u32).collect(),
                dtype: s.dtype,
            })
            .collect();

        let mut client = DecodeServiceClient::connect(r.decode_worker_addr)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        client
            .decode(DecodeRequest {
                job_id: r.job_id.clone(),
                first_token: result.first_token,
                n_prompt_tokens: result.n_tokens as u32,
                max_tokens: r.max_tokens,
                kv_blocks,
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(PrefillResponse {
            job_id: r.job_id,
            first_token: result.first_token,
            n_tokens: result.n_tokens as u32,
            prefill_worker: result.prefill_worker,
            ok: true,
            error: String::new(),
        }))
    }

    async fn health(&self, _req: Request<Empty>) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            ok: true,
            worker_id: self.worker.worker_id.clone(),
            gpu_util: 0.0,
            active_jobs: 0,
        }))
    }
}

pub async fn serve_prefill(worker: PrefillWorker, port: u16) -> Result<()> {
    let addr = format!("0.0.0.0:{}", port).parse()?;
    let server = PrefillServer::new(worker);

    println!("prefill server listening on {}", addr);

    Server::builder()
        .add_service(PrefillServiceServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}
