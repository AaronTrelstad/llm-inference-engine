use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tonic::{Request, Response, Status, transport::Server};

use super::proto::{
    DecodeRequest, DecodeResponse, Empty, HealthResponse,
    decode_service_server::{DecodeService, DecodeServiceServer},
};
use crate::generate::decode::DecodeWorker;
use crate::generate::kv_transfer::SerializedKV;

pub struct DecodeServer {
    worker: Arc<DecodeWorker>,
}

impl DecodeServer {
    pub fn new(worker: DecodeWorker) -> Self {
        Self {
            worker: Arc::new(worker),
        }
    }
}

#[async_trait]
impl DecodeService for DecodeServer {
    async fn decode(
        &self,
        req: Request<DecodeRequest>,
    ) -> Result<Response<DecodeResponse>, Status> {
        let r = req.into_inner();

        let serialized_kv: Vec<SerializedKV> = r
            .kv_blocks
            .into_iter()
            .map(|b| SerializedKV {
                layer_idx: b.layer_idx as usize,
                k_data: b.k_data,
                v_data: b.v_data,
                k_shape: b.k_shape.iter().map(|&d| d as usize).collect(),
                v_shape: b.v_shape.iter().map(|&d| d as usize).collect(),
                dtype: b.dtype,
            })
            .collect();

        let result = self
            .worker
            .decode(
                &r.job_id,
                r.first_token,
                r.n_prompt_tokens as usize,
                serialized_kv,
                r.max_tokens as usize,
            )
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(DecodeResponse {
            job_id: r.job_id,
            output: result.output,
            n_tokens: result.n_tokens as u32,
            decode_worker: result.decode_worker,
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

pub async fn serve_decode(worker: DecodeWorker, port: u16) -> Result<()> {
    let addr = format!("0.0.0.0:{}", port).parse()?;
    let server = DecodeServer::new(worker);

    println!("decode server listening on {}", addr);

    Server::builder()
        .add_service(DecodeServiceServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}
