pub mod prefill_server;
pub mod decode_server;

pub mod proto {
    tonic::include_proto!("inference");
}
