# llm-serving-stack

End-to-end LLM serving platform built from scratch in Rust and Go, implementing disaggregated inference with dedicated prefill and decode workers. The Rust core implements a LLaMA-compatible transformer with sparse attention, a PagedAttention KV cache, and a custom time-series database. A Go HTTP backend handles request intake and GPU-aware scheduling, routing inference jobs to Rust workers via gRPC by KV cache locality.
