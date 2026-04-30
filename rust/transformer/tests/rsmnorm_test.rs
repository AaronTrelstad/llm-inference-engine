use transformer::rmsnorm::RMSNorm;
use candle_core::{Device, DType, Tensor};

#[test]
fn test_rmsnorm_shape_preserved() {
    let device = Device::Cpu;
    let weight = Tensor::ones(4096, DType::F32, &device).unwrap();
    let norm = RMSNorm::new(weight, 1e-5);

    let x = Tensor::randn(0f32, 1f32, (1, 3, 4096), &device).unwrap();
    let out = norm.forward(&x).unwrap();

    assert_eq!(out.dims(), &[1, 3, 4096]);
}

#[test]
fn test_rmsnorm_normalizes() {
    let device = Device::Cpu;
    let weight = Tensor::ones(4096, DType::F32, &device).unwrap();
    let norm = RMSNorm::new(weight, 1e-5);

    let x = Tensor::randn(0f32, 1f32, (1, 1, 4096), &device).unwrap();
    let out = norm.forward(&x).unwrap();

    let rms = out.sqr().unwrap()
        .mean_all().unwrap()    
        .sqrt().unwrap()
        .to_scalar::<f32>().unwrap(); 

    assert!((rms - 1.0).abs() < 0.01, "RMS should be ~1.0, got {}", rms);
}
