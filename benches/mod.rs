use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use chacha20::{algo, quad_simd, simd};

const KEY: [u32; 8] = [
    0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c, 0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
];

const NONCE: [u32; 3] = [0x09000000, 0x4a000000, 0x00000000];

fn test_gen_key_stream_iterative(c: &mut Criterion) {
    c.bench_function("gen_key_stream_iterative", |b| {
        b.iter(|| {
            let mut key_stream = [[0u32; 16]; 4];

            key_stream[0] = algo::gen_key_stream(&KEY, &NONCE, 1);
            key_stream[1] = algo::gen_key_stream(&KEY, &NONCE, 2);
            key_stream[2] = algo::gen_key_stream(&KEY, &NONCE, 3);
            key_stream[3] = algo::gen_key_stream(&KEY, &NONCE, 4);

            black_box(key_stream);
        });
    });
}

fn test_gen_key_stream_simd(c: &mut Criterion) {
    c.bench_function("gen_key_stream_simd", |b| {
        b.iter(|| {
            let mut key_stream = [[0u32; 16]; 4];

            key_stream[0] = simd::ChaCha20::new(&KEY, &NONCE, 1).gen_key_stream();
            key_stream[1] = simd::ChaCha20::new(&KEY, &NONCE, 2).gen_key_stream();
            key_stream[2] = simd::ChaCha20::new(&KEY, &NONCE, 3).gen_key_stream();
            key_stream[3] = simd::ChaCha20::new(&KEY, &NONCE, 4).gen_key_stream();

            black_box(key_stream);
        });
    });
}

fn test_gen_key_stream_quad_simd(c: &mut Criterion) {
    c.bench_function("gen_key_stream_quad_simd", |b| {
        b.iter(|| {
            let mut key_stream = [[0u32; 64]; 1];

            key_stream[0] = quad_simd::ChaCha20::new(&KEY, &NONCE, 1).gen_key_stream();

            black_box(key_stream);
        });
    });
}

criterion_group!(
    benches,
    test_gen_key_stream_iterative,
    test_gen_key_stream_simd,
    test_gen_key_stream_quad_simd
);

criterion_main!(benches);
