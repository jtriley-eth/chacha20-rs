#![feature(portable_simd)]
//! # ChaCha20 With SIMD Support
//!
//! > WARNING: DO NOT USE IN PRODUCTION
//!
//! This is a simple implementation of the ChaCha20 stream cipher with readability in mind. SIMD
//! support is also available.
//!
//! Impementation and tests are from [RFC-7539](https://datatracker.ietf.org/doc/html/rfc7539).
//!
//! This implementation does not include a MAC and without it, is susceptible to bit-flipping
//! attacks.
//!
//! As is recommended in [RFC-7539](https://datatracker.ietf.org/doc/html/rfc7539), Poly1205 is a
//! common and recommended MAC to use in tandem with ChaCha20.

pub mod algo;
pub mod simd;
pub mod util;
