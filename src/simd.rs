use crate::util::{le_u8s_to_u32s, u32s_to_le_u8s};
use std::simd::Simd;
/// Encrypts a plaintext using the ChaCha20 algorithm
///
/// The ChaCha20 algorithm generates a keystream the length of at least the plaintext, then XOR's it
/// with the plaintext. If the plaintext is not a multiple of 64 bytes, the keystream is truncated.
///
/// This is the 128-bit vectorized version of the algorithm, it requires the AVX-2 extension to x86.
/// For the iterative version, see [`crate::algo`], for the quad-SIMD version, see
/// [`crate::quad_simd`].
///
/// ## Parameters
///
/// - `plaintext`: The plaintext bytes to encrypt
/// - `key`: The 256-bit key
/// - `nonce`: The 96-bit nonce
///
/// ## Returns
///
/// The encrypted ciphertext as a bytes vector
///
/// ## Example
///
/// ```rust
/// use chacha20::simd::encrypt;
///
/// let plaintext = b"Hello, world!";
///
/// let key: [u8; 32] = [
///     0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
///     0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
///     0x1c, 0x1d, 0x1e, 0x1f,
/// ];
///
/// let nonce: [u8; 12] = [
///     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
/// ];
///
/// let block_count: u32 = 1;
///
/// let ciphertext = encrypt(plaintext, &key, &nonce, block_count);
///
/// let decrypted = encrypt(&ciphertext, &key, &nonce, block_count);
///
/// assert_eq!(decrypted, plaintext);
/// ```
pub fn encrypt(plaintext: &[u8], key: &[u8; 32], nonce: &[u8; 12], block_count: u32) -> Vec<u8> {
    let key: [u32; 8] = le_u8s_to_u32s(key);
    let nonce: [u32; 3] = le_u8s_to_u32s(nonce);
    let blocks = (plaintext.len() / 64) as u32 + block_count;

    let key_stream = (block_count..=blocks)
        .map(|i| ChaCha20::new(&key, &nonce, i).gen_key_stream())
        .flat_map(|words| u32s_to_le_u8s::<64>(&words));

    plaintext
        .iter()
        .zip(key_stream)
        .map(|(p, k)| p ^ k)
        .collect()
}

/// Data type to encapsulate the ChaCha20 SIMD buffers
///
/// Rather than treat the state as a 16 element array of u32's, we load each row into a SIMD buffer.
///
/// Column-wise operations are performed by using the traits implemented on the SIMD type. To handle
/// operations on the diagonals, we transform the state by doing element-wise left-rotation before
/// the diagonal quarter-rounds, then right-rotation after.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ChaCha20 {
    a: Simd<u32, 4>,
    b: Simd<u32, 4>,
    c: Simd<u32, 4>,
    d: Simd<u32, 4>,
}

impl ChaCha20 {
    /// Creates a new ChaCha20 instance
    ///
    /// ## Parameters
    ///
    /// - `key`: The 256-bit key
    /// - `nonce`: The 96-bit nonce
    /// - `block_count`: The block count
    ///
    /// ## Returns
    ///
    /// A new ChaCha20 instance with the initial state
    ///
    /// ## Example
    ///
    /// ```rust
    /// use chacha20::simd::ChaCha20;
    ///
    /// let key: [u32; 8] = [
    ///     0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c, 0x13121110, 0x17161514, 0x1b1a1918,
    ///     0x1f1e1d1c,
    /// ];
    ///
    /// let nonce: [u32; 3] = [0x00000001, 0x00000000, 0x4a000000];
    ///
    /// let block_count: u32 = 1;
    ///
    /// let state = [
    ///     0x61707865, 0x3320646e, 0x79622d32, 0x6b206574, 0x03020100, 0x07060504, 0x0b0a0908,
    ///     0x0f0e0d0c, 0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c, 0x00000001, 0x00000001,
    ///     0x00000000, 0x4a000000
    /// ];
    ///
    /// let chacha = ChaCha20::new(&key, &nonce, block_count);
    /// ```
    pub fn new(key: &[u32; 8], nonce: &[u32; 3], block_count: u32) -> Self {
        Self {
            a: Simd::from([0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]),
            b: Simd::from([key[0], key[1], key[2], key[3]]),
            c: Simd::from([key[4], key[5], key[6], key[7]]),
            d: Simd::from([block_count, nonce[0], nonce[1], nonce[2]]),
        }
    }

    /// Generates the key stream using the ChaCha20 algorithm.
    ///
    /// All inputs are in state, so no external parameters are necessary. Since all columns are
    /// computed independently, we can use SIMD to parallelize these operations.
    ///
    /// ## Returns
    ///
    /// The key stream as a 16-element array of u32's
    ///
    /// ## Example
    ///
    /// ```rust
    /// use chacha20::simd::ChaCha20;
    /// use chacha20::util::le_u8s_to_u32s;
    ///
    /// let key: [u8; 32] = [
    ///     0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
    ///     0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
    ///     0x1c, 0x1d, 0x1e, 0x1f,
    /// ];
    ///
    /// let nonce: [u8; 12] = [
    ///     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x01,
    /// ];
    ///
    /// let block_count: u32 = 1;
    ///
    /// let key_stream = [
    ///     0x7c66869a, 0xb3f915d2, 0xca238319, 0x974c554a, 0xf2bb0a6b, 0xab028b9c, 0xcf878654,
    ///     0x19b185da, 0xf275298b, 0x67092355, 0x9d0650f9, 0xa861070b, 0x70d35dd8, 0xeca9486a,
    ///     0xe2c3fe4d, 0xf0ad6108
    /// ];
    ///
    /// let key_u32s = le_u8s_to_u32s(&key);
    /// let nonce_u32s = le_u8s_to_u32s(&nonce);
    ///
    /// let chacha = ChaCha20::new(&key_u32s, &nonce_u32s, block_count);
    ///
    /// assert_eq!(chacha.gen_key_stream(), key_stream);
    /// ```
    pub fn gen_key_stream(&self) -> [u32; 16] {
        let mut working_state = self.clone();

        for _ in 0..10 {
            working_state.quarter_round();

            working_state.b = working_state.b.rotate_elements_left::<1>();
            working_state.c = working_state.c.rotate_elements_left::<2>();
            working_state.d = working_state.d.rotate_elements_left::<3>();

            working_state.quarter_round();

            working_state.b = working_state.b.rotate_elements_right::<1>();
            working_state.c = working_state.c.rotate_elements_right::<2>();
            working_state.d = working_state.d.rotate_elements_right::<3>();
        }

        working_state.a += self.a;
        working_state.b += self.b;
        working_state.c += self.c;
        working_state.d += self.d;

        [
            working_state.a.to_array(),
            working_state.b.to_array(),
            working_state.c.to_array(),
            working_state.d.to_array(),
        ]
        .concat()
        .try_into()
        .unwrap()
    }

    /// Performs a quarter round on the state.
    ///
    /// The quarter round is the core operation of the ChaCha20 algorithm.
    ///
    /// > Note: The function mutates the state in-place.
    #[inline(always)]
    fn quarter_round(&mut self) {
        self.a += self.b;
        self.d ^= self.a;
        self.d = Self::rotl(self.d, 16);

        self.c += self.d;
        self.b ^= self.c;
        self.b = Self::rotl(self.b, 12);

        self.a += self.b;
        self.d ^= self.a;
        self.d = Self::rotl(self.d, 8);

        self.c += self.d;
        self.b ^= self.c;
        self.b = Self::rotl(self.b, 7);
    }

    /// Rotates the bits in each element of a SIMD buffer to the left by a given amount.
    ///
    /// While `u32::rotate_left` is not available in SIMD buffers, we can achieve the same effect by
    /// shifting the elements in opposite directions and then OR'ing them together.
    ///
    /// > Note: The SIMD type "splats" the shift amount across all elements.
    ///
    /// ## Parameters
    ///
    /// - `a`: The SIMD buffer to rotate
    /// - `b`: The amount by which to rotate
    ///
    /// ## Returns
    ///
    /// The rotated SIMD buffer
    #[inline(always)]
    fn rotl(a: Simd<u32, 4>, b: u32) -> Simd<u32, 4> {
        let lower = a >> (32 - b);
        let upper = a << b;

        lower | upper
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quarter_round() {
        let mut input = ChaCha20 {
            a: Simd::from([0x11111111, 0x00, 0x00, 0x00]),
            b: Simd::from([0x01020304, 0x00, 0x00, 0x00]),
            c: Simd::from([0x9b8d6f43, 0x00, 0x00, 0x00]),
            d: Simd::from([0x01234567, 0x00, 0x00, 0x00]),
        };

        let expected = ChaCha20 {
            a: Simd::from([0xea2a92f4, 0x00, 0x00, 0x00]),
            b: Simd::from([0xcb1cf8ce, 0x00, 0x00, 0x00]),
            c: Simd::from([0x4581472e, 0x00, 0x00, 0x00]),
            d: Simd::from([0x5881c4bb, 0x00, 0x00, 0x00]),
        };

        input.quarter_round();

        assert_eq!(input, expected);
    }

    #[test]
    fn test_gen_key_stream() {
        let key: [u8; 32] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ];

        let nonce: [u8; 12] = [
            0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00,
        ];

        let block_count: u32 = 1;

        let expected: [u32; 16] = [
            0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3, 0xc7f4d1c7, 0x0368c033, 0x9aaa2204,
            0x4e6cd4c3, 0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9, 0xd19c12b5, 0xb94e16de,
            0xe883d0cb, 0x4e3c50a2,
        ];

        let key_u32s = le_u8s_to_u32s(&key);
        let nonce_u32s = le_u8s_to_u32s(&nonce);

        let chacha = ChaCha20::new(&key_u32s, &nonce_u32s, block_count);

        assert_eq!(chacha.gen_key_stream(), expected);
    }

    #[test]
    fn test_encrypt() {
        let plaintext: [u8; 114] = [
            0x4c, 0x61, 0x64, 0x69, 0x65, 0x73, 0x20, 0x61, 0x6e, 0x64, 0x20, 0x47, 0x65, 0x6e,
            0x74, 0x6c, 0x65, 0x6d, 0x65, 0x6e, 0x20, 0x6f, 0x66, 0x20, 0x74, 0x68, 0x65, 0x20,
            0x63, 0x6c, 0x61, 0x73, 0x73, 0x20, 0x6f, 0x66, 0x20, 0x27, 0x39, 0x39, 0x3a, 0x20,
            0x49, 0x66, 0x20, 0x49, 0x20, 0x63, 0x6f, 0x75, 0x6c, 0x64, 0x20, 0x6f, 0x66, 0x66,
            0x65, 0x72, 0x20, 0x79, 0x6f, 0x75, 0x20, 0x6f, 0x6e, 0x6c, 0x79, 0x20, 0x6f, 0x6e,
            0x65, 0x20, 0x74, 0x69, 0x70, 0x20, 0x66, 0x6f, 0x72, 0x20, 0x74, 0x68, 0x65, 0x20,
            0x66, 0x75, 0x74, 0x75, 0x72, 0x65, 0x2c, 0x20, 0x73, 0x75, 0x6e, 0x73, 0x63, 0x72,
            0x65, 0x65, 0x6e, 0x20, 0x77, 0x6f, 0x75, 0x6c, 0x64, 0x20, 0x62, 0x65, 0x20, 0x69,
            0x74, 0x2e,
        ];

        let key: [u8; 32] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ];

        let nonce: [u8; 12] = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00,
        ];

        let block_count: u32 = 1;

        let first_block_expected: [u32; 16] = [
            0xf3514f22, 0xe1d91b40, 0x6f27de2f, 0xed1d63b8, 0x821f138c, 0xe2062c3d, 0xecca4f7e,
            0x78cff39e, 0xa30a3b8a, 0x920a6072, 0xcd7479b5, 0x34932bed, 0x40ba4c79, 0xcd343ec6,
            0x4c2c21ea, 0xb7417df0,
        ];

        let second_block_expected: [u32; 16] = [
            0x9f74a669, 0x410f633f, 0x28feca22, 0x7ec44dec, 0x6d34d426, 0x738cb970, 0x3ac5e9f3,
            0x45590cc4, 0xda6e8b39, 0x892c831a, 0xcdea67c1, 0x2b7e1d90, 0x037463f3, 0xa11a2073,
            0xe8bcfb88, 0xedc49139,
        ];

        let ciphertext_expected = [
            0x6e, 0x2e, 0x35, 0x9a, 0x25, 0x68, 0xf9, 0x80, 0x41, 0xba, 0x07, 0x28, 0xdd, 0x0d,
            0x69, 0x81, 0xe9, 0x7e, 0x7a, 0xec, 0x1d, 0x43, 0x60, 0xc2, 0x0a, 0x27, 0xaf, 0xcc,
            0xfd, 0x9f, 0xae, 0x0b, 0xf9, 0x1b, 0x65, 0xc5, 0x52, 0x47, 0x33, 0xab, 0x8f, 0x59,
            0x3d, 0xab, 0xcd, 0x62, 0xb3, 0x57, 0x16, 0x39, 0xd6, 0x24, 0xe6, 0x51, 0x52, 0xab,
            0x8f, 0x53, 0x0c, 0x35, 0x9f, 0x08, 0x61, 0xd8, 0x07, 0xca, 0x0d, 0xbf, 0x50, 0x0d,
            0x6a, 0x61, 0x56, 0xa3, 0x8e, 0x08, 0x8a, 0x22, 0xb6, 0x5e, 0x52, 0xbc, 0x51, 0x4d,
            0x16, 0xcc, 0xf8, 0x06, 0x81, 0x8c, 0xe9, 0x1a, 0xb7, 0x79, 0x37, 0x36, 0x5a, 0xf9,
            0x0b, 0xbf, 0x74, 0xa3, 0x5b, 0xe6, 0xb4, 0x0b, 0x8e, 0xed, 0xf2, 0x78, 0x5e, 0x42,
            0x87, 0x4d,
        ];

        let key_u32s = le_u8s_to_u32s(&key);
        let nonce_u32s = le_u8s_to_u32s(&nonce);

        assert_eq!(
            ChaCha20::new(&key_u32s, &nonce_u32s, 1).gen_key_stream(),
            first_block_expected
        );
        assert_eq!(
            ChaCha20::new(&key_u32s, &nonce_u32s, 2).gen_key_stream(),
            second_block_expected
        );
        assert_eq!(
            encrypt(&plaintext, &key, &nonce, block_count),
            ciphertext_expected.to_vec()
        );
    }
}
