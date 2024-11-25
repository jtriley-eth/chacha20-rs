use crate::util::{le_u8s_to_u32s, u32s_to_le_u8s};
use std::simd::{simd_swizzle, Simd};
/// Encrypts a plaintext using the ChaCha20 algorithm
///
/// The ChaCha20 algorithm generates a keystream the length of at least the plaintext, then XOR's it
/// with the plaintext. If the plaintext is not a multiple of 64 bytes, the keystream is truncated.
///
/// This is the 512-bit vectorized version of the algorithm, it requires the AVX-512 extension to
/// x86. For the iterative version, see [`crate::algo`], for the simplified SIMD version, see
/// [`crate::simd`].
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
/// let ciphertext = encrypt(plaintext, &key, &nonce);
///
/// let decrypted = encrypt(&ciphertext, &key, &nonce);
///
/// assert_eq!(decrypted, plaintext);
/// ```
pub fn encrypt(plaintext: &[u8], key: &[u8; 32], nonce: &[u8; 12]) -> Vec<u8> {
    let key: [u32; 8] = le_u8s_to_u32s(key);
    let nonce: [u32; 3] = le_u8s_to_u32s(nonce);
    let blocks = plaintext.len() / 265 + 1;

    let key_stream = (1..=blocks as u32)
        .map(|i| ChaCha20::new(&key, &nonce, i).gen_key_stream())
        .flat_map(|words| u32s_to_le_u8s::<256>(&words));

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
    a: Simd<u32, 16>,
    b: Simd<u32, 16>,
    c: Simd<u32, 16>,
    d: Simd<u32, 16>,
}

impl ChaCha20 {
    /// Creates a new ChaCha20 instance
    ///
    /// ## Parameters
    ///
    /// - `key`: The 256-bit key
    /// - `nonce`: The 96-bit nonce
    /// - `start_block_count`: The start block count (it will return the next four)
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
    /// let start_block_count: u32 = 1;
    ///
    /// let chacha = ChaCha20::new(&key, &nonce, start_block_count);
    /// ```
    pub fn new(key: &[u32; 8], nonce: &[u32; 3], start_block_count: u32) -> Self {
        Self {
            a: Simd::from([
                0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
                0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
                0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
                0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
            ]),
            b: Simd::from([
                key[0], key[1], key[2], key[3],
                key[0], key[1], key[2], key[3],
                key[0], key[1], key[2], key[3],
                key[0], key[1], key[2], key[3],
            ]),
            c: Simd::from([
                key[4], key[5], key[6], key[7],
                key[4], key[5], key[6], key[7],
                key[4], key[5], key[6], key[7],
                key[4], key[5], key[6], key[7],
            ]),
            d: Simd::from([
                start_block_count,
                nonce[0],
                nonce[1],
                nonce[2],
                start_block_count + 1,
                nonce[0],
                nonce[1],
                nonce[2],
                start_block_count + 2,
                nonce[0],
                nonce[1],
                nonce[2],
                start_block_count + 3,
                nonce[0],
                nonce[1],
                nonce[2],
            ]),
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
    /// let start_block_count: u32 = 1;
    /// 
    /// let key_u32s = le_u8s_to_u32s(&key);
    /// 
    /// let nonce_u32s = le_u8s_to_u32s(&nonce);
    ///
    /// let chacha = ChaCha20::new(&key_u32s, &nonce_u32s, start_block_count);
    ///
    /// let key_stream = chacha.gen_key_stream();
    /// ```
    pub fn gen_key_stream(&self) -> [u32; 64] {
        let mut working_state = self.clone();

        for _ in 0..10 {
            working_state.quarter_round();
            working_state.swizzle_left();

            working_state.quarter_round();
            working_state.swizzle_right();
        }

        working_state.a += self.a;
        working_state.b += self.b;
        working_state.c += self.c;
        working_state.d += self.d;

        let a = working_state.a.to_array();
        let b = working_state.b.to_array();
        let c = working_state.c.to_array();
        let d = working_state.d.to_array();

        [
            a[00], a[01], a[02], a[03], b[00], b[01], b[02], b[03], c[00], c[01], c[02], c[03],
            d[00], d[01], d[02], d[03], a[04], a[05], a[06], a[07], b[04], b[05], b[06], b[07],
            c[04], c[05], c[06], c[07], d[04], d[05], d[06], d[07], a[08], a[09], a[10], a[11],
            b[08], b[09], b[10], b[11], c[08], c[09], c[10], c[11], d[08], d[09], d[10], d[11],
            a[12], a[13], a[14], a[15], b[12], b[13], b[14], b[15], c[12], c[13], c[14], c[15],
            d[12], d[13], d[14], d[15],
        ]
    }

    /// Swizzles the state to the left
    ///
    /// Since the SIMD buffer holds four instances of the state row, when diagonalizing state
    /// between quarter rounds, we need to perform element-wise left-rotation, relative to the four
    /// element rows. That is to say we rotate as follows.
    ///
    /// ```plaintext
    /// .-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.
    /// | a_0 | b_0 | c_0 | d_0 | a_1 | b_1 | c_1 | d_1 | a_2 | b_2 | c_2 | d_2 | a_3 | b_3 | c_3 | d_3 |
    /// '-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'
    ///    |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    ///    |  .--'  .--'  .--'     |  .--'  .--'  .--'     |  .--'  .--'  .--'     |  .--'  .--'  .--'
    ///    '--|-----|-----|--.     '--|-----|-----|--.     '--|-----|-----|--.     '--|-----|-----|--.
    ///    .--'  .--'  .--'  |     .--'  .--'  .--'  |     .--'  .--'  .--'  |     .--'  .--'  .--'  |
    ///    |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    ///    V     V     V     V     V     V     V     V     V     V     V     V     V     V     V     V
    /// .-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.
    /// | b_0 | c_0 | d_0 | a_0 | b_1 | c_1 | d_1 | a_1 | b_2 | c_2 | d_2 | a_2 | b_3 | c_3 | d_3 | a_3 |
    /// '-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'
    /// ```
    #[inline(always)]
    fn swizzle_left(&mut self) {
        self.b = simd_swizzle!(self.b, [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12]);
        self.c = simd_swizzle!(self.c, [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13]);
        self.d = simd_swizzle!(self.d, [3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14]);
    }

    /// Swizzles the state to the right
    ///
    /// Since the SIMD buffer holds four instances of the state row, when diagonalizing state
    /// between quarter rounds, we need to perform element-wise left-rotation, relative to the four
    /// element rows. That is to say we rotate as follows.
    ///
    /// ```plaintext
    /// .-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.
    /// | b_0 | c_0 | d_0 | a_0 | b_1 | c_1 | d_1 | a_1 | b_2 | c_2 | d_2 | a_2 | b_3 | c_3 | d_3 | a_3 |
    /// '-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'
    ///    |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    ///    '--.  '--.  '--.  |     '--.  '--.  '--.  |     '--.  '--.  '--.  |     '--.  '--.  '--.  |
    ///    .--|-----|-----|--'     .--|-----|-----|--'     .--|-----|-----|--'     .--|-----|-----|--'
    ///    |  '--.  '--.  '--.     |  '--.  '--.  '--.     |  '--.  '--.  '--.     |  '--.  '--.  '--.
    ///    |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    ///    V     V     V     V     V     V     V     V     V     V     V     V     V     V     V     V
    /// .-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.-----.
    /// | a_0 | b_0 | c_0 | d_0 | a_1 | b_1 | c_1 | d_1 | a_2 | b_2 | c_2 | d_2 | a_3 | b_3 | c_3 | d_3 |
    /// '-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'-----'
    /// ```
    #[inline(always)]
    fn swizzle_right(&mut self) {
        self.b = simd_swizzle!(self.b, [3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14]);
        self.c = simd_swizzle!(self.c, [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13]);
        self.d = simd_swizzle!(self.d, [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12]);
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
    fn rotl(a: Simd<u32, 16>, b: u32) -> Simd<u32, 16> {
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
            a: Simd::from([
                0x11111111, 0x00, 0x00, 0x00, 0x11111111, 0x00, 0x00, 0x00, 0x11111111, 0x00, 0x00,
                0x00, 0x11111111, 0x00, 0x00, 0x00,
            ]),
            b: Simd::from([
                0x01020304, 0x00, 0x00, 0x00, 0x01020304, 0x00, 0x00, 0x00, 0x01020304, 0x00, 0x00,
                0x00, 0x01020304, 0x00, 0x00, 0x00,
            ]),
            c: Simd::from([
                0x9b8d6f43, 0x00, 0x00, 0x00, 0x9b8d6f43, 0x00, 0x00, 0x00, 0x9b8d6f43, 0x00, 0x00,
                0x00, 0x9b8d6f43, 0x00, 0x00, 0x00,
            ]),
            d: Simd::from([
                0x01234567, 0x00, 0x00, 0x00, 0x01234567, 0x00, 0x00, 0x00, 0x01234567, 0x00, 0x00,
                0x00, 0x01234567, 0x00, 0x00, 0x00,
            ]),
        };

        let expected = ChaCha20 {
            a: Simd::from([
                0xea2a92f4, 0x00, 0x00, 0x00, 0xea2a92f4, 0x00, 0x00, 0x00, 0xea2a92f4, 0x00, 0x00,
                0x00, 0xea2a92f4, 0x00, 0x00, 0x00,
            ]),
            b: Simd::from([
                0xcb1cf8ce, 0x00, 0x00, 0x00, 0xcb1cf8ce, 0x00, 0x00, 0x00, 0xcb1cf8ce, 0x00, 0x00,
                0x00, 0xcb1cf8ce, 0x00, 0x00, 0x00,
            ]),
            c: Simd::from([
                0x4581472e, 0x00, 0x00, 0x00, 0x4581472e, 0x00, 0x00, 0x00, 0x4581472e, 0x00, 0x00,
                0x00, 0x4581472e, 0x00, 0x00, 0x00,
            ]),
            d: Simd::from([
                0x5881c4bb, 0x00, 0x00, 0x00, 0x5881c4bb, 0x00, 0x00, 0x00, 0x5881c4bb, 0x00, 0x00,
                0x00, 0x5881c4bb, 0x00, 0x00, 0x00,
            ]),
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

        let start_block_count: u32 = 1;

        let expected = [
            0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3, 0xc7f4d1c7, 0x0368c033, 0x9aaa2204,
            0x4e6cd4c3, 0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9, 0xd19c12b5, 0xb94e16de,
            0xe883d0cb, 0x4e3c50a2, 0x7783880a, 0x4ebfd739, 0xb0acccf8, 0xd6b92bea, 0x94c3569d,
            0xfd1d35aa, 0x9f45bfa5, 0xe89f2e0a, 0x92f821e7, 0x86c4f955, 0x9c6721bf, 0x9c4f3d68,
            0x27faf25c, 0x00265586, 0x37ca065b, 0x3baf864c, 0xcbbdbfdc, 0x8665be83, 0x0ec2d52e,
            0x24435aae, 0xda926a1d, 0x159aca6d, 0x9752e26b, 0x18271cf5, 0x931e868a, 0x12eb3acc,
            0x8b59769a, 0x4527cdac, 0x1b94c63a, 0x511e4e4b, 0xe9fea953, 0x0ea01b5d, 0x0d9fd069,
            0xca786433, 0x5a336890, 0x0909b3e2, 0xe50ffb05, 0x371551d4, 0x5b6e121d, 0x24995ea8,
            0xa79a7232, 0x5edc7dd7, 0xd889c63c, 0xb71a5c44, 0x9e40a754, 0x2bfcbee8, 0xd26838dd,
            0xd81a6e7f,
        ];

        let key_u32s = le_u8s_to_u32s(&key);
        let nonce_u32s = le_u8s_to_u32s(&nonce);

        let chacha = ChaCha20::new(&key_u32s, &nonce_u32s, start_block_count);

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

        assert_eq!(
            encrypt(&plaintext, &key, &nonce),
            ciphertext_expected.to_vec()
        );
    }
}
