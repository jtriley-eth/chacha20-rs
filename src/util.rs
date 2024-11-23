/// Transforms Little-Endian u8 slices into u32 arrays
/// 
/// We use the const-generic parameter to guarantee size at compile-time.
/// 
/// ## Type Parameters
/// 
/// - `WORDS`: The number of u32s in the output
/// 
/// ## Parameters
/// 
/// - `input`: The input slice
/// 
/// ## Returns
/// 
/// The array of u32s
/// 
/// ## Example
/// 
/// ```rust
/// use chacha20::util::le_u8s_to_u32s;
/// 
/// let input = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
/// let output = le_u8s_to_u32s::<2>(&input);
/// 
/// assert_eq!(output, [0x04030201, 0x08070605]);
/// ```
pub fn le_u8s_to_u32s<const WORDS: usize>(input: &[u8]) -> [u32; WORDS] {
    let mut output = [0; WORDS];

    for i in 0..WORDS {
        output[i] = u32::from_le_bytes([
            input[i * 4],
            input[i * 4 + 1],
            input[i * 4 + 2],
            input[i * 4 + 3],
        ]);
    }

    output
}

/// Transforms u32 arrays into Little-Endian u8 slices
/// 
/// We use the const-generic parameter to guarantee size at compile-time.
/// 
/// ## Type Parameters
/// 
/// - `BYTES`: The number of u8s in the output
/// 
/// ## Parameters
/// 
/// - `input`: The input array
/// 
/// ## Returns
/// 
/// The slice of u8s
/// 
/// ## Example
/// 
/// ```rust
/// use chacha20::util::u32s_to_le_u8s;
/// 
/// let input = [0x04030201, 0x08070605];
/// let output = u32s_to_le_u8s::<8>(&input);
/// 
/// assert_eq!(output, [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
/// ```
pub fn u32s_to_le_u8s<const BYTES: usize>(input: &[u32]) -> [u8; BYTES] {
    let mut output = [0; BYTES];

    for i in 0..input.len() {
        output[i * 4] = (input[i] & 0xff) as u8;
        output[i * 4 + 1] = ((input[i] >> 8) & 0xff) as u8;
        output[i * 4 + 2] = ((input[i] >> 16) & 0xff) as u8;
        output[i * 4 + 3] = ((input[i] >> 24) & 0xff) as u8;
    }

    output
}
