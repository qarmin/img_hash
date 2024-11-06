//! A crate that provides several perceptual hashing algorithms for images.
//! Supports images opened with the [image] crate from Piston.
//!
//! ```rust,no_run
//!
//! use image_hasher::{HasherConfig, HashAlg};
//!
//! let image1 = image::open("image1.png").unwrap();
//! let image2 = image::open("image2.png").unwrap();
//!
//! let hasher = HasherConfig::new().to_hasher();
//!
//! let hash1 = hasher.hash_image(&image1);
//! let hash2 = hasher.hash_image(&image2);
//!
//! println!("Image1 hash: {}", hash1.to_base64());
//! println!("Image2 hash: {}", hash2.to_base64());
//!
//! println!("Hamming Distance: {}", hash1.dist(&hash2));
//! ```
//! [image]: https://github.com/PistonDevelopers/image
#![deny(missing_docs)]
#![cfg_attr(feature = "nightly", feature(specialization))]

#[macro_use]
extern crate serde;

use std::borrow::Cow;
use std::fmt;
use std::marker::PhantomData;

pub use alg::{BitOrder, HashAlg};
use base64::Engine;
use dct::DctCtxt;
pub use image::imageops::FilterType;
use image::{imageops, GrayImage};
use serde::{Deserialize, Serialize};
pub(crate) use traits::BitSet;
pub use traits::{DiffImage, HashBytes, Image};

mod dct;

mod alg;
mod traits;

/// **Start here**. Configuration builder for [`Hasher`](::Hasher).
///
/// Playing with the various options on this struct allows you to tune the performance of image
/// hashing to your needs.
///
/// Sane, reasonably fast defaults are provided by the [`::new()`](#method.new) constructor. If
/// you just want to start hashing images and don't care about the details, it's as simple as:
///
/// ```rust
/// use image_hasher::HasherConfig;
///
/// let hasher = HasherConfig::new().to_hasher();
/// // hasher.hash_image(image);
/// ```
///
/// # Configuration Options
/// The hash API is highly configurable to tune both performance characteristics and hash
/// resilience.
///
/// ### Hash Size
/// Setter: [`.hash_size()`](#method.hash_size)
///
/// Dimensions of the final hash, as width x height, in bits. A hash size of `8, 8` produces an
/// 8 x 8 bit (8 byte) hash. Larger hash sizes take more time to compute as well as more memory,
/// but aren't necessarily better for comparing images. The best hash size depends on both
/// the [hash algorithm](#hash-algorithm) and the input dataset. If your images are mostly
/// wide aspect ratio (landscape) then a larger width and a smaller height hash size may be
/// preferable. Optimal values can really only be discovered empirically though.
///
/// (As the author experiments, suggested values will be added here for various algorithms.)
///
/// ### Hash Algorithm
/// Setter: [`.hash_alg()`](#method.hash_alg)
/// Definition: [`HashAlg`](enum.HashAlg.html)
///
/// Multiple methods of calculating image hashes are provided in this crate under the `HashAlg`
/// enum. Each algorithm is different but they all produce the same size hashes as governed by
/// `hash_size`.
///
/// ### Hash Bytes Container / `B` Type Param
/// Use [`with_bytes_type::<B>()`](#method.with_bytes_type) instead of `new()` to customize.
///
/// This hash API allows you to specify the bytes container type for generated hashes. The default
/// allows for any arbitrary hash size (see above) but requires heap-allocation. Instead, you
/// can select an array type which allows hashes to be allocated inline, but requires consideration
/// of the possible sizes of hash you want to generate so you don't waste memory.
///
/// Another advantage of using a constant-sized hash type is that the compiler may be able to
/// produce more optimal code for generating and comparing hashes.
///
/// ```rust
/// # use image_hasher::*;
///
/// // Use default container type, good for any hash size
/// let config = HasherConfig::new();
///
/// /// Inline hash container that exactly fits the default hash size
/// let config = HasherConfig::with_bytes_type::<[u8; 8]>();
/// ```
///
#[derive(Serialize, Deserialize)]
pub struct HasherConfig<B = Box<[u8]>> {
    width: u32,
    height: u32,
    gauss_sigmas: Option<[f32; 2]>,
    #[serde(with = "SerdeFilterType")]
    resize_filter: FilterType,
    dct: bool,
    hash_alg: HashAlg,
    bit_order: BitOrder,
    _bytes_type: PhantomData<B>,
}

impl HasherConfig<Box<[u8]>> {
    /// Construct a new hasher config with sane, reasonably fast defaults.
    ///
    /// A default hash container type is provided as a default type parameter which is guaranteed
    /// to fit any hash size.
    pub fn new() -> Self {
        Self::with_bytes_type()
    }

    /// Construct a new config with the selected [`HashBytes`](trait.HashBytes.html) impl.
    ///
    /// You may opt for an array type which allows inline allocation of hash data.
    ///
    /// ### Note
    /// The default hash size requires 64 bits / 8 bytes of storage. You can change this
    /// with [`.hash_size()`](#method.hash_size).
    pub fn with_bytes_type<B_: HashBytes>() -> HasherConfig<B_> {
        HasherConfig {
            width: 8,
            height: 8,
            gauss_sigmas: None,
            resize_filter: FilterType::Lanczos3,
            dct: false,
            hash_alg: HashAlg::Gradient,
            bit_order: BitOrder::LsbFirst,
            _bytes_type: PhantomData,
        }
    }
}

impl Default for HasherConfig<Box<[u8]>> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: HashBytes> HasherConfig<B> {
    /// Set a new hash width and height; these can be the same.
    ///
    /// The number of bits in the resulting hash will be `width * height`. If you are using
    /// a fixed-size `HashBytes` type then you must ensure it can hold at least this many bits.
    /// You can check this with [`HashBytes::max_bits()`](#method.max_bits).
    ///
    /// ### Rounding Behavior
    /// Certain hash algorithms need to round this value to function properly:
    ///
    /// * [`DoubleGradient`](enum.HashAlg.html#variant.DoubleGradient) rounds to the next multiple of 2;
    /// * [`Blockhash`](enum.HashAlg.html#variant.Blockhash) rounds to the next multiple of 4.
    ///
    /// If the chosen values already satisfy these requirements then nothing is changed.
    ///
    /// ### Recommended Values
    /// The hash granularity increases with `width * height`, although there are diminishing
    /// returns for higher values. Start small. A good starting value to try is `8, 8`.
    ///
    /// When using DCT preprocessing having `width` and `height` be the same value will improve
    /// hashing performance as only one set of coefficients needs to be used.
    #[must_use]
    pub fn hash_size(self, width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..self
        }
    }

    /// Set the filter used to resize images during hashing.
    ///
    /// Note when picking a filter that images are almost always reduced in size.
    /// Has no effect with the Blockhash algorithm as it does not resize.
    #[must_use]
    pub fn resize_filter(self, resize_filter: FilterType) -> Self {
        Self {
            resize_filter,
            ..self
        }
    }

    /// Set the algorithm used to generate hashes.
    ///
    /// Each algorithm has different performance characteristics.
    #[must_use]
    pub fn hash_alg(self, hash_alg: HashAlg) -> Self {
        Self { hash_alg, ..self }
    }

    /// Enable preprocessing with the Discrete Cosine Transform (DCT).
    ///
    /// Does nothing when used with [the Blockhash.io algorithm](HashAlg::Blockhash)
    /// which does not scale the image.
    /// (RFC: it would be possible to shoehorn a DCT into the Blockhash algorithm but it's
    /// not clear what benefits, if any, that would provide).
    ///
    /// After conversion to grayscale, the image is scaled down to `width * 2 x height * 2`
    /// and then the Discrete Cosine Transform is performed on the luminance values. The DCT
    /// essentially transforms the 2D image from the spatial domain with luminance values
    /// to a 2D frequency domain where the values are amplitudes of cosine waves. The resulting
    /// 2D matrix is then cropped to the low `width * height` corner and the
    /// configured hash algorithm is performed on that.
    ///
    /// In layman's terms, this essentially converts the image into a mathematical representation
    /// of the "broad strokes" of the data, which allows the subsequent hashing step to be more
    /// robust against changes that may otherwise produce different hashes, such as significant
    /// edits to portions of the image.
    ///
    /// However, on most machines this usually adds an additional 50-100% to the average hash time.
    ///
    /// This is a very similar process to JPEG compression, although the implementation is too
    /// different for this to be optimized specifically for JPEG encoded images.
    ///
    /// Further Reading:
    /// * http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html
    /// Krawetz describes a "pHash" algorithm which is equivalent to Mean + DCT preprocessing here.
    /// However there is nothing to say that DCT preprocessing cannot compose with other hash
    /// algorithms; Gradient + DCT might well perform better in some aspects.
    /// * https://en.wikipedia.org/wiki/Discrete_cosine_transform
    #[must_use]
    pub fn preproc_dct(self) -> Self {
        Self { dct: true, ..self }
    }

    /// Enable preprocessing with the Difference of Gaussians algorithm with default sigma values.
    ///
    /// Recommended only for use with [the Blockhash.io algorithm](enum.HashAlg#variant.Blockhash)
    /// as it significantly reduces entropy in the scaled down image for other algorithms.
    ///
    /// See [`Self::preproc_diff_gauss_sigmas()](#method.preproc_diff_gauss_sigmas) for more info.
    #[must_use]
    pub fn preproc_diff_gauss(self) -> Self {
        self.preproc_diff_gauss_sigmas(5.0, 10.0)
    }

    /// Enable preprocessing with the Difference of Gaussians algorithm with the given sigma values.
    ///
    /// Recommended only for use with [the Blockhash.io algorithm](enum.HashAlg#variant.Blockhash)
    /// as it significantly reduces entropy in the scaled down image for other algorithms.
    ///
    /// After the image is converted to grayscale, it is blurred with a Gaussian blur using
    /// two different sigmas, and then the images are subtracted from each other. This reduces
    /// the image to just sharp transitions in luminance, i.e. edges. Varying the sigma values
    /// changes how sharp the edges are^[citation needed].
    ///
    /// Further reading:
    /// * https://en.wikipedia.org/wiki/Difference_of_Gaussians
    /// * http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    /// (Difference of Gaussians is an approximation of a Laplacian of Gaussian filter)
    #[must_use]
    pub fn preproc_diff_gauss_sigmas(self, sigma_a: f32, sigma_b: f32) -> Self {
        Self {
            gauss_sigmas: Some([sigma_a, sigma_b]),
            ..self
        }
    }

    /// Change the bit order of the resulting hash.
    ///
    /// After the image has been turned into a series of bits using the [`hash_alg`](#method.hash_alg)
    /// this series of bits has to be turned into a hash. There are two major ways this can be done.
    /// This library defaults to `BitOrder::LsbFirst`, meaning the first bit of the hash algo's output
    /// forms the least significant bit of the first byte of the hash. This means a hash alog output of
    /// 1011 0100 results in a hash of 0010 1101 (or 0x2E). For compatability with hashes created by
    /// other libraries there is the option to instead use `BitOrder::MsbFirst`, which would creat the
    /// hash 1011 0100 (0xB4)
    #[must_use]
    pub fn bit_order(self, bit_order: BitOrder) -> Self {
        Self { bit_order, ..self }
    }

    /// Create a [`Hasher`](struct.Hasher.html) from this config which can be used to hash images.
    ///
    /// ### Panics
    /// If the chosen hash size (`width x height`, rounded for the algorithm if necessary)
    /// is too large for the chosen container type (`B::max_bits()`).
    pub fn to_hasher(&self) -> Hasher<B> {
        let Self {
            hash_alg,
            width,
            height,
            gauss_sigmas,
            resize_filter,
            dct,
            bit_order,
            ..
        } = *self;

        let (width, height) = hash_alg.round_hash_size(width, height);

        assert!(
            (width * height) as usize <= B::max_bits(),
            "hash size too large for container: {width} x {height}",
        );

        // Blockhash doesn't resize the image so don't waste time calculating coefficients
        let dct_coeffs = if dct && hash_alg != HashAlg::Blockhash {
            // calculate the coefficients based on the resize dimensions
            let (dct_width, dct_height) = hash_alg.resize_dimensions(width, height);
            Some(DctCtxt::new(dct_width, dct_height))
        } else {
            None
        };

        Hasher {
            ctxt: HashCtxt {
                gauss_sigmas,
                dct_ctxt: dct_coeffs,
                width,
                height,
                resize_filter,
                bit_order,
            },
            hash_alg,
            bytes_type: PhantomData,
        }
    }
}

// cannot be derived because of `FilterType`
impl<B> fmt::Debug for HasherConfig<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("HasherConfig")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("hash_alg", &self.hash_alg)
            .field("resize_filter", &debug_filter_type(&self.resize_filter))
            .field("gauss_sigmas", &self.gauss_sigmas)
            .field("use_dct", &self.dct)
            .field("bit_order", &self.bit_order)
            .finish()
    }
}

/// Generates hashes for images.
///
/// Constructed via [`HasherConfig::to_hasher()`](struct.HasherConfig#method.to_hasher).
pub struct Hasher<B = Box<[u8]>> {
    ctxt: HashCtxt,
    hash_alg: HashAlg,
    bytes_type: PhantomData<B>,
}

impl<B> Hasher<B>
where
    B: HashBytes,
{
    /// Calculate a hash for the given image with the configured options.
    pub fn hash_image<I: Image>(&self, img: &I) -> ImageHash<B> {
        let hash = self.hash_alg.hash_image(&self.ctxt, img);
        ImageHash {
            hash,
            __backcompat: (),
        }
    }
}

enum CowImage<'a, I: Image> {
    Borrowed(&'a I),
    Owned(I::Buf),
}

impl<'a, I: Image> CowImage<'a, I> {
    fn to_grayscale(&self) -> Cow<GrayImage> {
        match *self {
            CowImage::Borrowed(img) => img.to_grayscale(),
            CowImage::Owned(ref img) => img.to_grayscale(),
        }
    }
}

enum HashVals {
    Floats(Vec<f32>),
    Bytes(Vec<u8>),
}

// TODO: implement `Debug`, needs adaptor for `FilterType`
struct HashCtxt {
    gauss_sigmas: Option<[f32; 2]>,
    dct_ctxt: Option<DctCtxt>,
    resize_filter: FilterType,
    bit_order: BitOrder,
    width: u32,
    height: u32,
}

impl HashCtxt {
    /// If Difference of Gaussians preprocessing is configured, produce a new image with it applied.
    fn gauss_preproc<'a, I: Image>(&self, image: &'a I) -> CowImage<'a, I> {
        if let Some([sigma_a, sigma_b]) = self.gauss_sigmas {
            let mut blur_a = image.blur(sigma_a);
            let blur_b = image.blur(sigma_b);
            blur_a.diff_inplace(&blur_b);

            CowImage::Owned(blur_a)
        } else {
            CowImage::Borrowed(image)
        }
    }

    /// If DCT preprocessing is configured, produce a vector of floats, otherwise a vector of bytes.
    fn calc_hash_vals(&self, img: &GrayImage, width: u32, height: u32) -> HashVals {
        if let Some(ref dct_ctxt) = self.dct_ctxt {
            let img =
                imageops::resize(img, dct_ctxt.width(), dct_ctxt.height(), self.resize_filter);

            let img_vals = img.into_vec();
            let input_len = img_vals.len() + dct_ctxt.required_scratch();

            let mut vals_with_scratch = Vec::with_capacity(input_len);

            // put the image values in [..width * height] and provide scratch space
            vals_with_scratch.extend(img_vals.into_iter().map(|x| x as f32));
            // TODO: compare with `.set_len()`
            vals_with_scratch.resize(input_len, 0.);

            let hash_vals = dct_ctxt.dct_2d(vals_with_scratch);
            HashVals::Floats(dct_ctxt.crop_2d(hash_vals))
        } else {
            let img = imageops::resize(img, width, height, self.resize_filter);
            HashVals::Bytes(img.into_vec())
        }
    }
}

/// A struct representing an image processed by a perceptual hash.
/// For efficiency, does not retain a copy of the image data after hashing.
///
/// Get an instance with `ImageHash::hash()`.
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct ImageHash<B = Box<[u8]>> {
    hash: B,
    __backcompat: (),
}

/// Error that can happen constructing a `ImageHash` from bytes.
#[derive(Debug, PartialEq, Eq)]
pub enum InvalidBytesError {
    /// Byte slice passed to `from_bytes` was the wrong length.
    BytesWrongLength {
        /// Number of bytes the `ImageHash` type expected.
        expected: usize,
        /// Number of bytes found when parsing the hash bytes.
        found: usize,
    },
    /// String passed was not valid base64.
    Base64(base64::DecodeError),
}

impl<B: HashBytes> ImageHash<B> {
    /// Get the bytes of this hash.
    pub fn as_bytes(&self) -> &[u8] {
        self.hash.as_slice()
    }

    /// Create an `ImageHash` instance from the given bytes.
    ///
    /// ## Errors:
    /// Returns a `InvalidBytesError::BytesWrongLength` error if the slice passed can't fit in `B`.
    pub fn from_bytes(bytes: &[u8]) -> Result<ImageHash<B>, InvalidBytesError> {
        if bytes.len() * 8 > B::max_bits() {
            return Err(InvalidBytesError::BytesWrongLength {
                expected: B::max_bits() / 8,
                found: bytes.len(),
            });
        }

        Ok(ImageHash {
            hash: B::from_iter(bytes.iter().copied()),
            __backcompat: (),
        })
    }

    /// Calculate the Hamming distance between this and `other`.
    ///
    /// Equivalent to counting the 1-bits of the XOR of the two hashes.
    ///
    /// Essential to determining the perceived difference between `self` and `other`.
    ///
    /// ### Note
    /// This return value is meaningless if these two hashes are from different hash sizes or
    /// algorithms.
    pub fn dist(&self, other: &Self) -> u32 {
        BitSet::hamming(&self.hash, &other.hash)
    }

    /// Create an `ImageHash` instance from the given Base64-encoded string.
    ///
    /// ## Errors:
    /// Returns `InvalidBytesError::Base64(DecodeError::InvalidLength)` if the string wasn't valid base64.
    /// Otherwise returns the same errors as `from_bytes`.
    pub fn from_base64(encoded_hash: &str) -> Result<ImageHash<B>, InvalidBytesError> {
        let bytes = base64::engine::general_purpose::STANDARD_NO_PAD
            .decode(encoded_hash)
            .map_err(InvalidBytesError::Base64)?;

        Self::from_bytes(&bytes)
    }

    /// Get a Base64 string representing the bits of this hash.
    ///
    /// Mostly for printing convenience.
    pub fn to_base64(&self) -> String {
        base64::engine::general_purpose::STANDARD_NO_PAD.encode(self.hash.as_slice())
    }

    /// Unwraps this `ImageHash` into its inner bytes.
    /// This is useful if you want to move ownership of the bytes to a new struct.
    pub fn into_inner(self) -> B {
        self.hash
    }
}

/// Provide Serde a typedef for `image::FilterType`: https://serde.rs/remote-derive.html
/// This is automatically checked, if Serde complains then double-check with the original definition
#[derive(Serialize, Deserialize)]
#[serde(remote = "FilterType")]
enum SerdeFilterType {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

fn debug_filter_type(ft: &FilterType) -> &'static str {
    use FilterType::*;

    match *ft {
        Triangle => "Triangle",
        Nearest => "Nearest",
        CatmullRom => "CatmullRom",
        Lanczos3 => "Lanczos3",
        Gaussian => "Gaussian",
    }
}

#[cfg(test)]
mod test {
    use image::{ImageBuffer, Rgba};
    use rand::rngs::SmallRng;
    use rand::{RngCore, SeedableRng};

    use super::{HashAlg, HasherConfig, ImageHash};

    type RgbaBuf = ImageBuffer<Rgba<u8>, Vec<u8>>;

    fn gen_test_img(width: u32, height: u32) -> RgbaBuf {
        let len = (width * height * 4) as usize;
        let mut buf = Vec::with_capacity(len);
        unsafe {
            buf.set_len(len);
        } // We immediately fill the buffer.
        let mut rng = SmallRng::seed_from_u64(0xc0ffee);
        rng.fill_bytes(&mut buf);

        ImageBuffer::from_raw(width, height, buf).unwrap()
    }

    macro_rules! test_hash_equality {
        ($fnname:ident, $size:expr, $type:ident, $preproc_dct:expr) => {
            #[test]
            fn $fnname() {
                // square, powers of two
                test_hash_equality!(1024, 1024, $size, $type, $preproc_dct);
                // rectangular, powers of two
                test_hash_equality!(512, 256, $size, $type, $preproc_dct);
                // odd size, square
                test_hash_equality!(967, 967, $size, $type, $preproc_dct);
                // odd size, rectangular
                test_hash_equality!(967, 1023, $size, $type, $preproc_dct);
            }
        };
        ($width:expr, $height:expr, $size:expr, $type:ident, $preproc_dct:expr) => {{
            let test_img = gen_test_img($width, $height);
            let mut cfg = HasherConfig::new()
                .hash_alg(HashAlg::$type)
                .hash_size($size, $size);
            if $preproc_dct {
                if HashAlg::$type != HashAlg::Blockhash {
                    cfg = cfg.preproc_dct();
                } else {
                    cfg = cfg.preproc_diff_gauss();
                }
            }
            let hasher = cfg.to_hasher();
            let hash1 = hasher.hash_image(&test_img);
            let hash2 = hasher.hash_image(&test_img);
            assert_eq!(hash1, hash2);
        }};
    }

    macro_rules! test_hash_type {
        ($type:ident, $modname:ident) => {
            mod $modname {
                use super::*;

                test_hash_equality!(hash_eq_8, 8, $type, false);
                test_hash_equality!(hash_eq_16, 16, $type, false);
                test_hash_equality!(hash_eq_32, 32, $type, false);

                test_hash_equality!(hash_eq_8_dct, 8, $type, true);
                test_hash_equality!(hash_eq_16_dct, 16, $type, true);
                test_hash_equality!(hash_eq_32_dct, 32, $type, true);
            }
        };
    }

    test_hash_type!(Mean, mean);
    test_hash_type!(Median, median);
    test_hash_type!(Blockhash, blockhash);
    test_hash_type!(Gradient, gradient);
    test_hash_type!(DoubleGradient, dbl_gradient);
    test_hash_type!(VertGradient, vert_gradient);

    #[test]
    fn size() {
        let test_img = gen_test_img(1024, 1024);
        let hasher = HasherConfig::new()
            .hash_alg(HashAlg::Mean)
            .hash_size(32, 32)
            .to_hasher();
        let hash = hasher.hash_image(&test_img);
        assert_eq!(32 * 32 / 8, hash.as_bytes().len());
    }

    #[test]
    fn base64_encoding_decoding() {
        let test_img = gen_test_img(1024, 1024);
        let hasher = HasherConfig::new()
            .hash_alg(HashAlg::Mean)
            .hash_size(32, 32)
            .to_hasher();
        let hash1 = hasher.hash_image(&test_img);

        let base64_string = hash1.to_base64();
        let decoded_result = ImageHash::from_base64(&*base64_string);

        assert_eq!(decoded_result.unwrap(), hash1);
    }

    #[test]
    fn test_hash_stability() {
        let image_buffer = gen_test_img(256, 256);

        let mut expected_hashes: Vec<(HashAlg, u32, String, String)> = vec![];
        let data = &[
            (HashAlg::Blockhash, 8, "3M0FbCH1SXo"),
            (HashAlg::Blockhash, 16, "Rmtwt5NMuVGZAHNooZz9P9IYk8R/t5BcqAnrPgTZ9pY"),
            (HashAlg::Blockhash, 32, "+JhF7rcRfDgFvjlWJBsrVxfiaTYPDVOQ509GlqOFS7dL0gFG+1tsEIc2YaxOjy2/UU7QWqXdmZaTiZ0GUt37CqRmUrOM8spCTlc2pHfVukj+Dy1JzuedykxTZBgEh/si2OhlgqulS0TKOvRP6twsFxJrVwUVaMD+gh2+J3zecM4"),
            (HashAlg::Blockhash, 64, "VVTBXKcKnFdaqibDc/X7xH8HZs/3W0SDJE0DI1BZyJ0zz7Xv4kg9sych6keNnokFcJyBuN+Xb4qkCs9lUYwcM7ZOh9SqOCkNj5GseqYmPL0dcOIyBD5Tz9+Fu4pXpoSVemr8PRp1r5dh2F/iaIC8CaeQE5hNAiKbSc8u5RNxPc+Pog2vrMRdczJ51nKRoJ7ajk3OV7IMBAl/9kZoT/UCYnHSFIsBNFKYJ0BeK1va9Fr/uLda0Dx3y+wXxUhpid0lSSZWsnC7fDvtG/g8DpOUwTaA42DDmyzBJ93129GgOsOPRaKodH9+kFlTbYqHgkEJRgFS387Wtphss/biT5/oFbCFZhwJF2tcPOx+mER0KZP2yRlHJfNMLBHiT/6s9G+i5yVdIjpPYk7oeJdmrOVwbjorItHOLWR5d6UTc4DGmpK8rLVB4phTkIen5wS9hWMm1XCxbOHJivCc0bt8Y6WuppjayHqU5tmHqjIXGxIfWlc1CS+wr+21SViRt8DtcQxaNGWcqmFtBuLFesLap2jOwe3qrqQnZCRheagz2YV0On5KgtUONP53YHz0XFUE430ZpATg4tE2IUPZ6lHX3mq/jYxWVZt+fjdADoduNDwboxKwy8Catfj0NhsnBDxK00fvBIX0UZT6WQxN54dK5Mc3baIuR3MkFn7asDvUx+M6nVA"),
            (HashAlg::Gradient, 8, "VCpSJptCLTU"),
            (HashAlg::Gradient, 16, "qjWoVkjWzBgmkpgmOJuomE3OjWsaGt12toyknmQYIcM"),
            (HashAlg::Gradient, 32, "jcwlxolJLwtOWy0bm+y4MagpNTM1k1mRkmBKmlCQUpMtSYRTrQ0t2zaZOCWh5aSEri3XmvrWlNbMtEVpy9marNOR3rVVUnpwZVubtatrTU5LhoxkmhLFmrFBkxui4hooJK1kyFRWaWNxVNTiNm5Gnc01YUFRJ2qtxmpLNpJMViM"),
            (HashAlg::Gradient, 64, "qutqsiVe2hakxNqpmVLaNG3ZYyaTFXOJi6SFJVpNZpfVIIpVWUndlSBZQ2KlgouUm+VWa9RSqUpYpWGkTIWSnkmR6U5Ltq2lU1qDyLPSEqemnlSzphJL40S2zaJNM7NVlSGPuqxsq0q8yiFpNtWWzFptME20ZrPZMmfHtphSjdYyWaZzJrQYMYxJZZpUzYay5CpjC5m0hG1TWWWmYpQGqRspo6XINlKcELqrlYliO21GRVMsWZaazZZFVSQp4lDNtpkhRTrJJqt2maoNZ6mc7kntTDZtUSr10+U6qTVSs7PQsrWqM1su3GK6c0xzUmFMybCJ0bpsM4yWqxFZoqWs5Epak9koSTPWkxI2a2aroElbZS4ppVJmvVKSKGssW0UztGKKRYxBc0Z3rEpypXUabcpUpXpjeaRNitGpWExrSpLGEq/UWmVRmcVSZJbh9HKTysYWJ3VNrubYSyuXMtWKIEdsJjnZskzlbYiySZpFW5O0dpHO7OKWTLzMytrkTGJD6whv1bAmUus6RXNNli6VUHY3VraiNJcttFtadk4buzIlmWljUiuFVksbZskyoyS9KZIxuVCyseUVtqhabxqByU3Ma2kzGyslUTOzRpLIqroeNWpuyjazuiaTVLZkk1EjklRZmN09SK2TNDJZW1URP0yzmCi2Ww/bGc1qcWStycw"),
            (HashAlg::Mean, 8, "eQ9N/ff/eOs"),
            (HashAlg::Mean, 16, "V2fQdN+UuXNcEnMs4bLRu17cG8f/85r96InpPsE7Ls4"),
            (HashAlg::Mean, 32, "OLlvnDeTfB4dvjl+BLk5I1NzYSdrpnOzp8+ekOOjBx/Y0gjX+ntIAkU/cSwHj609HHqevoWNuZeTOc9LFvO/7awn0PX4hf5Dj7IWvRbPmlD3n7lJtW+ev+5XMP9Nh/8qWfjBgMvtS0bHifDHydwMGRL5R0eGaMD9nQXa5PyZfOY"),
            (HashAlg::Mean, 64, "VVfVZ+0o+HdImRdPs9fyZZ4Xx2x+P0YTdk8PY9BZbLezx3WP8hiYF08C/G+LH581JMiF2sUXO043CM/pmQ0iPJMng9kCKr0FvjQOO+eWPv7NPY0mBT5fjtttuk7ZJibVfk7/dM1Bgs8jsF/ybIEUyDfYZdotBKaTTf4efTF1nZfN8g3O7KA9dTlyyTPFiZ7yiV2KPvAlCGnsvs8MzvkfIHTSVIuPPNSYJ3B2L5N+ckmdgP5K07wziXkTrWjjgMEJSQPummC7fL/NAAQ/TmOczzPQ0GRIwy7BJ4vz0mFUPuOHReMeZ/L8+M/gR5nGwsNJF0cHp75J9vgoVj/yT+/o/bC3J75RU3eeZOx1igzQJ9v0y1lT7fNuIITlS9JI/w8w688QjTgf4kbc6Z9GSsXwTRA57HTm4KFZdwfTsdhjWJK5//+BV8RDsZvv3waHxWemNZ19bOrLjs6xl38+ZYG855zYTXKSpl3P3zM9kwAf+zdpzSeQ3f+9WWORFYDvXUbGFhFQvyFtAIjFmuSbrmi90cnurMSuYRRhL7b//Is2OnReJsPGNv6P9Jf0TINE5n3wUgTj4/Emg4H7eDG13DI/gZgSVut3Mjt8jubn/bbbJiI4aoDYlGD2sB0mmCxAFvNv5QGyEZX5GYj856dS1kMxfLBuE2NgQzja8LvR5+k/n5w"),
            (HashAlg::Median, 8, "eQ9N/ff/eOs"),
            (HashAlg::Median, 16, "V2fQdN+UuXNcEnMs4bLRu17cG8f/85r96InpPsE7Ls4"),
            (HashAlg::Median, 32, "OLlvnDeTfB4dvjl+BLk5I1NzYSdrpnOzp8+ekOOjBx/Y0gjX+ntIAkU/cSwHj609HHqevoWNuZeTOc9LFvO/7awn0PX4hf5Dj7IWvRbPmlD3n7lJtW+ev+5XMP9Nh/8qWfjBgMvtS0bHifDHydwMGRL5R0eGaMD9nQXa5PyZfOY"),
            (HashAlg::Median, 64, "VVfVZ+0o+HdImRdPs9fyZZ4Xx2x+P0YTdk8PY9BZbLezx3WP8hiYF08C/G+LH581JMiF2sUXO043CM/pmQ0iPJMng9kCKr0FvjQOO+eWPv7NPY0mBT5fjtttuk7ZJibVfk7/dM1Bgs8jsF/ybIEUyDfYZdotBKaTTf4efTF1nZfN8g3O7KA9dTlyyTPFiZ7yiV2KPvAlCGnsvs8MzvkfIHTSVIuPPNSYJ3B2L5N+ckmdgP5K07wziXkTrWjjgMEJSQPummC7fL/NAAQ/TmOczzPQ0GRIwy7BJ4vz0mFUPuOHReMeZ/L8+M/gR5nGwsNJF0cHp75J9vgoVj/yT+/o/bC3J75RU3eeZOx1igzQJ9v0y1lT7fNuIITlS9JI/w8w688QjTgf4kbc6Z9GSsXwTRA57HTm4KFZdwfTsdhjWJK5//+BV8RDsZvv3waHxWemNZ19bOrLjs6xl38+ZYG855zYTXKSpl3P3zM9kwAf+zdpzSeQ3f+9WWORFYDvXUbGFhFQvyFtAIjFmuSbrmi90cnurMSuYRRhL7b//Is2OnReJsPGNv6P9Jf0TINE5n3wUgTj4/Emg4H7eDG13DI/gZgSVut3Mjt8jubn/bbbJiI4aoDYlGD2sB0mmCxAFvNv5QGyEZX5GYj856dS1kMxfLBuE2NgQzja8LvR5+k/n5w"),
            (HashAlg::DoubleGradient, 8, "EERhBSI"),
            (HashAlg::DoubleGradient, 16, "VCpCVqqSai21lZsbJimEbSSN"),
            (HashAlg::DoubleGradient, 32, "qjWoVmzWzGgmmYwmGCqqmEnKTUeIKkwb1Wa0jKSaYA0hw1ZStKGKxs611qS2hkiWq1RDCkRqJaKRrbY0azQ0ZqXk6tQ"),
            (HashAlg::DoubleGradient, 64, "ncxlxolZLwtOWy0Lm8yYMagtPTM1kVmRknFKmtC00pNl2cJTrQ2t23KdaSekpbSliSXXmqIW25LIlGVDyZ3FKcuZmrVX1mxwRFhaNSdLybRL7s5MSpKMJJISxZqxYZodpuI4qGStZMtRVmljcUTU5jZuBp3NNWFhUyZqrcZqS3aSTFYjaRebIlllThOpygyqmstTSYnpmlF5qViT3JFVrtoyW4dkNTjSrUZ0UrOWNWoypz3TqmlNzdLQMpFOSTIZpcwmilYm2JhaM5VadVziWmktrcy1WZZKFdpK5kWj2pIZKqlTpEwZdN9kWYrLODnSyLZWMnJmdq2ajDapS8Z2cqooSVOemLYl"),
            (HashAlg::VertGradient, 8, "lJsTJQiMJJw"),
            (HashAlg::VertGradient, 16, "VlaQwYrGzrWWpDSWSBSjTERKLKORqZIdazQkJqTk6tQ"),
            (HashAlg::VertGradient, 32, "aTebIhllThKpyizqmklTSYnpGtFcoVmrnNBZpnIyW8MsZzjSqcY0UjK2PWomLQ2VilEmzVpbMpGuTSYZ5cbGiloy3Fh0W4daaW0qzLEpnUoV20JGRYfKhhkiqZK8Xol032RZLss4OZLKlnQScGZWrJqMZqlLzjZyqipJU56YtiU"),
            (HashAlg::VertGradient, 64, "1DYyEcympk7SLRMm27rcqjapJC2lEj2LW69qXJ0ZxKrirGmy1JScZWwNzebSbM62lM+MOJtmWrrKpCKrpJnXsY2G8qye6pSxisjN5Vatm+eqSVJdySE1amVmImYTV8JksbMJtiWVqNSb0pWVpaG2pi5qCzqLWbcmqTplhhVJprJksuVupVReaNGUNjXsaEp7W9kstag1IqtX3WRKJxWlqolOOaWSpMxZCW8tTGElRxpKGrVMZ5lZ8VpuawaySk+TzV2ZOaq1o1gVs6ptlZpb9ZpGklE5r8XKq+RWmZImiUm4WVPVRi6pElZbWaN0TmXJMpZoVidXTuXtuswkiTFp4XFnGa6maWWViVU1CibZlhUyKWlZMumSE1quTGtaQobOa2UyrNxYWpIrubIsVldTcyaV5Y1mZqasNI1bM+pSSrIzTUXxHKsUm5Oi2snONomqE3EeVJ5ZrKRSSqtlyraYkqtTrklH+s7J0sauSSZKlZQlT5ZrRVGRF3XZdFImSzWbu/JmNilZ26xaIz0ljaPKrNY16SpVQ5LijFjhyuYWmliJ0zzpkCyvVKWklGYRS6VmhfLevpqbqk2dbEtZKCVLxotkMnQ6HXPMKmvVpFma6GpFaVjZGBZKrSIZE3kxVmWtVm3YTMU0mbVV6ZVER6UXpTWNlVM5V6stJVOtQVXVetY"),

        ];
        for (alg, size, hash) in data.iter() {
            let current_hash = HasherConfig::new()
                .hash_alg(*alg)
                .hash_size(*size, *size)
                .to_hasher()
                .hash_image(&image_buffer)
                .to_base64();

            expected_hashes.push((*alg, *size, hash.to_string(), current_hash));
        }

        // Helper to regenerate the expected hashes
        // for (alg, size, expected, current) in expected_hashes.iter() {
        //     println!("(HashAlg::{alg:?}, {size}, \"{current}\"),");
        // }

        let diff_hashes: Vec<_> = expected_hashes
            .into_iter()
            .filter(|(_, _, expected, current)| expected != current)
            .collect();

        if !diff_hashes.is_empty() {
            let diff_number = diff_hashes.len();
            let mut diff_str = String::new();
            for (alg, size, expected, current) in diff_hashes {
                diff_str.push_str(&format!(
                    "{alg:?} {size}x{size}: expected {expected}, got {current}\n"
                ));
            }
            panic!("{diff_number} hashes did not match:\n{}", diff_str);
        }
    }
}
