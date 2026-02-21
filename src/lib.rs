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
use image::{GrayImage, imageops};
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
    /// * <http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html>
    ///   Krawetz describes a "pHash" algorithm which is equivalent to Mean + DCT preprocessing here.
    ///   However there is nothing to say that DCT preprocessing cannot compose with other hash
    ///   algorithms; Gradient + DCT might well perform better in some aspects.
    /// * <https://en.wikipedia.org/wiki/Discrete_cosine_transform>
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
    /// * <https://en.wikipedia.org/wiki/Difference_of_Gaussians>
    /// * <http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm>
    ///   (Difference of Gaussians is an approximation of a Laplacian of Gaussian filter)
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
            .field("resize_filter", &debug_filter_type(self.resize_filter))
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

impl<I: Image> CowImage<'_, I> {
    fn to_grayscale(&self) -> Cow<'_, GrayImage> {
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
            let img_vals =
                resize_image(img, dct_ctxt.width(), dct_ctxt.height(), self.resize_filter);
            let input_len = img_vals.len() + dct_ctxt.required_scratch();

            let mut vals_with_scratch = Vec::with_capacity(input_len);

            // put the image values in [..width * height] and provide scratch space
            vals_with_scratch.extend(img_vals.into_iter().map(|x| x as f32));
            // TODO: compare with `.set_len()`
            vals_with_scratch.resize(input_len, 0.);

            let hash_vals = dct_ctxt.dct_2d(vals_with_scratch);
            HashVals::Floats(dct_ctxt.crop_2d(hash_vals))
        } else {
            let img_vals = resize_image(img, width, height, self.resize_filter);
            HashVals::Bytes(img_vals)
        }
    }
}

#[cfg(feature = "fast_resize_unstable")]
fn resize_image(img: &GrayImage, width: u32, height: u32, filter: FilterType) -> Vec<u8> {
    use image::DynamicImage;
    let src_image: DynamicImage = img.clone().into();
    let mut dst_image = DynamicImage::ImageLuma8(GrayImage::new(width, height));

    use fast_image_resize::{ResizeAlg, ResizeOptions, Resizer};

    let mut resizer = Resizer::new();
    let resize_alg = match filter {
        FilterType::Nearest => ResizeAlg::Nearest,
        FilterType::Triangle => ResizeAlg::Convolution(fast_image_resize::FilterType::Bilinear),
        FilterType::CatmullRom => ResizeAlg::Convolution(fast_image_resize::FilterType::CatmullRom),
        FilterType::Gaussian => ResizeAlg::Convolution(fast_image_resize::FilterType::Gaussian),
        FilterType::Lanczos3 => ResizeAlg::Convolution(fast_image_resize::FilterType::Lanczos3),
    };

    let resize_options = ResizeOptions::new().resize_alg(resize_alg);
    // TODO - probably here this should be handled better, than simple fallback to imageops::resize
    if resizer
        .resize(&src_image, &mut dst_image, Some(&resize_options))
        .is_err()
    {
        return imageops::resize(img, width, height, filter).to_vec();
    }

    dst_image.into_bytes()
}
#[cfg(not(feature = "fast_resize_unstable"))]
fn resize_image(img: &GrayImage, width: u32, height: u32, filter: FilterType) -> Vec<u8> {
    imageops::resize(img, width, height, filter).to_vec()
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
    /// Input bytes slice was empty.
    BytesEmpty,
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
    /// Returns `InvalidBytesError::BytesEmpty` if the input slice is empty.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, InvalidBytesError> {
        if bytes.is_empty() {
            return Err(InvalidBytesError::BytesEmpty);
        }

        if bytes.len() * 8 > B::max_bits() {
            return Err(InvalidBytesError::BytesWrongLength {
                expected: B::max_bits() / 8,
                found: bytes.len(),
            });
        }

        Ok(Self {
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
    pub fn from_base64(encoded_hash: &str) -> Result<Self, InvalidBytesError> {
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

/// Provide Serde a typedef for `image::FilterType`: <https://serde.rs/remote-derive.html>
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

fn debug_filter_type(ft: FilterType) -> &'static str {
    use FilterType::{CatmullRom, Gaussian, Lanczos3, Nearest, Triangle};

    match ft {
        Triangle => "Triangle",
        Nearest => "Nearest",
        CatmullRom => "CatmullRom",
        Lanczos3 => "Lanczos3",
        Gaussian => "Gaussian",
    }
}

#[cfg(test)]
#[expect(clippy::print_stdout)]
mod test {
    use std::time::Instant;

    use base64::Engine;
    use hamming_bitwise_fast::hamming_bitwise_fast;
    use image::imageops::FilterType;
    use image::{ImageBuffer, Rgba};
    use rand::rngs::Xoshiro256PlusPlus;
    use rand::{Rng, SeedableRng};

    use super::{HashAlg, HasherConfig, ImageHash, InvalidBytesError};

    type RgbaBuf = ImageBuffer<Rgba<u8>, Vec<u8>>;

    #[expect(clippy::uninit_vec)] // Safe because we immediately fill the buffer.
    fn gen_test_img(width: u32, height: u32) -> RgbaBuf {
        let len = (width * height * 4) as usize;
        let mut buf = Vec::with_capacity(len);
        // SAFETY:
        // We immediately fill the buffer.
        unsafe {
            buf.set_len(len);
        }
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xc0ffee);
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
        let decoded_result = ImageHash::from_base64(&base64_string);

        assert_eq!(decoded_result.unwrap(), hash1);

        // Test empty string input
        let empty_result = ImageHash::<Box<[u8]>>::from_base64("");
        assert_eq!(empty_result, Err(InvalidBytesError::BytesEmpty));

        // Test invalid base64
        let invalid_result = ImageHash::<Box<[u8]>>::from_base64("??");
        assert!(matches!(invalid_result, Err(InvalidBytesError::Base64(_))));
    }

    #[test]
    fn test_hash_stability() {
        let image_buffer = gen_test_img(256, 256);

        let mut expected_hashes: Vec<(FilterType, HashAlg, u32, String, String)> = Vec::new();
        #[cfg(not(feature = "fast_resize_unstable"))]
        #[rustfmt::skip]
        let data = &[
            (FilterType::Lanczos3, HashAlg::Blockhash, 8, "zcYPaHZw0Pg"),
            (FilterType::Lanczos3, HashAlg::Blockhash, 16, "lbH1ZDXhHHL3ke8AoJrlNl42nU8B3EwXeDpB08Cj2ts"),
            (FilterType::Lanczos3, HashAlg::Blockhash, 32, "I88CzyWTU+m6o7S8M7xAnDu7J3kRjgep2Kcq29ZaJAYq3RVjeS8ihhdtGEr3ZdWogMjtwSDWzRm1PjjLM/q9DTwiqIvHN2wU44duIdVq+RuDEtDz5jyJq+2fDhFxELfVRLdubZME6lsVMQdngqov85XgD0hFNhlPqAP5Y8z6bfk"),
            (FilterType::Lanczos3, HashAlg::Blockhash, 64, "H4yrlDSQvVNuFfTwjhlBdjcUHQNPcuq5aspvYys2Q/TMx+/s0C9Ex8KsGl9I77CJlwPW7Fgx1AeFjbX1Fgkq8sOr/uUFJMkvNgVKyvKMkiSbKPlNFmW1yExLPuJ3G8bdA3fqjmAq3ddY8i7s7FyGGxbf3gOQo38eHmGd+FIMoYCOGhGTXSWVtLlY9vYbSUVxVDteLkWaT5Dbrq6EbA56VQZKoizwG8Yguyf3JQXVnVFHtWIZM9Jjb321Vd97IYBYMOSw6ON8QoTTx8yvuf+n4IBMPuMC6OsTEwR7QfSV61cTUtmPZKvt+GC/RQbpnqliB2fFweKl46EKC9T1U9fiJOgYqQzDTc/Isu9W6hbk2WB59o0vcX2VBY+0PidRGHiGD2Zj6rSdZ4kl7tbIX5jAHBpXYJqCScXTouotPNn/CpanYAYNDdsOz5tyaAAit7s7HF6WN4dJa4mb/mlJKdUpnvK0tLfQWIMDl7bOUj0qRwNZrzBJH425Kws1Awh/d1uxGXC75nwffp4wGE+nnhDTOJpQOEy0/O5zrlkaJo1hY2R7WoFWPsB9MHcwRS3xFkmtWG70rr2ZCuwZAkzEPCbf31PDoPb+FJkZGh463TuWYuSw1HKCynNqand6+J2DQCexEIxHBOb7Cl3n+oj5g7AvmsH0x60mdOHOpWDNo9C1FfE"),
            (FilterType::Lanczos3, HashAlg::Gradient, 8, "ykchMjEySog"),
            (FilterType::Lanczos3, HashAlg::Gradient, 16, "iKCqNJpYZ5JTMoZKjEq6sjWcRmeUToMtmC2gSWgprc0"),
            (FilterType::Lanczos3, HashAlg::Gradient, 32, "kFjqaurZWm5EkZSOjcpKWs3FkLZIU0SpLEkuSSlJEqerbs4kLVWKU1nmrQlpqVykTtTmuNRCpprKxqUqWMks7QajluuznFaWWWElPXQ0O6tQjV1ZE8uMtWaLsoyKvJG0+lMjLkhJowSaXNdyatUkicpeViWpqFrNVK1lKUbJcWw"),
            (FilterType::Lanczos3, HashAlg::Gradient, 64, "Y8ekViRlyumZXNNaxqhMJJhZk2tmU6qtlfWmWAkrybRg/DRFMenJ5XV31m2zVZkJymVZd0qdxrLCxMpggo2sJbJUY/CVtGkmwrNxJbQqiiaxI5VmrRaz4pIVrTJyqeZsrDnJVnSp5FYkm9QWrCRn2aXJZnhasStbjZ2kJVnnrcRVpdhYzbT91FYsaVM9pVJRaR0pNXjZLZylV8aypmQDxaMlWXKmxWJUndBUnDnMaFW7MtWFaep0ZYTawkrdJbNOqfRUzimppvTNcCbXqNigVmVnt1lM1CzTscIcVGuSpMWk4WjFKspsbJRLpvFtTEhqbLKiajdkYk2o1eyimFVislqFVcbwTuesmyVTLTZmTDMWy0LRmHXaZJVZVZNaJr5mJjO1NSaOM2WiMyssZVpFtJUtJp2WbE9MU9piVlJJadqRdqWFxUnWym1T0lKpUau5liVLNmlkK5k2l6ykpNquTynJikI6TZFKWiJVRhrzo12sWLqtlGkZi6yS2a0ndZmppbNdcxZV2oxMLSazxlbbqlUtViaZTiYi0WRNGuVke1Qs7exTF9MclUmZEneRksyoqS0yt9RtQ+q2cSzTkpfR2YxZXCub0lmYxc3JtpLaMqpMasnEJjkqqqucpMpubaOpibcihqJJY0VwOWUsU5thm6pq4rCRNdke2DTCwBqWpDI"),
            (FilterType::Lanczos3, HashAlg::Mean, 8, "n9wPdO9nUJg"),
            (FilterType::Lanczos3, HashAlg::Mean, 16, "E+395TXz/PbnMD8dsFolP2442WzzHH5rMWvDS8B7X8k"),
            (FilterType::Lanczos3, HashAlg::Mean, 32, "L7PKzdWXE+yZ5728M7lT2huLB/QRnhy70Pt439baNA5+3BT9eu4CA/cNKxv/X9HhmIr9AYHfzj+VDSzuM5r5CXwopcvGP+S+8kdqYclocDsDGvFTZp65p/23phkQIaclADdmfJcajAn1saYAlKhP+xHxB2kCM9Kb+WvL69zyR+A"),
            (FilterType::Lanczos3, HashAlg::Mean, 64, "H4yrvUzA3uv/sY7FjLhRYHe3PsfN9gi9Y+s95zsC2/TPw2/cdIvZhcPuHJ1mf7F7Fdsy79oxjiOdibzXHoGo58Wr3uUnBcFuHSZGyugMGuYHRrhNenWnzwQ72uPmA87NAHOyz8CL7ffJdo/92H1Pm1rz3eMQL38SOjMJa3IO+YyuWJHzmiHx3bhb0vZx63dX0nNeZlDDabAL/px1DAEOBQZao6Tsj84UeqftIXOcSVFG5awZ0/tnbT+1H5IbZwBcQMXxsAPghISS502v6b8DxsLIbrd7vGmTDwVxYdCyb8d7w9nI9YrIaCH3bAPLydhqn2zMQ+7pZglXD9nlE+9CoPAYqZzhWc+o9Ov2GmzEX2JpsMWXMe7eRy+wvDfUbLDnXGZvdWwcM20F799Bxx7ANGPbSD6gCd9d7vDH/E/fW5anSAgNBfue2xMA7AMjVwszfA62fMNMb7Do/DnJqZ8vzvOzP4942IPL9d/u3Hzg50FZ42SbP095iwM1Bwtv7BuoCkS75nyw8p0bUEwGnFTaP7tD7ECy3KoGr0neJ8xBYkR72gPWfPQxATIyZezyJom7QnvEprXZX+5to1iENDTD2xGjwf7+EZMYCxgD7TeWYy6ZgZOBSHPuK14xyZ9NQSbhEg7HDG/fTl/n8sp5VrJPuFTdjac/MMD8semMr3C+iPc"),
            (FilterType::Lanczos3, HashAlg::Median, 8, "n9wPdO9nUJg"),
            (FilterType::Lanczos3, HashAlg::Median, 16, "E+395TXz/PbnMD8dsFolP2442WzzHH5rMWvDS8B7X8k"),
            (FilterType::Lanczos3, HashAlg::Median, 32, "L7PKzdWXE+yZ5728M7lT2huLB/QRnhy70Pt439baNA5+3BT9eu4CA/cNKxv/X9HhmIr9AYHfzj+VDSzuM5r5CXwopcvGP+S+8kdqYclocDsDGvFTZp65p/23phkQIaclADdmfJcajAn1saYAlKhP+xHxB2kCM9Kb+WvL69zyR+A"),
            (FilterType::Lanczos3, HashAlg::Median, 64, "H4yrvUzA3uv/sY7FjLhRYHe3PsfN9gi9Y+s95zsC2/TPw2/cdIvZhcPuHJ1mf7F7Fdsy79oxjiOdibzXHoGo58Wr3uUnBcFuHSZGyugMGuYHRrhNenWnzwQ72uPmA87NAHOyz8CL7ffJdo/92H1Pm1rz3eMQL38SOjMJa3IO+YyuWJHzmiHx3bhb0vZx63dX0nNeZlDDabAL/px1DAEOBQZao6Tsj84UeqftIXOcSVFG5awZ0/tnbT+1H5IbZwBcQMXxsAPghISS502v6b8DxsLIbrd7vGmTDwVxYdCyb8d7w9nI9YrIaCH3bAPLydhqn2zMQ+7pZglXD9nlE+9CoPAYqZzhWc+o9Ov2GmzEX2JpsMWXMe7eRy+wvDfUbLDnXGZvdWwcM20F799Bxx7ANGPbSD6gCd9d7vDH/E/fW5anSAgNBfue2xMA7AMjVwszfA62fMNMb7Do/DnJqZ8vzvOzP4942IPL9d/u3Hzg50FZ42SbP095iwM1Bwtv7BuoCkS75nyw8p0bUEwGnFTaP7tD7ECy3KoGr0neJ8xBYkR72gPWfPQxATIyZezyJom7QnvEprXZX+5to1iENDTD2xGjwf7+EZMYCxgD7TeWYy6ZgZOBSHPuK14xyZ9NQSbhEg7HDG/fTl/n8sp5VrJPuFTdjac/MMD8semMr3C+iPc"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 8, "mUUKgAI"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 16, "ysdFsnUyKg6IEpqUCeREFCmZ"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 32, "iKiqNpr4JpJRspZSjU6aKrOaRSVsRpEKry24KeBpaCmtzVoTjJSZpJWhNMlSVUOUjWOrplKtVTK1ozLWxkRFTVOrwqE"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 64, "kFjqSurZWm5kkZSOjVpKW83NkLZI00WpLEkOySlJEqOpaoqkLVXSURlGqQlpqaWsSqJeuETdppDQZoaqSM2kbA6rFOkjjdbKuVA1lVhkLSlkvDm7WclMSRfLprROoZKMmpiRNnpRIy5ISecEulTWUmrXIInKXlYlqalazVStZSlGyXFsUGdOU8wmGivtJhGn6pUpYi9FZMqESzZmZUEhZHNfZ+lqLqMUaVUXYVGlMikKM1VpZFY5sjdVq4XipWxeJNmm1oncOIzPVtZMRBJrlJt2cVe0cpYrc+6aNmZSrsgUzYo0qhU5QjjTGHpms40kanSrGLIyRSmzEWutmSSuTaxlls2ZpLus"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 8, "EpqUieQUCYk"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 16, "WhOIlJmklakgxVtUhWmtIjusVz3VIjCD1kZFTVOpoqk"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 32, "UCdOU8wmGiPtpjGn6pVpYi9FdMqESzRmMUUhZGNeZ+1oJqMUaVU3qdGhMilOM1VrJVS7kjKNahTilKZeLdk0lMnGmE5EMuJUWxRh1bJytmsx6J4ma9KqxHTcim2KVXkiONcYWjazjSZqNKuQsnJFKbMRaa2ZJK9NrCWWzZmku6w"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 64, "ihBVWiZRybKRcjZNZpkmG2ijKG6btO0qSLlNWo5m1SJheVN4ShsV1XfVvZhhtpa8mRSyVMWsVCyZmDGzySintqlsUWmpeuzFt00SUTtWtThSm200mqYYV5TmTWaxpkraszRJtmaOKiuVTTapSuoyrZcbLbOaJEWulTi1Urc8l3Vku9zN3s44W6mVS1Kb1ogWxxRWMhtTIlrGNjarLB3HrpIqwe4WVo0mVsuepgkrJS+uJZNrMiJrHGys1U22si0lV6o0M5JNimYjrWoSR8lJzSoaY1PEkllICFsqzWlktKlc2uVKZTacraSUisZs1ClpZSXRappmlSGkiM3SVBrZZsbUsdq2bGZZ1+a6dpRTtnnbFC5emfTZOSWTbBPZTMpNZTJyRUY8kanURZlqWxu1USpPOzVdloq1KY1OqS2mIrk7pmjpdmlslRIta93WSjmWmrJKVaLqstoZXQka3Exm2jMSG5GUyFeVE7ppZZakmDaaG7NipXJOT0lum1NTjVSTza46k1mr2qqmM00tWZNqEiifrWUltVaOZi2Gl2Nmt5RceSW152jIZFUqaCbNJlUnzc4oNnE1STauzsqaJRjhREqWpSYrMdmMHJupEsNI85LjzGypZixF09LWZAp9raFWlr1bUnm0sTDxiXKSDcOy6tTKeZUOnWVKTsu2xazsUK4"),
            (FilterType::Triangle, HashAlg::Blockhash, 8, "zcYPaHZw0Pg"),
            (FilterType::Triangle, HashAlg::Blockhash, 16, "lbH1ZDXhHHL3ke8AoJrlNl42nU8B3EwXeDpB08Cj2ts"),
            (FilterType::Triangle, HashAlg::Blockhash, 32, "I88CzyWTU+m6o7S8M7xAnDu7J3kRjgep2Kcq29ZaJAYq3RVjeS8ihhdtGEr3ZdWogMjtwSDWzRm1PjjLM/q9DTwiqIvHN2wU44duIdVq+RuDEtDz5jyJq+2fDhFxELfVRLdubZME6lsVMQdngqov85XgD0hFNhlPqAP5Y8z6bfk"),
            (FilterType::Triangle, HashAlg::Blockhash, 64, "H4yrlDSQvVNuFfTwjhlBdjcUHQNPcuq5aspvYys2Q/TMx+/s0C9Ex8KsGl9I77CJlwPW7Fgx1AeFjbX1Fgkq8sOr/uUFJMkvNgVKyvKMkiSbKPlNFmW1yExLPuJ3G8bdA3fqjmAq3ddY8i7s7FyGGxbf3gOQo38eHmGd+FIMoYCOGhGTXSWVtLlY9vYbSUVxVDteLkWaT5Dbrq6EbA56VQZKoizwG8Yguyf3JQXVnVFHtWIZM9Jjb321Vd97IYBYMOSw6ON8QoTTx8yvuf+n4IBMPuMC6OsTEwR7QfSV61cTUtmPZKvt+GC/RQbpnqliB2fFweKl46EKC9T1U9fiJOgYqQzDTc/Isu9W6hbk2WB59o0vcX2VBY+0PidRGHiGD2Zj6rSdZ4kl7tbIX5jAHBpXYJqCScXTouotPNn/CpanYAYNDdsOz5tyaAAit7s7HF6WN4dJa4mb/mlJKdUpnvK0tLfQWIMDl7bOUj0qRwNZrzBJH425Kws1Awh/d1uxGXC75nwffp4wGE+nnhDTOJpQOEy0/O5zrlkaJo1hY2R7WoFWPsB9MHcwRS3xFkmtWG70rr2ZCuwZAkzEPCbf31PDoPb+FJkZGh463TuWYuSw1HKCynNqand6+J2DQCexEIxHBOb7Cl3n+oj5g7AvmsH0x60mdOHOpWDNo9C1FfE"),
            (FilterType::Triangle, HashAlg::Gradient, 8, "wkcAMjEiDo4"),
            (FilterType::Triangle, HashAlg::Gradient, 16, "iKiKNLr4JpITEoRKikqaohUsRkaUBoctnC2oaWgopMw"),
            (FilterType::Triangle, HashAlg::Gradient, 32, "kEnqaurZWm5kkZSOzcgqW83FkLtIQ0SpLFkuyylLEqcpakolLFWKsRlGraloqdmkTtTumNRH5prIxqSqWM0kqQujtMuzmHaWWWAlPXQ0PStQiVxZEcuMlGaLsoyKrLGUelEjLkhZqyTKWNdSaNQkiahe1SWpqFLNVK3lOUbJcWw"),
            (FilterType::Triangle, HashAlg::Gradient, 64, "Y8akRiR1z6iZXMFaxqhINZhUk2tmW6qslfWmWAmrSbTw9DVHseHJxXF11mezZZkIykVYd0oVxjLC5Mphhq2sJPJUY/CUpGkmwrNxpbQqiiazIaVmrRKj6pIVrXJyqeZs7LlJU3Sp5lIkudVSriQnWa1JZFBaky9bhZykGVnnrdRXpdhYzbW9VFYsaVMcpRBVaR0pMzTpbJnlU1Y2pkRDxaMlGTKmxGZMnVBVtDnEbFWzWpWEactyZYbahorJJbFOqfRU3iitpsTNcCbTqNqgVnFmp1hN3iTTpOY8VGvapMGM4GhFKspsLJRLonFpTGxqbLOiajNkYk2o9eyqkHVjvBqFVcfwVuesmzVTzTZnTDMWzVLRmHXeZZdYVZNSJrhmJjOxNSeOM0WCMyt8ZVpNFZEtIpdWTc3MUVpiVlJJSdIRdqWFxUnGyilT09KpUaO5hydLNulkI5o2k6zspNuqxjnIikI6TbFK2mpXRBrzoU2sWLqtkmMZiazSma0iZ5mppbOZcxZVWIxNjSaz1lbbqlUkViKZTiJi0SRHEuVka1QN7exzF3IclQ2ZMneRksyqqS0yt9RNQuq2WWxzFpTQSYxZdCsT0FmbzM3JthKbcitMasnEIjkyqqucpcqubaOpKbcihqJZY0VwOeQsU5thn6pq4rCTtdEe2jTCkBqXhDI"),
            (FilterType::Triangle, HashAlg::Mean, 8, "39wPfO//EZg"),
            (FilterType::Triangle, HashAlg::Mean, 16, "V+395fXz/PbvoD8d8B8lP348/X7zHH57UWvBS8F738k"),
            (FilterType::Triangle, HashAlg::Mean, 32, "L7uKzdeXE+jZp7e8M78T3hubB+ARnh670Pt4/9LaNA9+2BT9e+4AA/cNOyv3X/HhmIv9gYHfzz+RHz3uM5r9C3wopM/Gf+S/8kdqccNjeDsDEvHTbr65t/23hplQMac1ATdGfBcejAnXsc4AlKjv+RHxB2kCc9Kd+Wvf8/zyR+A"),
            (FilterType::Triangle, HashAlg::Mean, 64, "H4yrvUzA3/n/uQ7BjPh5cHc7PseN/4i9Y+o/5zsC+/XHw1+c9wfZhcPuHJ8mf7F7l9sz79oxjiOdib7HH4G478Wr3+UnBcluDSfGyOoMmm4HRrrNfnWnzwRr2uPmA87tAHOyz+CL7/fIco/92H/u20Lz3eMQP38SOjOZa3IO+YS+GpHzmiHx3bhb0vZx63N/wnNeZlDDafEL/7h9DAEOBQ5e46Tsj84Re6ftMHOcS1VG5a0Z09tnbT+1n5IbZwBcQMHxsCPwBISS402v+f8DxsDIfrd7uGmTDwBxcdjy78c/gdnI7YrI6QH3bQPpjfhon27MQ+ft5gkHD9nlE/9GoPAYiJzhXc+o9O/2kmTEX2J4uM2XMe7eRS+wvDdUbvjnHOZ/dWwcN30F7t9hxj8APGf7SD7gC90f7/DH/Af/S5YnQAgPA/uenxoA7AOjVw8zPA6+fMPLb6Do/DnJqZ8vzvO3P4942IPL9f/u3Hzg50HZp2DbP0/5iwM1Bwtv7DuoAEA75jz48r0bUEwGnFDSP7tD7ECy2OoGr0veB8zBYwR72gPGfOQxAHJyZexyJwm6UnvE5rfZz+tto0iGNDTD+xGjwf7+EYMaAwAD7DfWYzqYgYOBSHPiKx5wz59NQSbhEw7HDO/fTl/n8sp5VrBPvPT/jac/MMD8MeGMr3y+iPc"),
            (FilterType::Triangle, HashAlg::Median, 8, "39wPfO//EZg"),
            (FilterType::Triangle, HashAlg::Median, 16, "V+395fXz/PbvoD8d8B8lP348/X7zHH57UWvBS8F738k"),
            (FilterType::Triangle, HashAlg::Median, 32, "L7uKzdeXE+jZp7e8M78T3hubB+ARnh670Pt4/9LaNA9+2BT9e+4AA/cNOyv3X/HhmIv9gYHfzz+RHz3uM5r9C3wopM/Gf+S/8kdqccNjeDsDEvHTbr65t/23hplQMac1ATdGfBcejAnXsc4AlKjv+RHxB2kCc9Kd+Wvf8/zyR+A"),
            (FilterType::Triangle, HashAlg::Median, 64, "H4yrvUzA3/n/uQ7BjPh5cHc7PseN/4i9Y+o/5zsC+/XHw1+c9wfZhcPuHJ8mf7F7l9sz79oxjiOdib7HH4G478Wr3+UnBcluDSfGyOoMmm4HRrrNfnWnzwRr2uPmA87tAHOyz+CL7/fIco/92H/u20Lz3eMQP38SOjOZa3IO+YS+GpHzmiHx3bhb0vZx63N/wnNeZlDDafEL/7h9DAEOBQ5e46Tsj84Re6ftMHOcS1VG5a0Z09tnbT+1n5IbZwBcQMHxsCPwBISS402v+f8DxsDIfrd7uGmTDwBxcdjy78c/gdnI7YrI6QH3bQPpjfhon27MQ+ft5gkHD9nlE/9GoPAYiJzhXc+o9O/2kmTEX2J4uM2XMe7eRS+wvDdUbvjnHOZ/dWwcN30F7t9hxj8APGf7SD7gC90f7/DH/Af/S5YnQAgPA/uenxoA7AOjVw8zPA6+fMPLb6Do/DnJqZ8vzvO3P4942IPL9f/u3Hzg50HZp2DbP0/5iwM1Bwtv7DuoAEA75jz48r0bUEwGnFDSP7tD7ECy2OoGr0veB8zBYwR72gPGfOQxAHJyZexyJwm6UnvE5rfZz+tto0iGNDTD+xGjwf7+EYMaAwAD7DfWYzqYgYOBSHPiKx5wz59NQSbhEw7HDO/fTl/n8sp5VrBPvPT/jac/MMD8MeGMr3y+iPc"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 8, "GUULgAM"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 16, "wsNEEjUyIw6qUoqUiKBEHAmZ"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 32, "iKiKIJrYNpIRMpZSjk6aK7OmBSdsRpUOry2YKeJpaCmkzFqTiJSJpJWlMMFCVUPUhWOpplCtVTK1IzAGxmBFTEOpQqU"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 64, "kEnqaurZWG5kkZSOzdoqS83FgDpIQ0WpLFkOiy1JEqMpagqlLFfWcBlGrSloqK2oSqJcuETdppjI5oaqWM2lbAq7FOkjrdbKuRA3lVhgLStkPDibUcncWRfLtpROofKMmpixNHpRIyxIS+ckqlTVemjWZImoXNUlqahKzVStZTlGyXFsUCfOU8wmGyPlphmn6sUp5q9FZMqEQzZmYUEhZHNXZ+VoLqdUaVUTYVmlMilKMVVrZFYxsibRipWypGwUZtmm1q3cPIxNVsZcRBJj1Jpzc1a2cpYrc6aeFmLWrkAUzIkkKhU5QjrTGHpmM41kYnSrOLIyRamzESktmSSmTazllsWZpLqs"),
            (FilterType::Triangle, HashAlg::VertGradient, 8, "UpKEgWQcCZk"),
            (FilterType::Triangle, HashAlg::VertGradient, 16, "WhOIlImklaEgxVMVhWCNIlGuVTXVIqSC1kZFTUOpQqE"),
            (FilterType::Triangle, HashAlg::VertGradient, 32, "UCdOU8QmGiPlpjnn6tUpYi9FZMqEQjZmYUUjZGNeZ+VoJqN0SVUz4dGhMiFOMVVrZVSpkjaVapTilC6WLdgk3MnGnMxEUuNcWxRh1bJylkMz4p4mY9KqxHScjkWKVSkiONcYWjazjWZidKs4snJFKbMRaS2ZJKdNrOWW1Zmkuqw"),
            (FilterType::Triangle, HashAlg::VertGradient, 64, "ghJVXi5ZzbORcjZNppkHG2ijKE7atc0qQOlNXk5m1SpheVF4CgMVRefQvZhhJJa8mRyyVGUsVCyZmBOzySwnNqlsU2mpeuzll00SUTkW1TiSm000mqZYVdLkTeYxpkpaMzRJtmaOOiuVTTapSm4yrZUbbbO6LGWslTi1E6cs1XWkudzdntq4U6mVS1Kb1ogWxxB2MhtTY1rWNnarKB2HrpYq4e4eRq0kRsuehg0vJS6OJZNqMiIrHGysVU1yli1ld6o0M5JNim5zrWoSV9nJ7WoaY1PElhlJDFtizW1EtKlM2mlKZTScLaQUi8Js1ClpZSXRahpmlSG0iMnSVBzZZObksdq0ZG5b1ea6dpRSdnnZFi5WmeTZOWWTbBOZTMhFZDJqR0c9kanERZlaUxu10apPuz0dlq6VKY1OKS2GIpk7pmipdEEolRItS93WSriWkjJKVabqnNoZXQkb3Exm2jMSm4OUyGcTE7t5ZZakmHaaE7NipfBOT0lsG0tTjVWTzao6k1mruiqkOw0tSZNqEiwfLeUlJWaOZi0Ol2FkEZRcOSW1x+jYRFUqKCbNJkUnxc6oNmE0STYuzoqaJRjhRAqWpScrMdmcvpqpAsNI85KnxGiqbqxl0tKWbAp97KE2lp1Skn2UsTBxiXKULdOy6lTKcZUOnWRKTku2xazIeKw"),
            (FilterType::Nearest, HashAlg::Blockhash, 8, "zcYPaHZw0Pg"),
            (FilterType::Nearest, HashAlg::Blockhash, 16, "lbH1ZDXhHHL3ke8AoJrlNl42nU8B3EwXeDpB08Cj2ts"),
            (FilterType::Nearest, HashAlg::Blockhash, 32, "I88CzyWTU+m6o7S8M7xAnDu7J3kRjgep2Kcq29ZaJAYq3RVjeS8ihhdtGEr3ZdWogMjtwSDWzRm1PjjLM/q9DTwiqIvHN2wU44duIdVq+RuDEtDz5jyJq+2fDhFxELfVRLdubZME6lsVMQdngqov85XgD0hFNhlPqAP5Y8z6bfk"),
            (FilterType::Nearest, HashAlg::Blockhash, 64, "H4yrlDSQvVNuFfTwjhlBdjcUHQNPcuq5aspvYys2Q/TMx+/s0C9Ex8KsGl9I77CJlwPW7Fgx1AeFjbX1Fgkq8sOr/uUFJMkvNgVKyvKMkiSbKPlNFmW1yExLPuJ3G8bdA3fqjmAq3ddY8i7s7FyGGxbf3gOQo38eHmGd+FIMoYCOGhGTXSWVtLlY9vYbSUVxVDteLkWaT5Dbrq6EbA56VQZKoizwG8Yguyf3JQXVnVFHtWIZM9Jjb321Vd97IYBYMOSw6ON8QoTTx8yvuf+n4IBMPuMC6OsTEwR7QfSV61cTUtmPZKvt+GC/RQbpnqliB2fFweKl46EKC9T1U9fiJOgYqQzDTc/Isu9W6hbk2WB59o0vcX2VBY+0PidRGHiGD2Zj6rSdZ4kl7tbIX5jAHBpXYJqCScXTouotPNn/CpanYAYNDdsOz5tyaAAit7s7HF6WN4dJa4mb/mlJKdUpnvK0tLfQWIMDl7bOUj0qRwNZrzBJH425Kws1Awh/d1uxGXC75nwffp4wGE+nnhDTOJpQOEy0/O5zrlkaJo1hY2R7WoFWPsB9MHcwRS3xFkmtWG70rr2ZCuwZAkzEPCbf31PDoPb+FJkZGh463TuWYuSw1HKCynNqand6+J2DQCexEIxHBOb7Cl3n+oj5g7AvmsH0x60mdOHOpWDNo9C1FfE"),
            (FilterType::Nearest, HashAlg::Gradient, 8, "xiVGUhUpVdk"),
            (FilterType::Nearest, HashAlg::Gradient, 16, "tGiWidqorTWmitLRDk0rl6dya1KlNtVqVVWijqq02vo"),
            (FilterType::Nearest, HashAlg::Gradient, 32, "VGryuLXKCqWFb1UW5hJbmZltWuor1YYu6pQmKSqtM7MqVduYUtWata+12isqq2YrZlXa1FqmVtWcjvBKSi7dtNxW1WqSnTKrTVMaNbtJKmOo7hhT1qrq0nYim6zZraVcS5OVq1ubqqTTiFRZUUyVlK2qlWpYpbrIe3aVStcppLQ"),
            (FilterType::Nearest, HashAlg::Gradient, 64, "cnK1KmUl1S5WqslSZisRVnMW52JrdcnsWLWQS4oqTaWRkiVVSbKpLC0Vp3ApKy3WTZmyBlJVwqRXdEqp10qsZrKWw1Y1bVvLlBItq6WM8yVUmyQvmcooq5UUrXJGreesrKpUW1UI93U1MZW2qqS0yblWlfXWuitRpVYtjZW0MZVhWdkVrTajDlGnKmcttavcYiyqWnG1UrxFWW6bKiyvlXMXqUJyS2tZvZN6rumkWIMzsVRUUZNmTZfc0oVKNbpq7bRWSiuZNlfNUyvNqsI2VVZni2o2STlVqjKLonVpYZTs0kplyeZqLZdKZVpqZq0+LDtWKa100myqrlqzkU4to71GTbfocM0IW1UoTbWjbNUWykRUWVU1pVZJVUlFsWolpjWpKKoMN20iRaIy815VnbQtsRVpTUsldJWqNApbtpJF1axP003nvCkl0mZ9daVpZ23LjGFKNWoqTSbLVJuG1CWVqmJrJlU0s6pNxnKaaS0rYqqkh1Zb88oW2afyM5kvlLqYVbSLcswapYaVTLSxoUqlVWurWErOUaRRq6txS1k1bcljM5vbirwvKlWqONWSqSV3tyxV47a0WFFaRrguTWhaxW4bNVrdSVTTVnqjIqpSbFupw6jT0hq3bZZrWdNMceUxxaUbJVVWW5wsU9AnWSriojSzM3VS0rSZl1W2yrE"),
            (FilterType::Nearest, HashAlg::Mean, 8, "t4VWBIk+/3o"),
            (FilterType::Nearest, HashAlg::Mean, 16, "ToXa9j68DVUhUsnrqUJ+u8XMn0w1DopUjxGEy9lFCPM"),
            (FilterType::Nearest, HashAlg::Mean, 32, "L6inPSfBP+Vpvsb0n/4Muxnso8o1CfZnWV8u6SXqcqS0BRAeiCpqs7pMOmHzktMl6u1QkOeDDdKP0F1cgpAyIXQlbumXf9o3sp3Lfsk2BktkEpHccEfp1pTJbKIGumRXfcFjrH1JurgExWnQBwZCn8RjwO0zQZKJOwm/SnSCMqk"),
            (FilterType::Nearest, HashAlg::Mean, 64, "S6IZmWDCkSDU6RbSmokwFNU5Fgex24jsLgIHyBmsSa3mQD9KxvWrLNL023G0mTnWl/E3Tx9LjqAXAelnL7G877Qiby35kFPY3WJhzCDexiXDQvYXMBYgqoHxTPxC3YepYTU3rTim5mVYdMe54+WkyViX/MyPFn9Xe2rx56YK4Ze6yBeqCCSmfoB9HkaptKvZBfhJKVXU0KAzv+gppdAsHzUqk9j8zmoYHkU9QhTnWJ6O0/XNctpEXU021LeL5TBK8ZHyVPPwdsbY6zNrofok1WnO7adM3HvBKYIxUEfF4xW/4tmGpZdqb1c2tg1hn7hwnBVBk9xIw0hrOLMlsvcvorAgyqN1wI05tN+/kCzAbIVMY0XEcbcFjSqxNV6MMAplxqUSeU5XJm0cYdcoy51wHGYQw/67NM9nf9XiuJH/9gMHVSATyVzEsZWoLUPo5YTrZsyacJ+J98pCZGWrOJuPhECTj6ya0cfx1lY/zD1gYz07HxlxYT9T4spzo1HZRDvuJsfK3niR4hheYQ1V4jChoTvAtKnR0ciIgMtOPeah2xNAwoJyu+fYgMYQ2WARMZ0Cae2RJlLIjvZI7is/IQcuGUJX8r7r0BIRsMVZ8Iw85oMSY5znCHP30lz6HjcYVN9JeDL9PLm+f8XAfsM43JVnURkajD2WoMVWrxxPClS+m7M"),
            (FilterType::Nearest, HashAlg::Median, 8, "t4VUBIk+f3o"),
            (FilterType::Nearest, HashAlg::Median, 16, "ToXa9j68DVUhVsvrqUJ+u8XMn0w1DopUjxGEy9lFCPM"),
            (FilterType::Nearest, HashAlg::Median, 32, "L6inPSfBP+Vpvsb0n/4Muxnso8q1CfZnWV8+6SXrcqS0BRAeyCpqt7pMOmHzktMl6u1QkefDDdKP1F1cgpAyIXQlbumXf9o3sp3Lfsk2Bkt0EpHccEfp1pTJbKoGumRXfcFjrH1JurgExWnUBwZCn8RjwO2zQZKJOwm/SnSCMqk"),
            (FilterType::Nearest, HashAlg::Median, 64, "S6MZnWDCkSDV6xbS2okwFNU5Fgex24jsLgIHyBmsSa3mQD9KxvXrLNL023G1mTnWl/E3Tx9LjqAXEelnL7G877Qiby35kFPY3WJpzCDexiXLQvYXMBYoqoHxXPxC3YepYTU3rTim5mVYdMe54+WkyViX/MyPFn9Xe2rx5+YK4Ze6yBeqCCWmfoR9HkaptKvZBfhJKVXU0KEzv+kppdAsHzUqk9z8zmoYnkU9QhTnWJ6O0/XNctpEXU021LeL5TBK8ZHy1PPwdsbY6zNvqfok1WnO7adM3HvBKYIxUEfF5xW/4tuGpZdqb1c2tg1hn7hwnBVBk9xIw0hrOLMls/cvorAgyqN1wI05tN+/sGzAbIXMY0XEcbcFrSqxNV6MMgpl5qUSe05XJm0cYdcoy91wHGYQw/67Nc9nf9XiuJH/9gMHVSBTyVzMsZWoLcPo5YXrZsyacJ+J98pCZGWrOJuPhECTj6ya0cf11lY/zP9gYz07HxlxYT9T4spzo1HZRDvuJsfK3niR4hheYQ1V4jChoTvAtKvR0ciIgMtOPeeh2xNA4oJyu+/YgMYQ2WAxMZ0Cae2RJlLIjvZI7is/IQc+GUNX8r7r0BIRsM1Z8Iw85oMSY7znCHP30lz6HjcYVN9JeDL9PLm+f8XAf8M43JVnURkajD2WoMVWrzxPClS+m7M"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 8, "tb9/VXo"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 16, "uGZJtWu5QlXvqFatNap2lKm2"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 32, "VPem9RZPl5NJ7NRu2l6LR25ULVaqKddZajKVUZukuaza+rKqDFFtplO3a6lqtklTqkysJnVSVItUKVt3t8xUilrJVcw"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 64, "nqrUZlrFTMttVSXNqzEqUaSOxqpmdFsrjJHKqSznGLMUbWp2RllTkrp0rWnKq3U5j5KULV1Bo7xqRvammYhVk/KQmpoqx6RYchJOjVxmq6qVMauaVg1WWNa1lpQ1pmSpLNqN6iXVCFNISZRdc1gRMhrVlSytqpVqWKW6yHt2lUrXKaS0qfSYamNRTVVFtxlF6YyW1fNMtSrhSiWqYZpqVNGZdmal4y5WtdWsiSSFGlppmlVtyFRbkirJusVW1irVU1GlS3XtcYlRVe1EllRkNYo3UU412mt1WKmqFVOSLbWxNFs9LJMt2jUndetVscRVVLk6axOrcitXQaprXVe2x5RK0VMpmcOy"),
            (FilterType::Nearest, HashAlg::VertGradient, 8, "6lSVq6naSlo"),
            (FilterType::Nearest, HashAlg::VertGradient, 16, "SlRKoioRXZNWKdkylBR7G12ZQ+OhkWtq+5ZR6dplVcw"),
            (FilterType::Nearest, HashAlg::VertGradient, 32, "XbRUdaTS9CnMiirTrlSlbl2jpiKp2+bqppy5lcqaT5Wq1uoSW9Ur03qr2k26rVRT58yqlUqsVUpuy3OVblo1qs3YvVVVqayq2aW6TKWqK0c1p5ktmlGl1UlKbBkwHbNVWdqarCpTWG51SjqLksplQ1dBqmtdV7bHlErRUymZw7I"),
            (FilterType::Nearest, HashAlg::VertGradient, 64, "sjx1EiszRVlHbk1LtIoklVnWJEqmtamtaiZlTqRi8sgdjUVde56iU6VSLSqL1lWboXWabbU4uimdqVeX3K0usqdqUVIXVUk1tCSTkxxSiWVYKrY2NTpFuu5UjXZbVc3qLVU5y2TOqK3VKharKrZavNVZpUmpa9MgtKkyM66dxYq0kVWVltUquildpnqp1oyo41uxc2ySTk9VqDaZtVSqShNlaUqSWiXZs0KVRjVtlZSqpTKqcjKpmSxCktRYUlV1Nko07ZpRz2Q6lXmWl2TVzEu7RG+mUmldTNuczW5uxLlV6e3KvE3JSuZ6qprbtDZLco1UEpOiOVPFuqvS1GzVUQpL+qql7pbYFJMqLdfaWknqVqdUU6xKu2e0jNM0RZmrFhauyia9VKmVSbaSXMmK0cCoda0sWUu1JN2YnFejilmmKmmp1tWttHqxke1UW9mWstq6KgTXnprJTWKrtSRXm6jWltVWlDeVSIgsJVUUC1bSkpWm6bNWTatOrUnyZVVZppZilRmqKs8gO6UYNltJqqqdfL0oSW0+S6WUrJlmlJyV2SI1z0i6xdZbKXNGVNsSZc6Yvpc1TVKq0WpWZYo1xk0VmSFTpMqalsxM2lZNKlOmxrTrZCxyVlSXZ5egTsWojJmWjcpizXVTrbRdt61CqlbKVNVo405faua0zqNqVpc"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 8, "zcYPaHZw0Pg"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 16, "lbH1ZDXhHHL3ke8AoJrlNl42nU8B3EwXeDpB08Cj2ts"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 32, "I88CzyWTU+m6o7S8M7xAnDu7J3kRjgep2Kcq29ZaJAYq3RVjeS8ihhdtGEr3ZdWogMjtwSDWzRm1PjjLM/q9DTwiqIvHN2wU44duIdVq+RuDEtDz5jyJq+2fDhFxELfVRLdubZME6lsVMQdngqov85XgD0hFNhlPqAP5Y8z6bfk"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 64, "H4yrlDSQvVNuFfTwjhlBdjcUHQNPcuq5aspvYys2Q/TMx+/s0C9Ex8KsGl9I77CJlwPW7Fgx1AeFjbX1Fgkq8sOr/uUFJMkvNgVKyvKMkiSbKPlNFmW1yExLPuJ3G8bdA3fqjmAq3ddY8i7s7FyGGxbf3gOQo38eHmGd+FIMoYCOGhGTXSWVtLlY9vYbSUVxVDteLkWaT5Dbrq6EbA56VQZKoizwG8Yguyf3JQXVnVFHtWIZM9Jjb321Vd97IYBYMOSw6ON8QoTTx8yvuf+n4IBMPuMC6OsTEwR7QfSV61cTUtmPZKvt+GC/RQbpnqliB2fFweKl46EKC9T1U9fiJOgYqQzDTc/Isu9W6hbk2WB59o0vcX2VBY+0PidRGHiGD2Zj6rSdZ4kl7tbIX5jAHBpXYJqCScXTouotPNn/CpanYAYNDdsOz5tyaAAit7s7HF6WN4dJa4mb/mlJKdUpnvK0tLfQWIMDl7bOUj0qRwNZrzBJH425Kws1Awh/d1uxGXC75nwffp4wGE+nnhDTOJpQOEy0/O5zrlkaJo1hY2R7WoFWPsB9MHcwRS3xFkmtWG70rr2ZCuwZAkzEPCbf31PDoPb+FJkZGh463TuWYuSw1HKCynNqand6+J2DQCexEIxHBOb7Cl3n+oj5g7AvmsH0x60mdOHOpWDNo9C1FfE"),
            (FilterType::CatmullRom, HashAlg::Gradient, 8, "ykcBMjMiDq4"),
            (FilterType::CatmullRom, HashAlg::Gradient, 16, "iKiqNJr4JpJTEoZKikq6sjWcRmeURoMtnC2gSWgprc0"),
            (FilterType::CatmullRom, HashAlg::Gradient, 32, "kEjqaurZWm5kkZSOjcgqSs3FtLJIQ0WpLEkuyylJEqepakolLFWKE5nmrYlpqVikTtTmmNRG5prKxqSqWM0k6QqjluuznHaWWWElPWQ0PStQjVxZEcuMlWaLsoyKrJG0+lMjLkhJqySqWNdSatUkiape1yWpqFrNVK1lOUbJcWw"),
            (FilterType::CatmullRom, HashAlg::Gradient, 64, "Y8akViRlzuiZXNNSxqhINJhck2tmW6qtlfWmWAmrybTg/DVFtePJ5XF11mWzRZkIykVYV0qdxjLC5Mpgho2sJPJUY/CVpGkmwrNxp7QqiiaxI5VmrRKj6pIVrXJyqeZsrDlpUnSp5FIkmdRWriRn2a1JZFBasytbhZykPRnnrcRXpdhYzbT9VFYsaVM9pVJRKR0pM3TZLJjlV1a2pkQDxaMlGXKmxWJMndFUvDnEbFW7MtWFacpyZYbaxsrJJbFOqfRUzimtptTNcCbTqNqgVnFnp1lN1CTTpOIcVWuapMGs4WhFKspsLJRLonVpTGxqbLKiajNkYk2o1eyqmHVjvlqFVcfwXsesmzVTjTZjTDMWy1LRmHXeZJVZVZtSJrhmJjO1NSeOM02iMyt8ZVpNlZEtJpXWTc1MU9piVlJJadJRdqWFxUnGy2lT0tKpUau5liVLNmlkK5s2k6yspNuqzznJikI6TZFKWipVRhrzoU2uWLqtlmkZq7TS2a0idZmppbNdcxZV2oxNjSazxlbbqlUpViKZTiZi0WRFGuVke1QM7exzF9IclUmZEleRksyqqS0yt9RtQ+u2WWzzkpfRyYxZVCuT0lmaxc3JtpLacitMasnEJjk6qqucpcoubaOpqbcihqJJY0VwOeQsU5thm6pq4rCRtdke2jTCwRiWhDI"),
            (FilterType::CatmullRom, HashAlg::Mean, 8, "39wPdO9/EJg"),
            (FilterType::CatmullRom, HashAlg::Mean, 16, "E+395XXz/PbvID8dsF4lP3483U7zHH5rMWvDS8F738k"),
            (FilterType::CatmullRom, HashAlg::Mean, 32, "L7OKz9WXE+jZp7W8M71T2huLJ+QRnh670Pt439baNA9+2BT9ee4CA/cNOwv/X9HhmIr9gYHfzj+VDyzuM5r5CXwopMvGP+S+8kdqYcNoeDsDEvHTbp65p/23pplQMac1ATdGfJcajAnVsY4AlKhP+RHxB2kCM9Kf+WvL89zyR+A"),
            (FilterType::CatmullRom, HashAlg::Mean, 64, "H4yrvUzA3vv/sQ7BjLhRYHe/PsfN/oi9Y+o95zsC+/THw0+c9IvZhcPuHJ1mf7F7Fdsz79oxjiOdibzXHoGo78Wr3uUnBcFuHSZGyOhMGm4HRrhNOnWnzwQr2uPmA87tAHOyz+Cr7ffJdo/92H/u20Lz3eMQP38SOjOJa3IO+YSuGpHzmiHx3bhb0vZx63NX0nNeZlHDabEL/qx9jAEOBQZao6Tsj84Ze6ftMHOcS1VG5awZ09tnbT+1n5IbZwBcQMHxsCPwhISS402v6b8DxsDIbrd7vGmTDwBxcdjy78d/wtnI7YrIaCH3bAPJzfhon27MQ+7p5gkXD9nlE+9CoPAYqZzhWc+o9Ov2mmzEX2JpsM2XMe7eRS+wvDdUbrjnXGZ/dWwcN30F7t9Bxx7APGPbSD6gC90f7/DH/E/fS5anSAgNBfue2hMA7AMjVw8zPA6+fMNKb6Do/DnJqZ8vzvOzP4942IPL9d/u3Hzg50FZo2TbP0/5iwM1Bwtv7BuoCkC75nzw8p0bUEwGnFDSP7tD7ECy3KoGr0neJ8xFYkR72gPGfOQxADIyZexyJom6QnvEprfZ3+5to1iENDTD2xGjwf7+EYMaCxgD7TeWYy6ZgZOBSHPiKl5wy59NQSbhEg7HBG/fTl/n8sp5VrBvuNTdhac/MMD8MemMr3C+iPc"),
            (FilterType::CatmullRom, HashAlg::Median, 8, "39wPdO9/EJg"),
            (FilterType::CatmullRom, HashAlg::Median, 16, "E+395XXz/PbvID8dsF4lP3483U7zHH5rMWvDS8F738k"),
            (FilterType::CatmullRom, HashAlg::Median, 32, "L7OKz9WXE+jZp7W8M71T2huLJ+QRnh670Pt439baNA9+2BT9ee4CA/cNOwv/X9HhmIr9gYHfzj+VDyzuM5r5CXwopMvGP+S+8kdqYcNoeDsDEvHTbp65p/23pplQMac1ATdGfJcajAnVsY4AlKhP+RHxB2kCM9Kf+WvL89zyR+A"),
            (FilterType::CatmullRom, HashAlg::Median, 64, "H4yrvUzA3vv/sQ7BjLhRYHe/PsfN/oi9Y+o95zsC+/THw0+c9IvZhcPuHJ1mf7F7Fdsz79oxjiOdibzXHoGo78Wr3uUnBcFuHSZGyOhMGm4HRrhNOnWnzwQr2uPmA87tAHOyz+Cr7ffJdo/92H/u20Lz3eMQP38SOjOJa3IO+YSuGpHzmiHx3bhb0vZx63NX0nNeZlHDabEL/qx9jAEOBQZao6Tsj84Ze6ftMHOcS1VG5awZ09tnbT+1n5IbZwBcQMHxsCPwhISS402v6b8DxsDIbrd7vGmTDwBxcdjy78d/wtnI7YrIaCH3bAPJzfhon27MQ+7p5gkXD9nlE+9CoPAYqZzhWc+o9Ov2mmzEX2JpsM2XMe7eRS+wvDdUbrjnXGZ/dWwcN30F7t9Bxx7APGPbSD6gC90f7/DH/E/fS5anSAgNBfue2hMA7AMjVw8zPA6+fMNKb6Do/DnJqZ8vzvOzP4942IPL9d/u3Hzg50FZo2TbP0/5iwM1Bwtv7BuoCkC75nzw8p0bUEwGnFDSP7tD7ECy3KoGr0neJ8xFYkR72gPGfOQxADIyZexyJom6QnvEprfZ3+5to1iENDTD2xGjwf7+EYMaCxgD7TeWYy6ZgZOBSHPiKl5wy59NQSbhEg7HBG/fTl/n8sp5VrBvuNTdhac/MMD8MemMr3C+iPc"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 8, "GUUKgAM"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 16, "ysdFMnUyIwysUoqUiaRElCmZ"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 32, "iKiKNJr4NpJRspZSjE6aKrOaBSVkRpUKjy24KeBpbCmlzVoTiJSJpJWlMMFSVUPUjWOrplGtVTK1ozCG1kBFTEOrwqE"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 64, "kEjqaurZWm5ksZSOzdpqWs3FkDJIU0WpLEkOyy1JEiOpagqlLFXGVBlGqQlpqa2oSqJeuETdppDIZoaqWM2lbA6rFOkjrdbKuRAllVhkLSlkPDmbUclcWRfLsrROodKMmpixNHpRIyxIS+ckqlTXWmjWZImqXlclqalazVStZTlCyXFsUCdOU8wmGivlphGn6pUpYq9FZMqESzZmYUEhZHNXZ+loLqNUaVUTYVmlMilKMVVpZFY5sjfVqoWypGwWJpmm16ncPIzPVtZcRBJjlBtzcVa0cpYrc6KeNmLSrshUzIskqhUpQjjXWFpms41kanSrOLIyRamzESktmSSmTaxlls2ZpLqs"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 8, "UpqUieQUCZk"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 16, "WhOIlJmklaEwxVMVhemtIjmsVz2VIqSD1kZFSEOrwqE"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 32, "UCdOU8wmGiPlpjGn6tUpYi9FdMqESjRmcUUhZGNeZ+VoJqM0aVU36dGhMilOMVVrJVS5kjaFapTilK5WLdk0lMnGnM5EEuNUWxRh1bJytkMxqp4ma9KqxHScjmyKVWkiONVYWjazjWZqNKs4snJFKbMRaS2ZJKdNrGWWzZmkuqw"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 64, "ihJVWi5ZzbORcjZNZpkGG2ijKE6atO0qRKlNXk5m1SphfVN4ChMVRffUvZhhpJasmRSyVOUsVCyZmDOzySinNqlsUWmpeuzFl00SUTlWlThSm200mqZYVdb2zeaxpkpaMzRJtmaOKiuVTDapSuoyrZ8bbbO6LGWulTi1Eqcs13Ukudzdnto4W6mVS1Kb1ogWxxRWMhtbI1rWNjarKB2HrpIq4e4WVq0kVsuehg0rJS+OJZNqMiIrHGyuVU2yti1tV6o0M5JNim5jrWoSV9lJ7SoaY1PEkllJDFsizWlktKlc2mVKRTScLaQUisJu1ClpZSXVahpmlSG0iM3SVBrZZqbUsdq2ZGZZ1ea6dpRTNnnbFi5WmeTZOWWTbBPZTMhFZDJ6RUY8kanERZlKUxu1UapPuz0dlqq1KY1OqS2mIpk7pijpdGkolRItS93WSjmWkjJKVabqvtpZXQka3Exm2jMSG4OUyFcRE7tpZZakmDaaE7NipVJOT0lsm0tTjVWTzao6k1mr2iqkM00tSZNqEiifLWUlpVaOZi2Ol2NmF5RcOSW1x+jIRFUqKCbNJkUnzc6oNnE0STYuzsqaJRjhRAqWpSYrMdmMHpqpAsNI85LjxGipbqxF29LWZAp9raF0lp1S0n2UsTDxiXKUDdOy6tTKcZUOnWRKTsu2xazMUK4"),
            (FilterType::Gaussian, HashAlg::Blockhash, 8, "zcYPaHZw0Pg"),
            (FilterType::Gaussian, HashAlg::Blockhash, 16, "lbH1ZDXhHHL3ke8AoJrlNl42nU8B3EwXeDpB08Cj2ts"),
            (FilterType::Gaussian, HashAlg::Blockhash, 32, "I88CzyWTU+m6o7S8M7xAnDu7J3kRjgep2Kcq29ZaJAYq3RVjeS8ihhdtGEr3ZdWogMjtwSDWzRm1PjjLM/q9DTwiqIvHN2wU44duIdVq+RuDEtDz5jyJq+2fDhFxELfVRLdubZME6lsVMQdngqov85XgD0hFNhlPqAP5Y8z6bfk"),
            (FilterType::Gaussian, HashAlg::Blockhash, 64, "H4yrlDSQvVNuFfTwjhlBdjcUHQNPcuq5aspvYys2Q/TMx+/s0C9Ex8KsGl9I77CJlwPW7Fgx1AeFjbX1Fgkq8sOr/uUFJMkvNgVKyvKMkiSbKPlNFmW1yExLPuJ3G8bdA3fqjmAq3ddY8i7s7FyGGxbf3gOQo38eHmGd+FIMoYCOGhGTXSWVtLlY9vYbSUVxVDteLkWaT5Dbrq6EbA56VQZKoizwG8Yguyf3JQXVnVFHtWIZM9Jjb321Vd97IYBYMOSw6ON8QoTTx8yvuf+n4IBMPuMC6OsTEwR7QfSV61cTUtmPZKvt+GC/RQbpnqliB2fFweKl46EKC9T1U9fiJOgYqQzDTc/Isu9W6hbk2WB59o0vcX2VBY+0PidRGHiGD2Zj6rSdZ4kl7tbIX5jAHBpXYJqCScXTouotPNn/CpanYAYNDdsOz5tyaAAit7s7HF6WN4dJa4mb/mlJKdUpnvK0tLfQWIMDl7bOUj0qRwNZrzBJH425Kws1Awh/d1uxGXC75nwffp4wGE+nnhDTOJpQOEy0/O5zrlkaJo1hY2R7WoFWPsB9MHcwRS3xFkmtWG70rr2ZCuwZAkzEPCbf31PDoPb+FJkZGh463TuWYuSw1HKCynNqand6+J2DQCexEIxHBOb7Cl3n+oj5g7AvmsH0x60mdOHOpWDNo9C1FfE"),
            (FilterType::Gaussian, HashAlg::Gradient, 8, "wsMBEjEiCo4"),
            (FilterType::Gaussian, HashAlg::Gradient, 16, "iKiKcJrwNhITEoRKikqaohUORkYUDoMtmC2oSWwppcw"),
            (FilterType::Gaussian, HashAlg::Gradient, 32, "kFnKaurZWG5kkZSOzMkISszFgLNMQ0SpLEkOyylJEqcpS4IlLFeCMRlGrSlpqdi0TtTmuNRG5prIxqSqWM8E6Qej1MszmHaWWWAlPXA0PTtQiRxZE8uMlUaLsoyKuLGU+tEjLkhLoyTKWNdSatRkmOhcxi2oqFLNZK1FOWbJcXw"),
            (FilterType::Gaussian, HashAlg::Gradient, 64, "Y8akRmZhzaiZXMNyRqhIPJhMg2tmW+islfWGWQmjybRw9BRHMeHJ5XF1lmWzZZkIykRYd8odxjLCxMpgxo2sJPJUY/CUpGkmwrFxJ7QiiiazIYVmrxKj5pIdrXJyqeZs7DnpcnapZlIkmcVSbiRnWaVJZnBakw9bhZzkGRmnPcRXpchYzbX9VFYsaVMcpRBRaR0rMzzpLJ3lV08ypkQDxYMlGTKmxGJEnVFUlDnEYFWzGtWEScpwZYbaxovJJbFOqXJUzjitpsbNcCbTqNigVnFmp1lM3iTToOY8NWvaJMGE4GhFKspsbIRronFpTGxq7LOiaDNkYkyoteyikHVjvBqFRcbwRmesmyVTzTZnTDMWzVLZmHfeZJZJVZNSJrpmJjOxNSeOM2WCMyk8ZxpNBZEtIhdWzc3MEXpiFlJJSdIRdqeFwUnHywlT09KpUaO5hyNLNmlkI5s2l6zkpMuoxznJjkI+T7FLWmpXRh7zoU2sWLrtlmcZiazama0nZ5mppbOZcwZV2IxNrSazxlbbLlUkViKZTmIi0SRHMuVkc1Qd7exzF3IclQmZMHcRksypqSkyt9RMwemyWWxzkpbRyYzZbCsT0lmYzM3IthKbMivM7snEJjkzqKucpcoubaPpKbcihqNZY0VwOeYsU5lhn7po4rATtfke2jTC0RiXxDI"),
            (FilterType::Gaussian, HashAlg::Mean, 8, "394PfO//EJg"),
            (FilterType::Gaussian, HashAlg::Mean, 16, "P+X95/Xz/PbvcL8f+B8lP358/X7zHH57MWvBS8D738k"),
            (FilterType::Gaussian, HashAlg::Mean, 32, "L7OKzd+XE/jZtz/8O78T/hufB/ARnh770Pt8/9LaNA9+2BT9e+4AA/cNOwv/X/HhmIv9AYHfzz+THz3uM5r9C3yopI/mf+S/8kd6cMNj+DsDMvHTbp65l/23hhlQMec9ATfGfBcejgmXsc4AEaDP+QHwD2kDc9Kf/WvP8/zyR+A"),
            (FilterType::Gaussian, HashAlg::Mean, 64, "H4yvvczA3/n/uQ7HjPhZcHe7PseN/4i9Y+s/x3sC+/XHw1/cdwfZh8PuHI9Gf7B7h8sz79oxjiOdi77HH4G478Wr3+UnBcluDSfOyOkMGu4HRrrNfmWHzwQz2OPmA87tAHOK7+CL7//Ico//2H/+20rz3eMQP38SOjOZY3IO+YS+G5HzmiHx3bgb0vZx63N/wnPeZkDDa/AL/rh9DAEOBQ5e46Tsj84BfqfvIfOcD0Vm9a0Z0/tnbD+9n5IbZwBcQMXxsCPwBISC402v+f8DxsDIfvd7+GmXDwB5cdjw74c/gNnI7YrIaQH3bQPpjfhon2/MQ+ft5gkHD9ntE/9GoPAYiJzhXc+o9P/0GmTE32J4uM2XMe7+Rz+wPD9UbvjnHOZ/dWw8I20F/l9hxj+APCf7SH7gG90f7/DH/Af/254HQAgPA/uOmx8A7AGD1w8zPA6+bMPHb6D4/DnJqZ8vzvb3P4943IPL9f/+nHzA4wHZ52CbP077iwEkBwtv7DuoAEA75jz48L0bUGwGnFDSP79B7ACw2OIGr0veB8zBYwR72gPGfOQxAHJyRex2Jgm6QnvExrf5j+9tA0jGNjTD+xGjwf7+EcMaAwAD7DeWYzqYgYOBSHPmKR5wz59NQWbhEw7PHO/fT1/n9s95V7BHvPX/j6c/MMD8MeGMr368gPc"),
            (FilterType::Gaussian, HashAlg::Median, 8, "394PfO//EJg"),
            (FilterType::Gaussian, HashAlg::Median, 16, "P+X95/Xz/PbvcL8f+B8lP358/X7zHH57MWvBS8D738k"),
            (FilterType::Gaussian, HashAlg::Median, 32, "L7OKzd+XE/jZtz/8O78T/hufB/ARnh770Pt8/9LaNA9+2BT9e+4AA/cNOwv/X/HhmIv9AYHfzz+THz3uM5r9C3yopI/mf+S/8kd6cMNj+DsDMvHTbp65l/23hhlQMec9ATfGfBcejgmXsc4AEaDP+QHwD2kDc9Kf/WvP8/zyR+A"),
            (FilterType::Gaussian, HashAlg::Median, 64, "H4yvvczA3/n/uQ7HjPhZcHe7PseN/4i9Y+s/x3sC+/XHw1/cdwfZh8PuHI9Gf7B7h8sz79oxjiOdi77HH4G478Wr3+UnBcluDSfOyOkMGu4HRrrNfmWHzwQz2OPmA87tAHOK7+CL7//Ico//2H/+20rz3eMQP38SOjOZY3IO+YS+G5HzmiHx3bgb0vZx63N/wnPeZkDDa/AL/rh9DAEOBQ5e46Tsj84BfqfvIfOcD0Vm9a0Z0/tnbD+9n5IbZwBcQMXxsCPwBISC402v+f8DxsDIfvd7+GmXDwB5cdjw74c/gNnI7YrIaQH3bQPpjfhon2/MQ+ft5gkHD9ntE/9GoPAYiJzhXc+o9P/0GmTE32J4uM2XMe7+Rz+wPD9UbvjnHOZ/dWw8I20F/l9hxj+APCf7SH7gG90f7/DH/Af/254HQAgPA/uOmx8A7AGD1w8zPA6+bMPHb6D4/DnJqZ8vzvb3P4943IPL9f/+nHzA4wHZ52CbP077iwEkBwtv7DuoAEA75jz48L0bUGwGnFDSP79B7ACw2OIGr0veB8zBYwR72gPGfOQxAHJyRex2Jgm6QnvExrf5j+9tA0jGNjTD+xGjwf7+EcMaAwAD7DeWYzqYgYOBSHPmKR5wz59NQWbhEw7PHO/fT1/n9s95V7BHvPX/j6c/MMD8MeGMr368gPc"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 8, "GUUCgAI"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 16, "wsNFEjEyIw6MUoKUkKAEHAmZ"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 32, "iKiKMJrwNhIzMpZSjk6aK7OGBidERpEOjy2YKeBpbCilzFgTiJSJpJWBMMFSBUHEhWOpplGtVTK1IzRCxkBFSUOroqE"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 64, "kFnKaurZWG5kkZSOzNgIWszNgLJIw0SpLEkOyy1JEqMpSgqnLVfGcRlGqSlppKyoSqBcuETN5pjIRoaKWM2krA6rtOkjjdaCORA3lVhgLSl0PDmZEcncWRPLtpROqLKMipixJHhRIyxIS+ckqlTXemrWYAnoXNYlqahAzVSt5TlmzXF8QCZOU8wmGyPt5jGn68UpZq9FZMqEQzZmYUMhZHNGZ+VrLmd0aVUTYUmlMiFOMRFrZFQ5sibRq5WihW4cZpgm3o3cPMzPVsbMRBJj1Btzc1a2cpYrc+aeJmPWrsRU3IsiqJU5QjjXGHpmsolkYjCrOLIyRSmTECstmSSnTaxlks2ZpDvs"),
            (FilterType::Gaussian, HashAlg::VertGradient, 8, "ApKQwWQcCYk"),
            (FilterType::Gaussian, HashAlg::VertGradient, 16, "GJOIlImklaEwxVMVhWGFYjGsVTw1I7SCxkJFSEOpouE"),
            (FilterType::Gaussian, HashAlg::VertGradient, 32, "QCZOU8wmGyPt5jnn6sUp4i9FZMqEQzRmcUEjZGNOZ+VoJqd0SVUzYdGlMiFOMRVjZVSpkzaVapTilC6eLdg0nMnGnMxEEuNcWxRh1LJylkMz4p4mY9KqxHScjkSAVTliOJcYWjaziWZiMKk4snJhKbMSay2JJKdNrCWWzZmku+w"),
            (FilterType::Gaussian, HashAlg::VertGradient, 64, "gpJVXiZZzLORcjZN5pkGO2jjKE7aFE8qQPltXk5m1SpheVlYCgMXxefQvZhhJpasmRyz1OUsVDyZmDOzySwmNulMU3mpOszlt00SUTkW/TiSm000miZYX5KmTeYxpkpaMzxJtmeOOiuRTDa5TiwyrZ8bbbO6LGWsFDi1EzY8nXWkudzdHtw4U6mVS1Kb1ogWxzB2MhtTI1rGNnaqKB2HLpYi4e4eRo0kRsuehg0vJQ7OJZNqMiIjHHwsFU2yli1td6o0M5ZNim9zrGIzR9nJ6SoaY1PEkhlJDFsizW1ktKlM2mnKZTScLaQUi0JsxClhZSXRahpmlSHkjMnS1B7ZZObksdK0Zm5Z1+S4dpRzNnlZFg5WmfTYOWWSbBMZXNhNZDJ4R0c8kenERZkbUxu10S5POz0dlq6VKY1OOT2GJpk7pmi5dEFolRMtS93WSjiWmjlKVabqvNIZXQlb3Exm2jMSGcOUyGOTE7txZZbkWDaaEzNi5/BOT0lsM0NTjVSTzao7g1mr3jqkMw0tSZNuEiwfLeVhJWaeZj0Ol2NkM5RcOSW1x+zYRFUqKCbNJkknzc4oNmE2yTYuxoqeJRjhRAqWpSYrOfGcHpqpAsNM85LjxGiqZqxh0tLWZAp97KE0kp1Tkn2VsTDxiHOcDcMyatTKcZUPnWRORstyxazMcK4"),
        ];
        #[cfg(feature = "fast_resize_unstable")]
        #[rustfmt::skip]
        let data = &[
            (FilterType::Lanczos3, HashAlg::Blockhash, 8, "zcYPaHZw0Pg"),
            (FilterType::Lanczos3, HashAlg::Blockhash, 16, "lbH1ZDXhHHL3ke8AoJrlNl42nU8B3EwXeDpB08Cj2ts"),
            (FilterType::Lanczos3, HashAlg::Blockhash, 32, "I88CzyWTU+m6o7S8M7xAnDu7J3kRjgep2Kcq29ZaJAYq3RVjeS8ihhdtGEr3ZdWogMjtwSDWzRm1PjjLM/q9DTwiqIvHN2wU44duIdVq+RuDEtDz5jyJq+2fDhFxELfVRLdubZME6lsVMQdngqov85XgD0hFNhlPqAP5Y8z6bfk"),
            (FilterType::Lanczos3, HashAlg::Blockhash, 64, "H4yrlDSQvVNuFfTwjhlBdjcUHQNPcuq5aspvYys2Q/TMx+/s0C9Ex8KsGl9I77CJlwPW7Fgx1AeFjbX1Fgkq8sOr/uUFJMkvNgVKyvKMkiSbKPlNFmW1yExLPuJ3G8bdA3fqjmAq3ddY8i7s7FyGGxbf3gOQo38eHmGd+FIMoYCOGhGTXSWVtLlY9vYbSUVxVDteLkWaT5Dbrq6EbA56VQZKoizwG8Yguyf3JQXVnVFHtWIZM9Jjb321Vd97IYBYMOSw6ON8QoTTx8yvuf+n4IBMPuMC6OsTEwR7QfSV61cTUtmPZKvt+GC/RQbpnqliB2fFweKl46EKC9T1U9fiJOgYqQzDTc/Isu9W6hbk2WB59o0vcX2VBY+0PidRGHiGD2Zj6rSdZ4kl7tbIX5jAHBpXYJqCScXTouotPNn/CpanYAYNDdsOz5tyaAAit7s7HF6WN4dJa4mb/mlJKdUpnvK0tLfQWIMDl7bOUj0qRwNZrzBJH425Kws1Awh/d1uxGXC75nwffp4wGE+nnhDTOJpQOEy0/O5zrlkaJo1hY2R7WoFWPsB9MHcwRS3xFkmtWG70rr2ZCuwZAkzEPCbf31PDoPb+FJkZGh463TuWYuSw1HKCynNqand6+J2DQCexEIxHBOb7Cl3n+oj5g7AvmsH0x60mdOHOpWDNo9C1FfE"),
            (FilterType::Lanczos3, HashAlg::Gradient, 8, "ykcBMjEyTog"),
            (FilterType::Lanczos3, HashAlg::Gradient, 16, "iKCqNJpYZ5JTEoZKjEq6sjWcRmeUToMtmC2gSWgprc0"),
            (FilterType::Lanczos3, HashAlg::Gradient, 32, "kFnqaurZ2m5EkZSOjcpKWs3FkLZIU0WpLEkuSSlJEqerbs4kLVWKU1nmrQlpqVykTtTmuNRC5prKxqUqWMks7QajluuzmFaWWWElPXQ0O6tQjVxZE8uMtWaLsoyKvJG0elMjLkhJpwSaXNdyatUkicpeViWpqFrNVK1lKUbJcWw"),
            (FilterType::Lanczos3, HashAlg::Gradient, 64, "Y8ekViRlyum5XNNaxqhINJhZk2tmW6qtlfWmWAkrybRg/DRFMenJ5XV3122zVZkJymVZd0qdxrLCxMpgho2sJbJUY/CVtGkmwrNxJbQqiiaxI5Vmrxaz4pIVrTJyqeZsrDnJVnSp5FYkm9QWrCRn2aXJZnBasStbjZ2kJVnnrcRVpdhYzbT91FYsaVM9pVJRaR0pNXhZLZylV9aypmQDxaMtWVKmxeJUndBUnDnMbFW7MtWFaep0ZYTawkrdJbNOqfRUzimpptTNcCbXqNigVmVnt1lM1CzTscIcVGuSpMWk4WjFKspsbJRLpvFtTEhqbLKiajdkYk2o1eyimFVislqFVcbwTuesmzVTLTZmTDMWy0LRmHXaZJRZVZNaJr5mJjO1NSaOM2WiMyssZVpFtJUtJp2WbE9MU9piVlJJadqRdqWFxUnWym1T0lKpUau5liVLNmlkK5k2l6ykpNqsTynJikI6TZFKWiJVRhrzo12kWLqtlGkZi6yS2a0ndZmppbNdcxZV2oxMLSazxlbbqlUtViaZTiYi0WRNGuVke1Qs7exzF9MclUmZMnaRksyoqS0yt9RtQ+q2cSTTkpfR2YxZXCub0lmYxc3JtpLaMqpMasnEJjkqqqucpMpubaOpibcihqJJY0VwOWUsU5thm6pq4rCRNdke2DTCwBqXpDI"),
            (FilterType::Lanczos3, HashAlg::Mean, 8, "39wPdO9nUJg"),
            (FilterType::Lanczos3, HashAlg::Mean, 16, "E+395TXz/PbnMD8dsFolP2442U7zHH5rMWvDS8B7X8k"),
            (FilterType::Lanczos3, HashAlg::Mean, 32, "L7PKzdWXE+yZ5728M7lT2huLB/QRnBy70Pt839baNA9+3BT9eu4CA/cNKxv/X9HhmIr9AYHfzj+VDSzuM5r5CXwopcvGP+S+8kdqYclocDsDGvFTZp65p/23phkQIaclADdmfJcajAn1sa4AlKhP+xHxB2kCM9Kb+WvL69zyR+A"),
            (FilterType::Lanczos3, HashAlg::Mean, 64, "H4yrvUzA3uv/sY7FjLhRYHe3PsfN9gi9Y+s95zsC2/TPw2/cdIvZhcPuHJ1mf7F7Fcsy79oxjiOdibzXHoGo58Wr3uUnBcFuHSZGyugMGuYHRrhNenWnzwQ72uPmA87NAHOyz+CL7ffJdo/92H1Pm1rz3eMQL38SOjMJa3IO+YyuWJHzmiHx3bhb0vZx63dX0nNeZlDDabAL/rx1DAEOBQZao6Tsj84UeqftIXOcSVFG5awZ0/tnbT+1H5IbZwBcQMXxsAPghISS502v6b8DxsLIbrd7vGmTDwVxYdCyb8d7w9nI9YrIaCH3bAPLydhqn2zMQ+7pZglXD9nlE+9CoPAYqZzhWc+o9Ov2GmzEX2JpsMWXMe/eRy+wvDfUbLDnXGZvdWwcM20F799Bxx7ANGPbSD6gCd9d7vDH/E/fW5anSAgNBfue2xMA7AMjVwszfA62fMNMb7Do/DnJqZ8vzvO3P4942IPL9d/u3Hzg50FZ42SbP095iwM1Bwtv7BuoCkS75nyw8p0bUEwGnFTaP7vD7ECy3KoGr0neJ8xBYkR72gPWfPQxATIyZezyJom7QnvEprXZX+5to1iGNDTD2xGjwf7+ERMYCxgD7TeWYy6ZgZOBSHPuK14xyZ9NQSbhEg7HDG/fTl/n8sp5VrJPuFTdjac/MMD8semMr3C+iPc"),
            (FilterType::Lanczos3, HashAlg::Median, 8, "39wPdO9nUJg"),
            (FilterType::Lanczos3, HashAlg::Median, 16, "E+395TXz/PbnMD8dsFolP2442U7zHH5rMWvDS8B7X8k"),
            (FilterType::Lanczos3, HashAlg::Median, 32, "L7PKzdWXE+yZ5728M7lT2huLB/QRnBy70Pt839baNA9+3BT9eu4CA/cNKxv/X9HhmIr9AYHfzj+VDSzuM5r5CXwopcvGP+S+8kdqYclocDsDGvFTZp65p/23phkQIaclADdmfJcajAn1sa4AlKhP+xHxB2kCM9Kb+WvL69zyR+A"),
            (FilterType::Lanczos3, HashAlg::Median, 64, "H4yrvUzA3uv/sY7FjLhRYHe3PsfN9gi9Y+s95zsC2/TPw2/cdIvZhcPuHJ1mf7F7Fcsy79oxjiOdibzXHoGo58Wr3uUnBcFuHSZGyugMGuYHRrhNenWnzwQ72uPmA87NAHOyz+CL7ffJdo/92H1Pm1rz3eMQL38SOjMJa3IO+YyuWJHzmiHx3bhb0vZx63dX0nNeZlDDabAL/rx1DAEOBQZao6Tsj84UeqftIXOcSVFG5awZ0/tnbT+1H5IbZwBcQMXxsAPghISS502v6b8DxsLIbrd7vGmTDwVxYdCyb8d7w9nI9YrIaCH3bAPLydhqn2zMQ+7pZglXD9nlE+9CoPAYqZzhWc+o9Ov2GmzEX2JpsMWXMe/eRy+wvDfUbLDnXGZvdWwcM20F799Bxx7ANGPbSD6gCd9d7vDH/E/fW5anSAgNBfue2xMA7AMjVwszfA62fMNMb7Do/DnJqZ8vzvO3P4942IPL9d/u3Hzg50FZ42SbP095iwM1Bwtv7BuoCkS75nyw8p0bUEwGnFTaP7vD7ECy3KoGr0neJ8xBYkR72gPWfPQxATIyZezyJom7QnvEprXZX+5to1iGNDTD2xGjwf7+ERMYCxgD7TeWYy6ZgZOBSHPuK14xyZ9NQSbhEg7HDG/fTl/n8sp5VrJPuFTdjac/MMD8semMr3C+iPc"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 8, "mUUKgAI"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 16, "ysdFsnUyKg6IEpqUCeREFCmZ"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 32, "iKiqNJr4JpJRspZSjU6aKrOaRSVsRpEKry24KeBJaCmtzVoTjJSZpJWhNMlSVUOUjWOrplGtVTK1ozLWxkRFTVOrwqE"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 64, "kFjqaurZWm5ksZSOjdpKW83NkLZI00WpLEkOySlJEiepaoqkLVXSURlGqQlpraWsSqJeuETdppDQZqaqSM2kbB6rFOkjjdbKuVA1lVhkLSlkvDm7WclMSRfLprROoZKMmpiRNnpRIy5ISecEulTWUmrXIInKXlYlqalKzVStZSlGyXFsUGdOU8wmGivtZhGn6pUpYi9FZMqESzZmZUEhZHNfZ+lqLqMUaVUXYVGlMikKM1VpZFY5sjVVq4XipWxeJNmm1oncOMzPVtZMRBJrlJt2cde0cpYrc+6aNmZSrsgUzYqkqhU5QjjXWHpms40kanSrGLIyRSmzEWutmSSuTaxlls2ZpLus"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 8, "EpqUieQUCYk"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 16, "WhOIlJmklakgxVuUhWmtIjusVz3VIjCD1kZFTVOpgqk"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 32, "UCdOU8xmGiPtpjGn6pVp4i9FdMqESzRmMUUhZGNeZ+1oJqNUaVU3qdGhMilOM1VrJVS7kjKNahTilKZeLdk2lcnGmE5EMuJUWxRh1bJytmsx6J4ma9KqxDTcim2KVXkiONcYWjazjSZqNKuQsnJFKbMRaa2ZJK9NrCWWzZmku6w"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 64, "ihBVWiZRybKRcjZNZpkmG2ijKG6btO0qSLlNWs5m1SJheVN4ShsV1XfVvZhhtpa0mRSyVMWsVCyZmDGzySintqlsUWmpeuzFt00SUTlWtThSm200mqYYV5TmTWaxpkraszRJtmaOKiuVTTapSuoyrZcbbbOaJEWulTi1Urc8l3Vku9jN3s44W6mVS1Kb1ogWxxRWMhtTIlrGNjarKB3HrpIqwe4WVo0mVsuepgkrJS+uJZNrMiIrHGysVU22si2lV6o0M5JNimYjrWoSR8lJzSoaY1PEkllICFsqzWlktKlc2uVKZTacraSUisZs1ClpZSXRappmlSGkiM3SVBrZZsbUsdq2bGZZ1+a6dpRTtnnbFC5emfRZOSWTbBPZzMpNZTJ6RUY8kanURZlqWxu1USpPOzVdloq1KY1OqS2mIpk7pmjpdmlslRIta93WahmWmpJKVaLqstoZXQka3Exm2jMSG5GUyFeVE7ppZZakmDaaG7NipXJOT0lum1NTjVSTza46k1mr2qqmM00tSZNqEiifrWUltXaOZi2Gl2Nmt5RceSW152jIZFUqaCbNJlUnzc4oNnE1STauzsqaJRjhREqWpSYrMdmMHJupEsNI85LjzGipZixF0dLWZAp9raFWlr1bUnm0sTDxiHKSDcOy6tTKeZUOnWVKTsu2xazsUK4"),
            (FilterType::Triangle, HashAlg::Blockhash, 8, "zcYPaHZw0Pg"),
            (FilterType::Triangle, HashAlg::Blockhash, 16, "lbH1ZDXhHHL3ke8AoJrlNl42nU8B3EwXeDpB08Cj2ts"),
            (FilterType::Triangle, HashAlg::Blockhash, 32, "I88CzyWTU+m6o7S8M7xAnDu7J3kRjgep2Kcq29ZaJAYq3RVjeS8ihhdtGEr3ZdWogMjtwSDWzRm1PjjLM/q9DTwiqIvHN2wU44duIdVq+RuDEtDz5jyJq+2fDhFxELfVRLdubZME6lsVMQdngqov85XgD0hFNhlPqAP5Y8z6bfk"),
            (FilterType::Triangle, HashAlg::Blockhash, 64, "H4yrlDSQvVNuFfTwjhlBdjcUHQNPcuq5aspvYys2Q/TMx+/s0C9Ex8KsGl9I77CJlwPW7Fgx1AeFjbX1Fgkq8sOr/uUFJMkvNgVKyvKMkiSbKPlNFmW1yExLPuJ3G8bdA3fqjmAq3ddY8i7s7FyGGxbf3gOQo38eHmGd+FIMoYCOGhGTXSWVtLlY9vYbSUVxVDteLkWaT5Dbrq6EbA56VQZKoizwG8Yguyf3JQXVnVFHtWIZM9Jjb321Vd97IYBYMOSw6ON8QoTTx8yvuf+n4IBMPuMC6OsTEwR7QfSV61cTUtmPZKvt+GC/RQbpnqliB2fFweKl46EKC9T1U9fiJOgYqQzDTc/Isu9W6hbk2WB59o0vcX2VBY+0PidRGHiGD2Zj6rSdZ4kl7tbIX5jAHBpXYJqCScXTouotPNn/CpanYAYNDdsOz5tyaAAit7s7HF6WN4dJa4mb/mlJKdUpnvK0tLfQWIMDl7bOUj0qRwNZrzBJH425Kws1Awh/d1uxGXC75nwffp4wGE+nnhDTOJpQOEy0/O5zrlkaJo1hY2R7WoFWPsB9MHcwRS3xFkmtWG70rr2ZCuwZAkzEPCbf31PDoPb+FJkZGh463TuWYuSw1HKCynNqand6+J2DQCexEIxHBOb7Cl3n+oj5g7AvmsH0x60mdOHOpWDNo9C1FfE"),
            (FilterType::Triangle, HashAlg::Gradient, 8, "wkcAMjEiDo4"),
            (FilterType::Triangle, HashAlg::Gradient, 16, "iKiKNLr4JpITkoRKikqaohUsRkaUBoctmC2oaWgopMw"),
            (FilterType::Triangle, HashAlg::Gradient, 32, "kEnqaurZWm5kkZSOzcgqW83FkLtIQ8WpLFkuyylLEqcpakolLFWKsRlGraloqdmkTtTumNRH5prIxqSqWM0kqQujtMuzmHaWWWAlPXQ0PStQiVxZEcuMlGaLsoyKrLGU+lEjLkhZqySqWNdSaNQkiKhe1SWpqFLNVK3lOUbJcWw"),
            (FilterType::Triangle, HashAlg::Gradient, 64, "Y8akRiR1zaiZXMFaxqhINZhUk2tmW6islfWmWAmrSbTw9DVHseHJxXF11mezZZkIykVYd0oVxjLC5Mphhq2sJPJUY/CUpGkmwrNxpbQqiiazIaVmrxKj6pIVrXJyqeZs7LlJU3SpZnIkmdVSriQnWa1JZFBakw9bhZykGVnnrdRXpdhYzbW9VFYsaVMcpRBVaR0pMzTpbJnlU1Y2pkRDxaMlGTKmxGZMnVBVtDnGbFWzEtWEactyZYbahorJJbFOqfRU3iitpsTNcCbTqNqgVnFmp1hN3iTTpOY8VGvapMGM4GhFKspsLJRLonFpTGxqbLOiajNkYk2o9eyqkHVjvBqFVcfwVuesmzVTzTZnTDMWz1LRmHXeZZdYVZNSJrhmJjOxNSeOM0WCMyt8ZVpNFZEtIpdWTc3MUVpiVlJJSdIRdqWFxUnGyilT09KpUaO5hydLNulkI5o2k6zspNuoxjnIikIaTbFK2mpVRBrzoU2sWLqtkmMZibzSma0iZ5mppbOZcxZVWIxNjSaz1lbbqlUkViKZTiJi0SRHEuVka1QN7exzF3IclQ2ZMneRksyKqS0yt9RNQuq2WWxzFpTQSYxZdCsT0FmbzM3JthKbcitMbtnEIjkyqqucpcqubaOpKbcihqJZY0VwOeQsU5thn6pq4rATtdEe2jTCkBqXhDI"),
            (FilterType::Triangle, HashAlg::Mean, 8, "39wPfO//EZg"),
            (FilterType::Triangle, HashAlg::Mean, 16, "V+395fXz/PbvoD8d+F8lP348/X7zHH57UWvBS8B738k"),
            (FilterType::Triangle, HashAlg::Mean, 32, "L7uKzdeXE+zZp7e8M78T3hubB+ARnh670Pt4/9LaNA9+3BT9e+4AA/cNOwv3X/HhmIv9gYHfzz+RDz3uM5r9C3wopM/Gf+S/8kdqccNheDsDEvHTbp65t/23hplQMac1ATfGfBcejAnXsc4AlKjv+RHxB2kCc9Kd/Wvf8/zyR+A"),
            (FilterType::Triangle, HashAlg::Mean, 64, "H4yrvUzA3/n/uQ7BjPh5cHc7PseN/4i9Y+o/5zsC+/XHw1+c9wfZhcPuHJ8mf7F7l9sz79oxjiOdib7HH4G478Wr3+UnBcluDSdGyOpMmm4HRrrNfnWnzwRr2uPmA87tAHOyz+CL7/fIco/92H/u20Lz3eMQP38SOrOZa3IO+YS+GpHzmiHx3bhb0vZx63N/wnNeZlDDafEL/7x9DAEOBQ5e46Tsj84Re6ftMPOcS1VG5a0Z09tnbT+1n5IbZwBcQMHxsCPwBISS402v+f8DxMDIfrd7vGmTDwBxcdjy78c/gdnI7YrI6QH3bQPpjfhon27MQ+ft5gkHD9nlE/9GoPAYiJzhXc+o9O/2kmTEX2Z4uM2XMe7eRS+wvDdUbvjnHOZ/dXwcN30F7t9hxj8APGf7SD7gC90f7/DH/Af/S5YnQAgPA/uenxoA7AOjVw8zPA6+fMPLb6Do/DnJqZ8vzvOzP4942IPL9f+u3Hzg50HZp2DbP0/5iwM1Bwtv7DuoAEA75jz48r0bUEwGnFDSP7tD7ECy2OoGr0veB8zBYwR72gPGfOQxAHJyZexyJwm6UnvE5rfZz+ttI0iGNDTD+xGjwf7+EYMaAwAD7DfWYzqYgYOBSHPiKx5wz59NQSbhEw7HBO/fTl/n8sp5VrBPvPT/jac/MMD8MeGMr3y+iPc"),
            (FilterType::Triangle, HashAlg::Median, 8, "39wPfO//EZg"),
            (FilterType::Triangle, HashAlg::Median, 16, "V+395fXz/PbvoD8d+F8lP348/X7zHH57UWvBS8B738k"),
            (FilterType::Triangle, HashAlg::Median, 32, "L7uKzdeXE+zZp7e8M78T3hubB+ARnh670Pt4/9LaNA9+3BT9e+4AA/cNOwv3X/HhmIv9gYHfzz+RDz3uM5r9C3wopM/Gf+S/8kdqccNheDsDEvHTbp65t/23hplQMac1ATfGfBcejAnXsc4AlKjv+RHxB2kCc9Kd/Wvf8/zyR+A"),
            (FilterType::Triangle, HashAlg::Median, 64, "H4yrvUzA3/n/uQ7BjPh5cHc7PseN/4i9Y+o/5zsC+/XHw1+c9wfZhcPuHJ8mf7F7l9sz79oxjiOdib7HH4G478Wr3+UnBcluDSdGyOpMmm4HRrrNfnWnzwRr2uPmA87tAHOyz+CL7/fIco/92H/u20Lz3eMQP38SOrOZa3IO+YS+GpHzmiHx3bhb0vZx63N/wnNeZlDDafEL/7x9DAEOBQ5e46Tsj84Re6ftMPOcS1VG5a0Z09tnbT+1n5IbZwBcQMHxsCPwBISS402v+f8DxMDIfrd7vGmTDwBxcdjy78c/gdnI7YrI6QH3bQPpjfhon27MQ+ft5gkHD9nlE/9GoPAYiJzhXc+o9O/2kmTEX2Z4uM2XMe7eRS+wvDdUbvjnHOZ/dXwcN30F7t9hxj8APGf7SD7gC90f7/DH/Af/S5YnQAgPA/uenxoA7AOjVw8zPA6+fMPLb6Do/DnJqZ8vzvOzP4942IPL9f+u3Hzg50HZp2DbP0/5iwM1Bwtv7DuoAEA75jz48r0bUEwGnFDSP7tD7ECy2OoGr0veB8zBYwR72gPGfOQxAHJyZexyJwm6UnvE5rfZz+ttI0iGNDTD+xGjwf7+EYMaAwAD7DfWYzqYgYOBSHPiKx5wz59NQSbhEw7HBO/fTl/n8sp5VrBPvPT/jac/MMD8MeGMr3y+iPc"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 8, "GUULgAM"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 16, "wsNEEjUyIw6uUoKUiKBEHAmZ"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 32, "iKiKIJrQNpIRMpZSjk6aK7OmBSdkRpUOry2YKeJoaCml7FoTiJSJpJWlMMFCVUPUhWOpplGtVTK1IzQGxkBFTEOpwqU"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 64, "kEnqaurZWG5kkZSOzdoqS83FgDpIQ0WpLFkOiy1JEqMpagqlLFfWcRlGrSlorK2oSrJcuETdppjI5oaqWM2lbAq7FOkjjdbKuRA3lVhgLStkPDibUcncWRfLspROofKMmpixJHpRIyxISeMkqlTVemjWZImoXNUlqahIzVStZTlGyXFsUCfOU8wmGyPlphmn6sUpZq9FZMqEQzZmYUEhZHNXZ+VpLqdUaVUTYVmlMilKMVVrZFYxsibRqpWypGwUZtmm1q3cPIxNVsZcRBJj1BpzcVa0cpYrc6aeNmLWrkAUjIskKhU5QjzTGHpmM41kYnSrOLIyRamzECktmSSmTazllsWZpLqs"),
            (FilterType::Triangle, HashAlg::VertGradient, 8, "UpKUiWQcCZk"),
            (FilterType::Triangle, HashAlg::VertGradient, 16, "WhOIlImklaEgxVMVhWGNIlGsVTXVIqSC1kZFTUOpwqE"),
            (FilterType::Triangle, HashAlg::VertGradient, 32, "UCdOU8QmGiPlpjmn6tUpYi9FZMqEQjZmYUUhZGNeZ+VoJqM0SVUz4dGhMiFOMVVrZVSpkjaVapTihC6WLdgk3MnGnMxEUuNcWxRh1bJylkMz4p4mY9KqxDScjkUKVSkieNcYWjazjWZidKs4snJFKbMSaS2ZJKdNrGWW1Zmkuqw"),
            (FilterType::Triangle, HashAlg::VertGradient, 64, "gpJVXi5ZzbORcjZNppkGG2ijKE7atc0qQPlNXk5m1SpheVFYCgMVRefQvZhhJJa8mRyyVGUsVCyZmBOzySwnNqlsU2mpeuzll00SUTkW1TiSm000mqZYVZLkTeYxpkpaMzRJtmaOOiuVTTapTi4yrZUbbbO6LGWslTi1E6cs1XWkuVzdntq4U6mVS1Kb1ogWxxB2MhtTY1rWNnarKB2HLpYq4e4eRq0kRsuehg0vJS6OJZNqMiIrHHysVU1yli1ld6o0M5JNim5zrWoSV9nJ7WoaY1PElhlJDFtizW1EtKlM2mlKZTScLaQUi8Js1ClpZSXRahpmlSG0iMnSVBzZZObksdq0ZG5b1ea6dpRSdnnZFi5WmeTZOWWTbBOZTMhFZDJqR0c9kanERZlbUxu10apPOz0dlq6VKY1OKS2GIpk7pmipdEEolRItS93WSjiWkjpKVabqnNoZXQk73ExmmjMSm4OUyGcTE7t5ZZakmHaaE7NipfBOT0lsG0tTjVWTzao6k1mrviqkM00tSZNuEiwfLeUlJWaOZi0Ol2NkEZRcOSW1x+jYRFUqKCbNJkUnxc6oNmE0STYuzoqaJRjhRAqWpScrMdmcPpqpAsNI85KnxGiqbqxl2tKWbApt7KE2lp1Skn2UsTBxiXKUDdOy6tTKcZUOnWRKTsu2xazIeKw"),
            (FilterType::Nearest, HashAlg::Blockhash, 8, "zcYPaHZw0Pg"),
            (FilterType::Nearest, HashAlg::Blockhash, 16, "lbH1ZDXhHHL3ke8AoJrlNl42nU8B3EwXeDpB08Cj2ts"),
            (FilterType::Nearest, HashAlg::Blockhash, 32, "I88CzyWTU+m6o7S8M7xAnDu7J3kRjgep2Kcq29ZaJAYq3RVjeS8ihhdtGEr3ZdWogMjtwSDWzRm1PjjLM/q9DTwiqIvHN2wU44duIdVq+RuDEtDz5jyJq+2fDhFxELfVRLdubZME6lsVMQdngqov85XgD0hFNhlPqAP5Y8z6bfk"),
            (FilterType::Nearest, HashAlg::Blockhash, 64, "H4yrlDSQvVNuFfTwjhlBdjcUHQNPcuq5aspvYys2Q/TMx+/s0C9Ex8KsGl9I77CJlwPW7Fgx1AeFjbX1Fgkq8sOr/uUFJMkvNgVKyvKMkiSbKPlNFmW1yExLPuJ3G8bdA3fqjmAq3ddY8i7s7FyGGxbf3gOQo38eHmGd+FIMoYCOGhGTXSWVtLlY9vYbSUVxVDteLkWaT5Dbrq6EbA56VQZKoizwG8Yguyf3JQXVnVFHtWIZM9Jjb321Vd97IYBYMOSw6ON8QoTTx8yvuf+n4IBMPuMC6OsTEwR7QfSV61cTUtmPZKvt+GC/RQbpnqliB2fFweKl46EKC9T1U9fiJOgYqQzDTc/Isu9W6hbk2WB59o0vcX2VBY+0PidRGHiGD2Zj6rSdZ4kl7tbIX5jAHBpXYJqCScXTouotPNn/CpanYAYNDdsOz5tyaAAit7s7HF6WN4dJa4mb/mlJKdUpnvK0tLfQWIMDl7bOUj0qRwNZrzBJH425Kws1Awh/d1uxGXC75nwffp4wGE+nnhDTOJpQOEy0/O5zrlkaJo1hY2R7WoFWPsB9MHcwRS3xFkmtWG70rr2ZCuwZAkzEPCbf31PDoPb+FJkZGh463TuWYuSw1HKCynNqand6+J2DQCexEIxHBOb7Cl3n+oj5g7AvmsH0x60mdOHOpWDNo9C1FfE"),
            (FilterType::Nearest, HashAlg::Gradient, 8, "xiVGUhUpVdk"),
            (FilterType::Nearest, HashAlg::Gradient, 16, "tGiWidqorTWmitLRDk0rl6dya1KlNtVqVVWijqq02vo"),
            (FilterType::Nearest, HashAlg::Gradient, 32, "VGryuLXKCqWFb1UW5hJbmZltWuor1YYu6pQmKSqtM7MqVduYUtWata+12isqq2YrZlXa1FqmVtWcjvBKSi7dtNxW1WqSnTKrTVMaNbtJKmOo7hhT1qrq0nYim6zZraVcS5OVq1ubqqTTiFRZUUyVlK2qlWpYpbrIe3aVStcppLQ"),
            (FilterType::Nearest, HashAlg::Gradient, 64, "cnK1KmUl1S5WqslSZisRVnMW52JrdcnsWLWQS4oqTaWRkiVVSbKpLC0Vp3ApKy3WTZmyBlJVwqRXdEqp10qsZrKWw1Y1bVvLlBItq6WM8yVUmyQvmcooq5UUrXJGreesrKpUW1UI93U1MZW2qqS0yblWlfXWuitRpVYtjZW0MZVhWdkVrTajDlGnKmcttavcYiyqWnG1UrxFWW6bKiyvlXMXqUJyS2tZvZN6rumkWIMzsVRUUZNmTZfc0oVKNbpq7bRWSiuZNlfNUyvNqsI2VVZni2o2STlVqjKLonVpYZTs0kplyeZqLZdKZVpqZq0+LDtWKa100myqrlqzkU4to71GTbfocM0IW1UoTbWjbNUWykRUWVU1pVZJVUlFsWolpjWpKKoMN20iRaIy815VnbQtsRVpTUsldJWqNApbtpJF1axP003nvCkl0mZ9daVpZ23LjGFKNWoqTSbLVJuG1CWVqmJrJlU0s6pNxnKaaS0rYqqkh1Zb88oW2afyM5kvlLqYVbSLcswapYaVTLSxoUqlVWurWErOUaRRq6txS1k1bcljM5vbirwvKlWqONWSqSV3tyxV47a0WFFaRrguTWhaxW4bNVrdSVTTVnqjIqpSbFupw6jT0hq3bZZrWdNMceUxxaUbJVVWW5wsU9AnWSriojSzM3VS0rSZl1W2yrE"),
            (FilterType::Nearest, HashAlg::Mean, 8, "t4VWBIk+/3o"),
            (FilterType::Nearest, HashAlg::Mean, 16, "ToXa9j68DVUhUsnrqUJ+u8XMn0w1DopUjxGEy9lFCPM"),
            (FilterType::Nearest, HashAlg::Mean, 32, "L6inPSfBP+Vpvsb0n/4Muxnso8o1CfZnWV8u6SXqcqS0BRAeiCpqs7pMOmHzktMl6u1QkOeDDdKP0F1cgpAyIXQlbumXf9o3sp3Lfsk2BktkEpHccEfp1pTJbKIGumRXfcFjrH1JurgExWnQBwZCn8RjwO0zQZKJOwm/SnSCMqk"),
            (FilterType::Nearest, HashAlg::Mean, 64, "S6IZmWDCkSDU6RbSmokwFNU5Fgex24jsLgIHyBmsSa3mQD9KxvWrLNL023G0mTnWl/E3Tx9LjqAXAelnL7G877Qiby35kFPY3WJhzCDexiXDQvYXMBYgqoHxTPxC3YepYTU3rTim5mVYdMe54+WkyViX/MyPFn9Xe2rx56YK4Ze6yBeqCCSmfoB9HkaptKvZBfhJKVXU0KAzv+gppdAsHzUqk9j8zmoYHkU9QhTnWJ6O0/XNctpEXU021LeL5TBK8ZHyVPPwdsbY6zNrofok1WnO7adM3HvBKYIxUEfF4xW/4tmGpZdqb1c2tg1hn7hwnBVBk9xIw0hrOLMlsvcvorAgyqN1wI05tN+/kCzAbIVMY0XEcbcFjSqxNV6MMAplxqUSeU5XJm0cYdcoy51wHGYQw/67NM9nf9XiuJH/9gMHVSATyVzEsZWoLUPo5YTrZsyacJ+J98pCZGWrOJuPhECTj6ya0cfx1lY/zD1gYz07HxlxYT9T4spzo1HZRDvuJsfK3niR4hheYQ1V4jChoTvAtKnR0ciIgMtOPeah2xNAwoJyu+fYgMYQ2WARMZ0Cae2RJlLIjvZI7is/IQcuGUJX8r7r0BIRsMVZ8Iw85oMSY5znCHP30lz6HjcYVN9JeDL9PLm+f8XAfsM43JVnURkajD2WoMVWrxxPClS+m7M"),
            (FilterType::Nearest, HashAlg::Median, 8, "t4VUBIk+f3o"),
            (FilterType::Nearest, HashAlg::Median, 16, "ToXa9j68DVUhVsvrqUJ+u8XMn0w1DopUjxGEy9lFCPM"),
            (FilterType::Nearest, HashAlg::Median, 32, "L6inPSfBP+Vpvsb0n/4Muxnso8q1CfZnWV8+6SXrcqS0BRAeyCpqt7pMOmHzktMl6u1QkefDDdKP1F1cgpAyIXQlbumXf9o3sp3Lfsk2Bkt0EpHccEfp1pTJbKoGumRXfcFjrH1JurgExWnUBwZCn8RjwO2zQZKJOwm/SnSCMqk"),
            (FilterType::Nearest, HashAlg::Median, 64, "S6MZnWDCkSDV6xbS2okwFNU5Fgex24jsLgIHyBmsSa3mQD9KxvXrLNL023G1mTnWl/E3Tx9LjqAXEelnL7G877Qiby35kFPY3WJpzCDexiXLQvYXMBYoqoHxXPxC3YepYTU3rTim5mVYdMe54+WkyViX/MyPFn9Xe2rx5+YK4Ze6yBeqCCWmfoR9HkaptKvZBfhJKVXU0KEzv+kppdAsHzUqk9z8zmoYnkU9QhTnWJ6O0/XNctpEXU021LeL5TBK8ZHy1PPwdsbY6zNvqfok1WnO7adM3HvBKYIxUEfF5xW/4tuGpZdqb1c2tg1hn7hwnBVBk9xIw0hrOLMls/cvorAgyqN1wI05tN+/sGzAbIXMY0XEcbcFrSqxNV6MMgpl5qUSe05XJm0cYdcoy91wHGYQw/67Nc9nf9XiuJH/9gMHVSBTyVzMsZWoLcPo5YXrZsyacJ+J98pCZGWrOJuPhECTj6ya0cf11lY/zP9gYz07HxlxYT9T4spzo1HZRDvuJsfK3niR4hheYQ1V4jChoTvAtKvR0ciIgMtOPeeh2xNA4oJyu+/YgMYQ2WAxMZ0Cae2RJlLIjvZI7is/IQc+GUNX8r7r0BIRsM1Z8Iw85oMSY7znCHP30lz6HjcYVN9JeDL9PLm+f8XAf8M43JVnURkajD2WoMVWrzxPClS+m7M"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 8, "tb9/VXo"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 16, "uGZJtWu5QlXvqFatNap2lKm2"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 32, "VPem9RZPl5NJ7NRu2l6LR25ULVaqKddZajKVUZukuaza+rKqDFFtplO3a6lqtklTqkysJnVSVItUKVt3t8xUilrJVcw"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 64, "nqrUZlrFTMttVSXNqzEqUaSOxqpmdFsrjJHKqSznGLMUbWp2RllTkrp0rWnKq3U5j5KULV1Bo7xqRvammYhVk1tj3Voqx6RYchJOjVxmq6qVMauaVg1WWNa1lpQ1pmSpLNqN6iXVCFNISZRdc1gRMhrVlSytqpVqWKW6yHt2lUrXKaS0qXSZamNRTVVFtxlF6YyX1fNMtSrhSiSqYZpqVNGZdmalYy9WtdWsiSSFGlppmlRtyNRakipJusVW1irVU9GkS3XtcYlRVexEllRlNYo3UU41Wmt1WKmqFVOSLbWxNFo9LJMt2jUndetVscRVVLk6axOrcitXQaprXVe2x5TK0VMpmcOy"),
            (FilterType::Nearest, HashAlg::VertGradient, 8, "6lSVq6naSlo"),
            (FilterType::Nearest, HashAlg::VertGradient, 16, "SlRKoioRXZNWKdkylBR7G12ZQ+OhkWtq+5ZR6dplVcw"),
            (FilterType::Nearest, HashAlg::VertGradient, 32, "XbRUdaRS9SnMiirTrlSlbl2jpiKp2+bqppy4lcqaTpWq1uoSW9Uq03qr2k26rVVT58yqlUqsVUpuS3OVbto0qs3YvVVVKa2q2SW6TKUqK0c1p5ktmtGl1UlKbBkwnbJVWdqarCpTWW51yjqLkspkQ1dBqmtdV7bHlMrRUymZw7I"),
            (FilterType::Nearest, HashAlg::VertGradient, 64, "sjx1EiszRVlHbk1LtIoklVnWJEqmtamtaiZlTqRi8sgdjUVde56iU6VSLSqL1lWboXWabbU4uimdqVeX3K0usqdqUVIXVUk1tCSTkxxSiWVYKrY2NTpFuu5UjXZbVc3qLVU5y2TOqK3VKharKrZavNVZpUmpa9MgtKkyM66dxYq0kVWVltUquildpnqp1oyo41uxc2ySTk9VqDaZtVSqShNlaUqSWiXZs0KVRjVtlZSqpTKqcjKpmSxCktRYUlV1Nko07ZpRz2Q6lXmWl2TVzEu7RG+mUmldTNuczW5uxLlV6e3KvE3JSuZ6qprbtDZLco1UEpOiOVPFuqvS1GzVUQpL+qql7pbYFJMqLdfaWknqVqdUU6xKu2e0jNM0RZmrFhauyia9VKmVSbaSXMmK0cCoda0sWUu1JN2YnFejilmmKmmp1tWttHqxke1UW9mWstq6KgTXnprJTWKrtSRXm6jWltVWlDeVSIgsJVUUC1bSkpWm6bNWTatOrUnyZVVZppZilRmqKs8gO6UYNltJqqqdfL0oSW0+S6WUrJlmlJyV2SI1z0i6xdZbKXNGVNsSZc6Yvpc1TVKq0WpWZYo1xk0VmSFTpMqalsxM2lZNKlOmxrTrZCxyVlSXZ5egTsWojJmWjcpizXVTrbRdt61CqlbKVNVo405faua0zqNqVpc"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 8, "zcYPaHZw0Pg"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 16, "lbH1ZDXhHHL3ke8AoJrlNl42nU8B3EwXeDpB08Cj2ts"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 32, "I88CzyWTU+m6o7S8M7xAnDu7J3kRjgep2Kcq29ZaJAYq3RVjeS8ihhdtGEr3ZdWogMjtwSDWzRm1PjjLM/q9DTwiqIvHN2wU44duIdVq+RuDEtDz5jyJq+2fDhFxELfVRLdubZME6lsVMQdngqov85XgD0hFNhlPqAP5Y8z6bfk"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 64, "H4yrlDSQvVNuFfTwjhlBdjcUHQNPcuq5aspvYys2Q/TMx+/s0C9Ex8KsGl9I77CJlwPW7Fgx1AeFjbX1Fgkq8sOr/uUFJMkvNgVKyvKMkiSbKPlNFmW1yExLPuJ3G8bdA3fqjmAq3ddY8i7s7FyGGxbf3gOQo38eHmGd+FIMoYCOGhGTXSWVtLlY9vYbSUVxVDteLkWaT5Dbrq6EbA56VQZKoizwG8Yguyf3JQXVnVFHtWIZM9Jjb321Vd97IYBYMOSw6ON8QoTTx8yvuf+n4IBMPuMC6OsTEwR7QfSV61cTUtmPZKvt+GC/RQbpnqliB2fFweKl46EKC9T1U9fiJOgYqQzDTc/Isu9W6hbk2WB59o0vcX2VBY+0PidRGHiGD2Zj6rSdZ4kl7tbIX5jAHBpXYJqCScXTouotPNn/CpanYAYNDdsOz5tyaAAit7s7HF6WN4dJa4mb/mlJKdUpnvK0tLfQWIMDl7bOUj0qRwNZrzBJH425Kws1Awh/d1uxGXC75nwffp4wGE+nnhDTOJpQOEy0/O5zrlkaJo1hY2R7WoFWPsB9MHcwRS3xFkmtWG70rr2ZCuwZAkzEPCbf31PDoPb+FJkZGh463TuWYuSw1HKCynNqand6+J2DQCexEIxHBOb7Cl3n+oj5g7AvmsH0x60mdOHOpWDNo9C1FfE"),
            (FilterType::CatmullRom, HashAlg::Gradient, 8, "ykcBMjMiDq4"),
            (FilterType::CatmullRom, HashAlg::Gradient, 16, "iKiqNJr4JpJTEoZGiEqasjWcRmeURoMtnC2gSWgprc0"),
            (FilterType::CatmullRom, HashAlg::Gradient, 32, "kFjqaurZWm5kkZSOjcgqSs3FlLJIQ0WpLEkuyylLEqerakolLFWKE5nmrYlpqVikTtTmuNRH5prKxqSqWM0k6QajluuzmHaWWWElPWQ0PStQjVxZEcuMlWaLsoyKrJG0+lMjLkhJqySqXNdSatUkiape1yWpqFrNVK1FOUbJcWw"),
            (FilterType::CatmullRom, HashAlg::Gradient, 64, "Y8akViRlzuiZXNNSxqhINJhck2tmU6qtlfWmWAmrybTg/DVFtePJ5XF11m2zRZkIykVYV0qdxjLC5Mpgho2sJPJUY/CVpGkmwrNxo7QqiiaxI5VmrRKj6pIVrXJyqeZsrDlJUnSp5FIkmdRWriRn2a1JZFBasytbhZykPRnnrcRXpdhYzbT9VFYsaVM9pVJRaR0pM3TZbJjlU1a2pmQDxaMlGXKmxWJMndFUvDnEbFW7MtWFacpyZYbaxsrJJbFOqfRUzimtptTNcCbTqNqgVnFnp1lN1CTTpOIcVWuapMGs4WhFKspsLJRLonVpTGxqbLKiajNkYk2o1eyqmHVjvlqFVcfQXses2zVTjTZjTDMWy1LRmHXeZJVZVZtSJrpmJjO1NSeOM02iMyt8ZVpNlZEtJpXWTc1MU9piVlJJadJRdqWFxUnGy2lT0tKpUau5liVLNmlkK5k2k6yspNuqzznJikI6TZFKWipVRhrzoU2mWLqtlmkZq7TS2a0idZmppbNdcxZV2oxNjSazxlbbqlUpViKZTiZi0WRFGuVke1QM7exzF9IclUmZEleRksyqqS0yt9RtS+u2WWzzkpfQyYxZVCuT0lmaxc3JtpLacitMasnEJjk6qqucpcoubaOpqbcihqJJY0VwOeQsU5thm6pq4rCRtdke2jTC0RiWhDI"),
            (FilterType::CatmullRom, HashAlg::Mean, 8, "39wPfO9/EJg"),
            (FilterType::CatmullRom, HashAlg::Mean, 16, "E+395XXz/PbvID8d8F4lP3483U7zHF5rMWvDS8F738k"),
            (FilterType::CatmullRom, HashAlg::Mean, 32, "L7OKz9WXE+jZp7W8M71T2huLJ+QRnh670Pt439baNA9+2BT9ee4CA/cNOwv/X9HhmIr9gYHfzj+VDyzuM5r5CXwopMvGP+S+8sdqYcNoeDsDEvHTbp65p/23pplQMac1ATfGfJcajAnVsY4AlKhP+RHxB2kCM9Kf+WvL+9zyR+A"),
            (FilterType::CatmullRom, HashAlg::Mean, 64, "H4yrvUzA3/v/uQ7BjLhRYHe/PsfN/oi9Y+o95zsC+/THw0+c9IvZhcPuHJ1mf7F7Fdsz79oxjiOdibzXHoGo78Wr3uUnBcFuHSZGyOhMGm4HRrhNenWnzwQr2uPmA87tAHOyz+Cr7ffJdo/92H/u20Lz3eMQP38SOjOJa3MO+YSuGpHzmiHx3bhb0vZx63NX0nNeZlHDabAb/qx9jAEOBQZao6Tsj84Ze6ftMHOcC1VG5awZ09tnbT+1n5IbZwBcQMHxsCPwhISS402v6b8DxsDIbrd7vGmTDwBxcdjy78d/wtnI7YrIaCH3bAPJzfhon27MQ+7p5gkXD9nlE+9CoPAYqZzhWc+o9Ov2mmzEX2JpsM2HMe7eRS+wvDfUbrjnXGZ/dWwcN30F7t9Bxx7APGPbSD6gC90f7/DH/E/fS5anSAgNBfue2hMA7AMjVw8zPA6+fMNKb6Do/DnJqZ8vzvOzP4942IPL9d/u3Hzg50FZo2TbP0/5iwM1Bwtv7BuoCkA75nzw8pkbUEwGnFDSP7tD7ECy3KoGr0neJ8xFYkR72gPGfOQxADIyZexyJom6QnvEprfZ3+5to1iGNDTD2xGjwf7+EYMaCxgD7TeWYy6ZgZOBSHPiKl5wy59NQSbhEg7HBG/fTl/n8sp5VrBvuNTfhacvMMD8MemMr3C+iPc"),
            (FilterType::CatmullRom, HashAlg::Median, 8, "39wPfO9/EJg"),
            (FilterType::CatmullRom, HashAlg::Median, 16, "E+395XXz/PbvID8d8F4lP3483U7zHF5rMWvDS8F738k"),
            (FilterType::CatmullRom, HashAlg::Median, 32, "L7OKz9WXE+jZp7W8M71T2huLJ+QRnh670Pt439baNA9+2BT9ee4CA/cNOwv/X9HhmIr9gYHfzj+VDyzuM5r5CXwopMvGP+S+8sdqYcNoeDsDEvHTbp65p/23pplQMac1ATfGfJcajAnVsY4AlKhP+RHxB2kCM9Kf+WvL+9zyR+A"),
            (FilterType::CatmullRom, HashAlg::Median, 64, "H4yrvUzA3/v/uQ7BjLhRYHe/PsfN/oi9Y+o95zsC+/THw0+c9IvZhcPuHJ1mf7F7Fdsz79oxjiOdibzXHoGo78Wr3uUnBcFuHSZGyOhMGm4HRrhNenWnzwQr2uPmA87tAHOyz+Cr7ffJdo/92H/u20Lz3eMQP38SOjOJa3MO+YSuGpHzmiHx3bhb0vZx63NX0nNeZlHDabAb/qx9jAEOBQZao6Tsj84Ze6ftMHOcC1VG5awZ09tnbT+1n5IbZwBcQMHxsCPwhISS402v6b8DxsDIbrd7vGmTDwBxcdjy78d/wtnI7YrIaCH3bAPJzfhon27MQ+7p5gkXD9nlE+9CoPAYqZzhWc+o9Ov2mmzEX2JpsM2HMe7eRS+wvDfUbrjnXGZ/dWwcN30F7t9Bxx7APGPbSD6gC90f7/DH/E/fS5anSAgNBfue2hMA7AMjVw8zPA6+fMNKb6Do/DnJqZ8vzvOzP4942IPL9d/u3Hzg50FZo2TbP0/5iwM1Bwtv7BuoCkA75nzw8pkbUEwGnFDSP7tD7ECy3KoGr0neJ8xFYkR72gPGfOQxADIyZexyJom6QnvEprfZ3+5to1iGNDTD2xGjwf7+EYMaCxgD7TeWYy6ZgZOBSHPiKl5wy59NQSbhEg7HBG/fTl/n8sp5VrBvuNTfhacvMMD8MemMr3C+iPc"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 8, "GUUKgAM"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 16, "ysdFMnUyIwysUoqUiaRElCmZ"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 32, "iKiKMJr4NpJRspZSjE6aKrOaBSVkRpUKjy24KeBpbCmlzVoTiJSJpJWlMMlCVUPUjWOrplGtVTK1ozCG1kBFTEOrwqE"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 64, "kEjqaurZWm5ksZSOzdpqWs3FkDJIU0WpLEkOyy1JEiOpagqlLVXWVBlGqQlpra2oSqJeuETdppDIZoaqWM2lbA66FOkjrdbKuRAllVhkLSlkPDmbUclcWRfLsrROodKMipixNHpRIyxIS+ckqlTXWmjWZImqXlYlqalazVStZTlCyXFsUCdOU8wmGivlphGn6pUpYi9FZMqEQzZmYUEhZHNXZ+loLqdUaVUXYVmlMilKMVVpZFY5sjfVqoWypGwWJpmm163cPIzPVtJMRBJjlBtzcVa0cpYrc6KeNmLSrshUzIskqhUpQjjXWFpms41kanSrOLIyRamzESktmSSmTaxlls2ZpLqs"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 8, "UpqUieQUCZk"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 16, "WhOIlJmklaEgxVNVhemtIjmsVz2VIqSD1kZFSEOrwqE"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 32, "WCdOU8wmGiPlpjGn6tUpYi9FdMqEQjRmcUUhZGNeZ+NoJqM0aVU36dGhMilOMVVrZVS5kjaFahTilK5WLdk0lMnGmM5EEuNcWxRh1bJytkMxqp4ma9KqxHWcjmyKVWkiOtVYWjazjWZqNKs4snJFKbMRaS2ZJKdNrGWWzZmkuqw"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 64, "ihJVWi5ZzbORcjZNZpkGG2ijKE6atO0qQKlNXk5m1SpheVN4ChMVRffUvZhhpJasmRSyVOUsVCyZmDOzySinNqlsUWmpeuzFt00SUTkWlThSm200mqZYVdL2zfaxpkpaMzRJtmaOKiuVTDapSuoyrZ8bbbO6LGWulTi1Eqcs13Vkudzdnto4W6mVS1Kb1ogWxxRWMhtbI1rWNjarKB2HrpIq6e4WVo0kVsuehgkrJS+OJZNqMiIrHGyuVU2yti1tV6o0M5JNim5jrWoSV9lJ7SoaY1PEkllJDFsizWlktKlc2mVKRTScLaQUisJu1ClpZSXVahpmlSG0iM3S1BrZZqbUsdq2bGZZ1ea6dpRTNnnbFi5WmeTZOWWTbBPZTMhFZDJ6RUY8kanERZlKUxu1UapPuz0dlqq1KY1OqS2mIpk7pmipdGkolVItS93WSjiWmjJKVabqvtpZXQka3Exm2jMSG4OUyFcRE7tpZJakmDaaE7NipVJOT0lsm0tTjVWTzao6k1mrmiqkM00tSZNqEiifLWUlpVaOZi2Ol2NkE5RcOSW1x+jIRFUqKCbNJkUnzc6oNnE0STYuzsqaJRjhRAqWpSYrMdmMHpqpAsNI85LnxGipbixF29LWZAp9raF0lp1S0n2UsTDziXKUDdOy6tTKWZUOnWRKTsu2xazMUK4"),
            (FilterType::Gaussian, HashAlg::Blockhash, 8, "zcYPaHZw0Pg"),
            (FilterType::Gaussian, HashAlg::Blockhash, 16, "lbH1ZDXhHHL3ke8AoJrlNl42nU8B3EwXeDpB08Cj2ts"),
            (FilterType::Gaussian, HashAlg::Blockhash, 32, "I88CzyWTU+m6o7S8M7xAnDu7J3kRjgep2Kcq29ZaJAYq3RVjeS8ihhdtGEr3ZdWogMjtwSDWzRm1PjjLM/q9DTwiqIvHN2wU44duIdVq+RuDEtDz5jyJq+2fDhFxELfVRLdubZME6lsVMQdngqov85XgD0hFNhlPqAP5Y8z6bfk"),
            (FilterType::Gaussian, HashAlg::Blockhash, 64, "H4yrlDSQvVNuFfTwjhlBdjcUHQNPcuq5aspvYys2Q/TMx+/s0C9Ex8KsGl9I77CJlwPW7Fgx1AeFjbX1Fgkq8sOr/uUFJMkvNgVKyvKMkiSbKPlNFmW1yExLPuJ3G8bdA3fqjmAq3ddY8i7s7FyGGxbf3gOQo38eHmGd+FIMoYCOGhGTXSWVtLlY9vYbSUVxVDteLkWaT5Dbrq6EbA56VQZKoizwG8Yguyf3JQXVnVFHtWIZM9Jjb321Vd97IYBYMOSw6ON8QoTTx8yvuf+n4IBMPuMC6OsTEwR7QfSV61cTUtmPZKvt+GC/RQbpnqliB2fFweKl46EKC9T1U9fiJOgYqQzDTc/Isu9W6hbk2WB59o0vcX2VBY+0PidRGHiGD2Zj6rSdZ4kl7tbIX5jAHBpXYJqCScXTouotPNn/CpanYAYNDdsOz5tyaAAit7s7HF6WN4dJa4mb/mlJKdUpnvK0tLfQWIMDl7bOUj0qRwNZrzBJH425Kws1Awh/d1uxGXC75nwffp4wGE+nnhDTOJpQOEy0/O5zrlkaJo1hY2R7WoFWPsB9MHcwRS3xFkmtWG70rr2ZCuwZAkzEPCbf31PDoPb+FJkZGh463TuWYuSw1HKCynNqand6+J2DQCexEIxHBOb7Cl3n+oj5g7AvmsH0x60mdOHOpWDNo9C1FfE"),
            (FilterType::Gaussian, HashAlg::Gradient, 8, "wsMAEjEiCo4"),
            (FilterType::Gaussian, HashAlg::Gradient, 16, "iKiKcJrwNhITEoRKikqaohUORkYUBoMtmCmoSWwppcw"),
            (FilterType::Gaussian, HashAlg::Gradient, 32, "kFnKaurZ2G5kkZSOzMkISs3FgLNMQ8SpLEkOyylJEqcpS4IlLFeCMRlGrSlpqdi0TtTmuNRG5prIxqSqWM8E6Qej1MszmHaWWWAlPXA0PTtQiRxZE8uMlUaLsoyKuLGU+NEjLkhLoyTKWNdSatRkmehcxi2oqFLNZK1FOWbJcXw"),
            (FilterType::Gaussian, HashAlg::Gradient, 64, "Y8akVmZhzqiZTMNyRqhIPJhMg2tmW+islfWGWQmjybRw9BZHsePJ5XF1lmezZZkIymRYd8odxjLCxMpgxo2sJPJUY/CUpGkmwrFxJ7QiiiazMYVmrzKj5pIVrXJyqeZs7DnpcnapZlIkmcVSbiRnWaVJZnBakw9bhZzkGRmnPcRXpchYzbX9VFYsaVMcpRBRaR0rMzzpLJ3lV08ypkQDxYMlGTKmxGZEnVFUlDnEYEWzGtWEScpwZYbaxovJJbNOqXJUzjitpsbNcCbTqNigVnFmp1lM3iTToOY8FWvaJMGk4GhFKspsbIRronFpTGxq7LOiaDNkYkyoteyikHVjvBqFRcbwRmesmyVTzTZmTDMWzVLZmHfeZJZJVZNSJrpmJjOxNSeOM2WCMyk8ZxpNBZEtIpdWyc3MEXpiFlJJSdJRdqfFwUnHywlT09KpUaO5hyNLNmlkI5s2l6zkpNuoxznJjkI+T7FLWmpXRh7zoU2sWLrtlmcZiazama0nZ5mppbOZcwZV2IxNrSazxlbbLlUkViKZTmIi0SRHMuVkc1Qd7exzF3IclQmZMHcRksypqSkyt9RMwemyWWxzkpbRyYzZbCsT0lmYzM3IthKbMyrM7snEJjkzqKucpcoubaPpKbcihqNZY01wOeYsU5thn7po4rATtfke2jTC0RiXxDI"),
            (FilterType::Gaussian, HashAlg::Mean, 8, "394PfO//EJg"),
            (FilterType::Gaussian, HashAlg::Mean, 16, "P+X95/Hz/Pbv8L8f8B8lP358/X7zHH57MWvBC8D738k"),
            (FilterType::Gaussian, HashAlg::Mean, 32, "L7OKzd+XE/jZtz/8O78T/hufB/ARnh770Pt8/9LaNA9+2BT8e+4AA/cNOwv/X/GhmIv9AYHfzz+THz3uM5r9C3yopI/mf+S/8kd6cMNj+DsDMvHTbp65l/23hhlQMec9ATfGfBcejgmXsc4AEaDP+QHwB2kDc9Kf/WvP8/zyR+A"),
            (FilterType::Gaussian, HashAlg::Mean, 64, "H4zvvczA3/n/uQ7HjPhZcHe7PseN/4i9Y+s/x3sC+/XHw1/cdwfZh8PuHJ9Gf7B7h8sz79oxjiOdib7HH4G478Wr3+UnBcluDSfOyOkMGu4HRrjNfmWHzwQz2OPmA87tAHOK7+CL7//Ico//2H/+20Lz3eMQP38QOjOZY3IO+YS+G5HzmiHx3bgb0vZx63N/wnPeZkDDafEL/rh9DAEOBQ5e46Tsj84BfqfvIPOcD0Vm9a0Z0/tnbD+9n5IbZwBcQMXxsCPwBISC402v+f8DxsDIfvd7+OmXDwB5cdjw74c/gNnI7YrI6QH3bQPpjfhon2/MQ+ft5gkHD9nlE/9GoPAYiJzhXc+o9P/0mmTE32J4uM2XMe7+Rz+wPD9UbvjnHOZ/dWw8I20F/l9hxj+APCf7SH7gG90f7/DH/Af/254HQAgPA/uOmx8A7AOD1w8zPA6+bMPHb6D4/DnJqZ8vzvf3P4942KPL9f/+nHzA4wHZ52CbP077iwEkBwtv7DuoAEA75jz48L0bUGwGnFDSP79B7ACw2OIGr0veB8zBYwR72gPGfOQxAHJyRex2Jgm6QnvExrf5j+9tA0jGNjTD+xGDwf7+EYMaAwAD/DeWYzqYgYOBSHPmKR5wz59NQWbhEw7PHO/fT1/n9s95V7BHvPX/jac/MMD8MeGMr368gPc"),
            (FilterType::Gaussian, HashAlg::Median, 8, "394PfO//EJg"),
            (FilterType::Gaussian, HashAlg::Median, 16, "P+X95/Hz/Pbv8L8f8B8lP358/X7zHH57MWvBC8D738k"),
            (FilterType::Gaussian, HashAlg::Median, 32, "L7OKzd+XE/jZtz/8O78T/hufB/ARnh770Pt8/9LaNA9+2BT8e+4AA/cNOwv/X/GhmIv9AYHfzz+THz3uM5r9C3yopI/mf+S/8kd6cMNj+DsDMvHTbp65l/23hhlQMec9ATfGfBcejgmXsc4AEaDP+QHwB2kDc9Kf/WvP8/zyR+A"),
            (FilterType::Gaussian, HashAlg::Median, 64, "H4zvvczA3/n/uQ7HjPhZcHe7PseN/4i9Y+s/x3sC+/XHw1/cdwfZh8PuHJ9Gf7B7h8sz79oxjiOdib7HH4G478Wr3+UnBcluDSfOyOkMGu4HRrjNfmWHzwQz2OPmA87tAHOK7+CL7//Ico//2H/+20Lz3eMQP38QOjOZY3IO+YS+G5HzmiHx3bgb0vZx63N/wnPeZkDDafEL/rh9DAEOBQ5e46Tsj84BfqfvIPOcD0Vm9a0Z0/tnbD+9n5IbZwBcQMXxsCPwBISC402v+f8DxsDIfvd7+OmXDwB5cdjw74c/gNnI7YrI6QH3bQPpjfhon2/MQ+ft5gkHD9nlE/9GoPAYiJzhXc+o9P/0mmTE32J4uM2XMe7+Rz+wPD9UbvjnHOZ/dWw8I20F/l9hxj+APCf7SH7gG90f7/DH/Af/254HQAgPA/uOmx8A7AOD1w8zPA6+bMPHb6D4/DnJqZ8vzvf3P4942KPL9f/+nHzA4wHZ52CbP077iwEkBwtv7DuoAEA75jz48L0bUGwGnFDSP79B7ACw2OIGr0veB8zBYwR72gPGfOQxAHJyRex2Jgm6QnvExrf5j+9tA0jGNjTD+xGDwf7+EYMaAwAD/DeWYzqYgYOBSHPmKR5wz59NQWbhEw7PHO/fT1/n9s95V7BHvPX/jac/MMD8MeGMr368gPc"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 8, "GUQCgAI"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 16, "wsNFEjEyIgyMEoKUkKAEHAmZ"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 32, "iKiKMJrwNpJzMpZSjE6aKrOWBidMRpEOjy2YKeBpbCilzFiTiJSJpJWBMMFSBUXUjWOpplGtVTK1IzRCxkBFSEOrQqE"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 64, "kFnKaurZWG5kkZSOzNgIWszNgLJIw0SpLEkOyy1JEqMpSgqnLVfGcRlGqSlppKyoSqBcuETN5pjIRoaKWM2k7A7rlOkjjdaCORA3lVhgLSl0PDmZEcncWRPLtpROqLKMipixJHhRIyxIS+ckqlTXemrWRAnoXNYtqahCzVStYTlmzXF8QCZOU8wmGyPtpjGn64UpZqtFZMqEQzZmYUMhZHNGZ+VrLmd0aVUTYVmlMiFOMRFrZFQ5sibRq5WihW4cZpgm3oncPMzPVsbMRBJj1Btzc1a0cpYrc+KOJmPWrkRU3ItkKJU5QjjXGHpmsolkYjCrOLIyRSmzECstmSSnTaxlks2ZpDvs"),
            (FilterType::Gaussian, HashAlg::VertGradient, 8, "EpKQgWQcCYk"),
            (FilterType::Gaussian, HashAlg::VertGradient, 16, "GJOIlImklaEwxVMVpWGNYjGsVTy1IrSCxkJFSEOpYuE"),
            (FilterType::Gaussian, HashAlg::VertGradient, 32, "QGZOU8wmGyPt5jnn6sVp4i9FZMqEQzRmYUEjZGNOZ+VoJqd0SVUzYVGhMiFOMRVjZVSpkzaVapTikC6eLdg0nMnGnMxEEuPcWxRh1bJylkMz4p4mY9KqxHScjkSAlTliOJcYWjaziWZiMKk4snJhKbMSay2JJKdNrGWWzZmku+w"),
            (FilterType::Gaussian, HashAlg::VertGradient, 64, "ghJVXiZZzbORcjZN5pkGO2jjKE7aFE8qQPltXk5m1SpheVlYCgMXxefQvZhhJpasmRyz1OUsVDyZmDMzySwmNulMU3mpOszlt00SUTkW/TiSm000miZYX5K2TeYxpkpaMzxJtmeOOiuRTTa5TiwyrZ8bb7O6LGWsFDi1EzY8nXWkuNzdHtw4U6mVS1Kb1ogWxzB2MhtTI1rGNnaqKB2HLpYi4e4eRo0kRsuehg0rJQzOJZNqMiIjHGwoFU2yli9td6o0M5JNim9zrGIzR9nJzSoaY1PEkhlJDFsizW1ktKlM2mHKZTScLaQUi0JsxClhZSXRahpmlSHkiNnS1B7ZZMbksdK0ZmxZ1+S4dpRzNnlZFg5WmfTYOWWSbBMZXNhNZDJ4R0c8kenERZkbUxu10S5POz0dlq6VKY1OOT2GIpk7pmj5dGFolRMta93WSjiWmjFKVbbKvNIZXQlb3Exm2jMSG8OUyGcTE7txZZbkWDaaEzNi5/BOT0lsM0NTjVSTzao7g1mrzjqkMw0tSZNuEiwfLeVhIWaOZj0Ol2NkE5RcOSW1x+zYRFUqKCbNJkknzc4oNmE2yTYuxoqeJRjhRAqWpSYrMfGMHpqpAsNM85LjxGiqbqxh0tLWZAp97KE0kp1Tkn2VsTDxiHOcDcMyatTKcZUPjWRORst2xazMcK4"),
        ];
        let start_time = Instant::now();
        for (filter, alg, size, hash) in data {
            let current_hash = HasherConfig::new()
                .hash_alg(*alg)
                .hash_size(*size, *size)
                .resize_filter(*filter)
                .to_hasher()
                .hash_image(&image_buffer)
                .to_base64();

            expected_hashes.push((*filter, *alg, *size, (*hash).to_string(), current_hash));
        }
        println!("Time: {:?}", start_time.elapsed());

        // Helper to regenerate the expected hashes
        // for (filter, alg, size, _expected, current) in expected_hashes.iter() {
        //     println!("(FilterType::{filter:?}, HashAlg::{alg:?}, {size}, \"{current}\"),");
        // }

        let diff_hashes: Vec<_> = expected_hashes
            .into_iter()
            .filter(|(_, _, _, expected, current)| expected != current)
            .collect();

        if !diff_hashes.is_empty() {
            let diff_number = diff_hashes.len();
            let mut diff_str = String::new();
            for (filter, alg, size, expected, current) in diff_hashes {
                let expected_bytes = base64::engine::general_purpose::STANDARD_NO_PAD
                    .decode(&expected)
                    .expect("Invalid base64");
                let current_bytes = base64::engine::general_purpose::STANDARD_NO_PAD
                    .decode(&current)
                    .expect("Invalid base64");

                let distance = if expected_bytes.len() != current_bytes.len() {
                    -9999
                } else {
                    hamming_bitwise_fast(&expected_bytes, &current_bytes) as i32
                };

                diff_str.push_str(&format!(
                    "{filter:?} {alg:?} {size}x{size} with diff {distance}: expected {expected}, got {current} - \n"
                ));
            }
            panic!("{diff_number} hashes did not match:\n{diff_str}");
        }
    }
}
