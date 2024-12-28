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
    /// * <http://www.hackerfactor.com/blog/?/archives/432-Looks-Like-It.html>
    /// Krawetz describes a "pHash" algorithm which is equivalent to Mean + DCT preprocessing here.
    /// However there is nothing to say that DCT preprocessing cannot compose with other hash
    /// algorithms; Gradient + DCT might well perform better in some aspects.
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

impl<I: Image> CowImage<'_, I> {
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
    use fast_image_resize::{PixelType, ResizeAlg, ResizeOptions, Resizer};

    let Ok(src_image) = fast_image_resize::images::Image::from_vec_u8(
        img.width(),
        img.height(),
        img.to_vec(),
        PixelType::U8, // Luma8 is always U8
    ) else {
        return imageops::resize(img, width, height, filter).to_vec();
    };

    let mut dst_image = fast_image_resize::images::Image::new(
        width,
        height,
        PixelType::U8, // Luma8 is always U8
    );
    let mut resizer = Resizer::new();
    let resize_alg = match filter {
        FilterType::Nearest => ResizeAlg::Nearest,
        FilterType::Triangle => ResizeAlg::Convolution(fast_image_resize::FilterType::Bilinear),
        FilterType::CatmullRom => ResizeAlg::Convolution(fast_image_resize::FilterType::CatmullRom),
        FilterType::Gaussian => ResizeAlg::Convolution(fast_image_resize::FilterType::Gaussian),
        FilterType::Lanczos3 => ResizeAlg::Convolution(fast_image_resize::FilterType::Lanczos3),
    };

    let resize_options = ResizeOptions::new().resize_alg(resize_alg);
    if let Err(_) = resizer.resize(&src_image, &mut dst_image, Some(&resize_options)) {
        return imageops::resize(img, width, height, filter).to_vec();
    };

    dst_image.buffer().to_vec()
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

fn debug_filter_type(ft: &FilterType) -> &'static str {
    use FilterType::{CatmullRom, Gaussian, Lanczos3, Nearest, Triangle};

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
    use std::time::Instant;

    use base64::Engine;
    use hamming_bitwise_fast::hamming_bitwise_fast;
    use image::imageops::FilterType;
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
        let decoded_result = ImageHash::from_base64(&base64_string);

        assert_eq!(decoded_result.unwrap(), hash1);
    }

    #[test]
    fn test_hash_stability() {
        let image_buffer = gen_test_img(256, 256);

        let mut expected_hashes: Vec<(FilterType, HashAlg, u32, String, String)> = Vec::new();
        #[cfg(not(feature = "fast_resize_unstable"))]
        let data = &[
            (FilterType::Lanczos3, HashAlg::Blockhash, 8, "3M0FbCH1SXo"),
            (FilterType::Lanczos3, HashAlg::Blockhash, 16, "Rmtwt5NMuVGZAHNooZz9P9IYk8R/t5BcqAnrPgTZ9pY"),
            (FilterType::Lanczos3, HashAlg::Blockhash, 32, "+JhF7rcRfDgFvjlWJBsrVxfiaTYPDVOQ509GlqOFS7dL0gFG+1tsEIc2YaxOjy2/UU7QWqXdmZaTiZ0GUt37CqRmUrOM8spCTlc2pHfVukj+Dy1JzuedykxTZBgEh/si2OhlgqulS0TKOvRP6twsFxJrVwUVaMD+gh2+J3zecM4"),
            (FilterType::Lanczos3, HashAlg::Blockhash, 64, "VVTBXKcKnFdaqibDc/X7xH8HZs/3W0SDJE0DI1BZyJ0zz7Xv4kg9sych6keNnokFcJyBuN+Xb4qkCs9lUYwcM7ZOh9SqOCkNj5GseqYmPL0dcOIyBD5Tz9+Fu4pXpoSVemr8PRp1r5dh2F/iaIC8CaeQE5hNAiKbSc8u5RNxPc+Pog2vrMRdczJ51nKRoJ7ajk3OV7IMBAl/9kZoT/UCYnHSFIsBNFKYJ0BeK1va9Fr/uLda0Dx3y+wXxUhpid0lSSZWsnC7fDvtG/g8DpOUwTaA42DDmyzBJ93129GgOsOPRaKodH9+kFlTbYqHgkEJRgFS387Wtphss/biT5/oFbCFZhwJF2tcPOx+mER0KZP2yRlHJfNMLBHiT/6s9G+i5yVdIjpPYk7oeJdmrOVwbjorItHOLWR5d6UTc4DGmpK8rLVB4phTkIen5wS9hWMm1XCxbOHJivCc0bt8Y6WuppjayHqU5tmHqjIXGxIfWlc1CS+wr+21SViRt8DtcQxaNGWcqmFtBuLFesLap2jOwe3qrqQnZCRheagz2YV0On5KgtUONP53YHz0XFUE430ZpATg4tE2IUPZ6lHX3mq/jYxWVZt+fjdADoduNDwboxKwy8Catfj0NhsnBDxK00fvBIX0UZT6WQxN54dK5Mc3baIuR3MkFn7asDvUx+M6nVA"),
            (FilterType::Lanczos3, HashAlg::Gradient, 8, "VCpSJptCLTU"),
            (FilterType::Lanczos3, HashAlg::Gradient, 16, "qjWoVkjWzBgmkpgmOJuomE3OjWsaGt12toyknmQYIcM"),
            (FilterType::Lanczos3, HashAlg::Gradient, 32, "jcwlxolJLwtOWy0bm+y4MagpNTM1k1mRkmBKmlCQUpMtSYRTrQ0t2zaZOCWh5aSEri3XmvrWlNbMtEVpy9marNOR3rVVUnpwZVubtatrTU5LhoxkmhLFmrFBkxui4hooJK1kyFRWaWNxVNTiNm5Gnc01YUFRJ2qtxmpLNpJMViM"),
            (FilterType::Lanczos3, HashAlg::Gradient, 64, "qutqsiVe2hakxNqpmVLaNG3ZYyaTFXOJi6SFJVpNZpfVIIpVWUndlSBZQ2KlgouUm+VWa9RSqUpYpWGkTIWSnkmR6U5Ltq2lU1qDyLPSEqemnlSzphJL40S2zaJNM7NVlSGPuqxsq0q8yiFpNtWWzFptME20ZrPZMmfHtphSjdYyWaZzJrQYMYxJZZpUzYay5CpjC5m0hG1TWWWmYpQGqRspo6XINlKcELqrlYliO21GRVMsWZaazZZFVSQp4lDNtpkhRTrJJqt2maoNZ6mc7kntTDZtUSr10+U6qTVSs7PQsrWqM1su3GK6c0xzUmFMybCJ0bpsM4yWqxFZoqWs5Epak9koSTPWkxI2a2aroElbZS4ppVJmvVKSKGssW0UztGKKRYxBc0Z3rEpypXUabcpUpXpjeaRNitGpWExrSpLGEq/UWmVRmcVSZJbh9HKTysYWJ3VNrubYSyuXMtWKIEdsJjnZskzlbYiySZpFW5O0dpHO7OKWTLzMytrkTGJD6whv1bAmUus6RXNNli6VUHY3VraiNJcttFtadk4buzIlmWljUiuFVksbZskyoyS9KZIxuVCyseUVtqhabxqByU3Ma2kzGyslUTOzRpLIqroeNWpuyjazuiaTVLZkk1EjklRZmN09SK2TNDJZW1URP0yzmCi2Ww/bGc1qcWStycw"),
            (FilterType::Lanczos3, HashAlg::Mean, 8, "eQ9N/ff/eOs"),
            (FilterType::Lanczos3, HashAlg::Mean, 16, "V2fQdN+UuXNcEnMs4bLRu17cG8f/85r96InpPsE7Ls4"),
            (FilterType::Lanczos3, HashAlg::Mean, 32, "OLlvnDeTfB4dvjl+BLk5I1NzYSdrpnOzp8+ekOOjBx/Y0gjX+ntIAkU/cSwHj609HHqevoWNuZeTOc9LFvO/7awn0PX4hf5Dj7IWvRbPmlD3n7lJtW+ev+5XMP9Nh/8qWfjBgMvtS0bHifDHydwMGRL5R0eGaMD9nQXa5PyZfOY"),
            (FilterType::Lanczos3, HashAlg::Mean, 64, "VVfVZ+0o+HdImRdPs9fyZZ4Xx2x+P0YTdk8PY9BZbLezx3WP8hiYF08C/G+LH581JMiF2sUXO043CM/pmQ0iPJMng9kCKr0FvjQOO+eWPv7NPY0mBT5fjtttuk7ZJibVfk7/dM1Bgs8jsF/ybIEUyDfYZdotBKaTTf4efTF1nZfN8g3O7KA9dTlyyTPFiZ7yiV2KPvAlCGnsvs8MzvkfIHTSVIuPPNSYJ3B2L5N+ckmdgP5K07wziXkTrWjjgMEJSQPummC7fL/NAAQ/TmOczzPQ0GRIwy7BJ4vz0mFUPuOHReMeZ/L8+M/gR5nGwsNJF0cHp75J9vgoVj/yT+/o/bC3J75RU3eeZOx1igzQJ9v0y1lT7fNuIITlS9JI/w8w688QjTgf4kbc6Z9GSsXwTRA57HTm4KFZdwfTsdhjWJK5//+BV8RDsZvv3waHxWemNZ19bOrLjs6xl38+ZYG855zYTXKSpl3P3zM9kwAf+zdpzSeQ3f+9WWORFYDvXUbGFhFQvyFtAIjFmuSbrmi90cnurMSuYRRhL7b//Is2OnReJsPGNv6P9Jf0TINE5n3wUgTj4/Emg4H7eDG13DI/gZgSVut3Mjt8jubn/bbbJiI4aoDYlGD2sB0mmCxAFvNv5QGyEZX5GYj856dS1kMxfLBuE2NgQzja8LvR5+k/n5w"),
            (FilterType::Lanczos3, HashAlg::Median, 8, "eQ9N/ff/eOs"),
            (FilterType::Lanczos3, HashAlg::Median, 16, "V2fQdN+UuXNcEnMs4bLRu17cG8f/85r96InpPsE7Ls4"),
            (FilterType::Lanczos3, HashAlg::Median, 32, "OLlvnDeTfB4dvjl+BLk5I1NzYSdrpnOzp8+ekOOjBx/Y0gjX+ntIAkU/cSwHj609HHqevoWNuZeTOc9LFvO/7awn0PX4hf5Dj7IWvRbPmlD3n7lJtW+ev+5XMP9Nh/8qWfjBgMvtS0bHifDHydwMGRL5R0eGaMD9nQXa5PyZfOY"),
            (FilterType::Lanczos3, HashAlg::Median, 64, "VVfVZ+0o+HdImRdPs9fyZZ4Xx2x+P0YTdk8PY9BZbLezx3WP8hiYF08C/G+LH581JMiF2sUXO043CM/pmQ0iPJMng9kCKr0FvjQOO+eWPv7NPY0mBT5fjtttuk7ZJibVfk7/dM1Bgs8jsF/ybIEUyDfYZdotBKaTTf4efTF1nZfN8g3O7KA9dTlyyTPFiZ7yiV2KPvAlCGnsvs8MzvkfIHTSVIuPPNSYJ3B2L5N+ckmdgP5K07wziXkTrWjjgMEJSQPummC7fL/NAAQ/TmOczzPQ0GRIwy7BJ4vz0mFUPuOHReMeZ/L8+M/gR5nGwsNJF0cHp75J9vgoVj/yT+/o/bC3J75RU3eeZOx1igzQJ9v0y1lT7fNuIITlS9JI/w8w688QjTgf4kbc6Z9GSsXwTRA57HTm4KFZdwfTsdhjWJK5//+BV8RDsZvv3waHxWemNZ19bOrLjs6xl38+ZYG855zYTXKSpl3P3zM9kwAf+zdpzSeQ3f+9WWORFYDvXUbGFhFQvyFtAIjFmuSbrmi90cnurMSuYRRhL7b//Is2OnReJsPGNv6P9Jf0TINE5n3wUgTj4/Emg4H7eDG13DI/gZgSVut3Mjt8jubn/bbbJiI4aoDYlGD2sB0mmCxAFvNv5QGyEZX5GYj856dS1kMxfLBuE2NgQzja8LvR5+k/n5w"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 8, "EERhBSI"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 16, "VCpCVqqSai21lZsbJimEbSSN"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 32, "qjWoVmzWzGgmmYwmGCqqmEnKTUeIKkwb1Wa0jKSaYA0hw1ZStKGKxs611qS2hkiWq1RDCkRqJaKRrbY0azQ0ZqXk6tQ"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 64, "ncxlxolZLwtOWy0Lm8yYMagtPTM1kVmRknFKmtC00pNl2cJTrQ2t23KdaSekpbSliSXXmqIW25LIlGVDyZ3FKcuZmrVX1mxwRFhaNSdLybRL7s5MSpKMJJISxZqxYZodpuI4qGStZMtRVmljcUTU5jZuBp3NNWFhUyZqrcZqS3aSTFYjaRebIlllThOpygyqmstTSYnpmlF5qViT3JFVrtoyW4dkNTjSrUZ0UrOWNWoypz3TqmlNzdLQMpFOSTIZpcwmilYm2JhaM5VadVziWmktrcy1WZZKFdpK5kWj2pIZKqlTpEwZdN9kWYrLODnSyLZWMnJmdq2ajDapS8Z2cqooSVOemLYl"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 8, "lJsTJQiMJJw"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 16, "VlaQwYrGzrWWpDSWSBSjTERKLKORqZIdazQkJqTk6tQ"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 32, "aTebIhllThKpyizqmklTSYnpGtFcoVmrnNBZpnIyW8MsZzjSqcY0UjK2PWomLQ2VilEmzVpbMpGuTSYZ5cbGiloy3Fh0W4daaW0qzLEpnUoV20JGRYfKhhkiqZK8Xol032RZLss4OZLKlnQScGZWrJqMZqlLzjZyqipJU56YtiU"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 64, "1DYyEcympk7SLRMm27rcqjapJC2lEj2LW69qXJ0ZxKrirGmy1JScZWwNzebSbM62lM+MOJtmWrrKpCKrpJnXsY2G8qye6pSxisjN5Vatm+eqSVJdySE1amVmImYTV8JksbMJtiWVqNSb0pWVpaG2pi5qCzqLWbcmqTplhhVJprJksuVupVReaNGUNjXsaEp7W9kstag1IqtX3WRKJxWlqolOOaWSpMxZCW8tTGElRxpKGrVMZ5lZ8VpuawaySk+TzV2ZOaq1o1gVs6ptlZpb9ZpGklE5r8XKq+RWmZImiUm4WVPVRi6pElZbWaN0TmXJMpZoVidXTuXtuswkiTFp4XFnGa6maWWViVU1CibZlhUyKWlZMumSE1quTGtaQobOa2UyrNxYWpIrubIsVldTcyaV5Y1mZqasNI1bM+pSSrIzTUXxHKsUm5Oi2snONomqE3EeVJ5ZrKRSSqtlyraYkqtTrklH+s7J0sauSSZKlZQlT5ZrRVGRF3XZdFImSzWbu/JmNilZ26xaIz0ljaPKrNY16SpVQ5LijFjhyuYWmliJ0zzpkCyvVKWklGYRS6VmhfLevpqbqk2dbEtZKCVLxotkMnQ6HXPMKmvVpFma6GpFaVjZGBZKrSIZE3kxVmWtVm3YTMU0mbVV6ZVER6UXpTWNlVM5V6stJVOtQVXVetY"),
            (FilterType::Triangle, HashAlg::Blockhash, 8, "3M0FbCH1SXo"),
            (FilterType::Triangle, HashAlg::Blockhash, 16, "Rmtwt5NMuVGZAHNooZz9P9IYk8R/t5BcqAnrPgTZ9pY"),
            (FilterType::Triangle, HashAlg::Blockhash, 32, "+JhF7rcRfDgFvjlWJBsrVxfiaTYPDVOQ509GlqOFS7dL0gFG+1tsEIc2YaxOjy2/UU7QWqXdmZaTiZ0GUt37CqRmUrOM8spCTlc2pHfVukj+Dy1JzuedykxTZBgEh/si2OhlgqulS0TKOvRP6twsFxJrVwUVaMD+gh2+J3zecM4"),
            (FilterType::Triangle, HashAlg::Blockhash, 64, "VVTBXKcKnFdaqibDc/X7xH8HZs/3W0SDJE0DI1BZyJ0zz7Xv4kg9sych6keNnokFcJyBuN+Xb4qkCs9lUYwcM7ZOh9SqOCkNj5GseqYmPL0dcOIyBD5Tz9+Fu4pXpoSVemr8PRp1r5dh2F/iaIC8CaeQE5hNAiKbSc8u5RNxPc+Pog2vrMRdczJ51nKRoJ7ajk3OV7IMBAl/9kZoT/UCYnHSFIsBNFKYJ0BeK1va9Fr/uLda0Dx3y+wXxUhpid0lSSZWsnC7fDvtG/g8DpOUwTaA42DDmyzBJ93129GgOsOPRaKodH9+kFlTbYqHgkEJRgFS387Wtphss/biT5/oFbCFZhwJF2tcPOx+mER0KZP2yRlHJfNMLBHiT/6s9G+i5yVdIjpPYk7oeJdmrOVwbjorItHOLWR5d6UTc4DGmpK8rLVB4phTkIen5wS9hWMm1XCxbOHJivCc0bt8Y6WuppjayHqU5tmHqjIXGxIfWlc1CS+wr+21SViRt8DtcQxaNGWcqmFtBuLFesLap2jOwe3qrqQnZCRheagz2YV0On5KgtUONP53YHz0XFUE430ZpATg4tE2IUPZ6lHX3mq/jYxWVZt+fjdADoduNDwboxKwy8Catfj0NhsnBDxK00fvBIX0UZT6WQxN54dK5Mc3baIuR3MkFn7asDvUx+M6nVA"),
            (FilterType::Triangle, HashAlg::Gradient, 8, "VEJCVpJqLDQ"),
            (FilterType::Triangle, HashAlg::Gradient, 16, "qjWoVkxWzFgmkhgmOBuoik3OjWsYCtU29IykjmQcIcM"),
            (FilterType::Triangle, HashAlg::Gradient, 32, "jMwlxolJLQ9MWzkLm+yYM7gpPbM001iRsnBOknCQUlMpWcRTLR1t0zCJOCWhZaSEhC3XmurWlNbMtMXpydmYrdOR3rRnUlZwZVubtatKzV5LxoxMihOFiqExGRum4jqoJK1kyXRWaWMxVNXidmZGnck1NUFRJkolx2JLcpZOVjM"),
            (FilterType::Triangle, HashAlg::Gradient, 64, "KuFisCVe2jSkzNopmVZaNO3RYySbFXOJi6TBIVpNZpWdIIpRWUnNlSBZx2LhgsuUi+VSa9TWiUJYpWEkRIWSnkmZ6U5blq2lQ1rDirPSEqcmnlW7pBML48S2jaJpMaNRlaULqqxtq0qc2yFtJtWWzVptMEWUcrPIMmeHtpxSjdQySadxppAcMUxJ5ZpUxIay5Cpjix2Uxm1TWeGm45QWqRspo6XIFtKcEKqLlckiOk1GRVMoSZaazRZBVTQp4lDNtplhRTrJJqt2mKsNZ7mW6lntXBZtUSrx0+U4qTFSMzPQsjWKM1su3GK6c01z0mFMybDL0a5tM4yW6pFVoiWsxE5ak8koSTPXm1Iya2apoMl7dGwppVhmvVKSIGusWUUTNGCKQYxBc0Z3vswypWUabcpUpXpjcaRVitGpWkxzTpvGUqrUWmVxmcRSdJbh9XKTysa+JnVNjubYyiqTstGuJEVuJjmbskzlZYiySZpFWxO0ZhHOrOKmTLzMStjkRmJDqwhv1bAmcus6VXJNtSyVeDYVVramNJcpNFsY9kYZmzYlm3ljUiuFVksbZkkyozS9KZIxuVCykeEdtqhabxqBiUfMa2kzG6sFUTMzRpLKqroWNXouSjazqiYVVZZkk1EiMlRZnF09Se2TcDJZ21URH1yjmCi2ew/LGc1qcnSrzcw"),
            (FilterType::Triangle, HashAlg::Mean, 8, "eQ9N/ff/eus"),
            (FilterType::Triangle, HashAlg::Mean, 16, "V2fQdN/V+RNcEnMs4f7Rv17eG+f/+7v96InpPsE5Ls4"),
            (FilterType::Triangle, HashAlg::Mean, 32, "uLlPHDeTfh4bvjl+BIs5J2dzYadvp3Gz58+akOOzBx/a2gjH+ntoAkU/cSwHj609GnrevofdmZeDec/PFvO//+y30OX8k/pDnwIWvBbPmtD/n5lJ9X+e/+9XMP/Nx/+6GPjBgMvuS0bHmfDHzNxMFQL5RweGQMDtng3Y4PydfOI"),
            (FilterType::Triangle, HashAlg::Mean, 64, "xdfFZ+0I+HdIm5cHu9/4ZR4Hx0R/P2YD9k8HY9BZ/J+yx3RP+hicF08A/G+LH54VFMiB2MUXO043CM/piR0iPJM3h/kCPj0FnjQOO6eWPv7NPb0mBz5fhvt9Os7ZNwbXfk7/dM1hgM8zsH/ybAEUwD/YZcotBKaHRf4e/TFxv9dN8gznrYA9N7nzyTPBAZ7yiV2OPvANCEnsvs8Mxv0fIHRS1ouLPNSZJzB2DQN+cgmNAP9I07xziWkD7WjigNMBSQPu2nD7/L/pAAQ/TmOczzOQ8GRIwy/Dp47z0mFQPuMHBeMfZ/J+8I/AR5vHwsdJF0cHp7/B9rgsFj/yT//p/bCnJ75RU3PeZOx1ikDwL9vk6FlT7fNuMITlS9bo/w8w78cQjjhP4k744Z9HTsXwTTEp/nTu4aFbNwPTsdhnCJK5//+Bx8RDsJ/v/wSHxWcmlZ39furJnu7xn39+JMH855rYfXqSpt/v3zs/swAf/zd5TSeQ3d+9WWORH4Dvf+ZGBhFQvyFtAMDBiuSbpmid0cnq6NSuYRRgT7L/3I92OnBetuPGJv4/9Bb0TQNA7n3wQnDj4/Emg4P7fHGx/DY/AYgRRvt3Mj88jubn/babLiI4aoDYFGD2qB0imChAlvPv5QGyEJP7Gaj+56dA0EMBfLBuU2NgQzja8L/T5/k/n9w"),
            (FilterType::Triangle, HashAlg::Median, 8, "eQ9N/ff/eus"),
            (FilterType::Triangle, HashAlg::Median, 16, "V2fQdN/V+RNcEnMs4f7Rv17eG+f/+7v96InpPsE5Ls4"),
            (FilterType::Triangle, HashAlg::Median, 32, "uLlPHDeTfh4bvjl+BIs5J2dzYadvp3Gz58+akOOzBx/a2gjH+ntoAkU/cSwHj609GnrevofdmZeDec/PFvO//+y30OX8k/pDnwIWvBbPmtD/n5lJ9X+e/+9XMP/Nx/+6GPjBgMvuS0bHmfDHzNxMFQL5RweGQMDtng3Y4PydfOI"),
            (FilterType::Triangle, HashAlg::Median, 64, "xdfFZ+0I+HdIm5cHu9/4ZR4Hx0R/P2YD9k8HY9BZ/J+yx3RP+hicF08A/G+LH54VFMiB2MUXO043CM/piR0iPJM3h/kCPj0FnjQOO6eWPv7NPb0mBz5fhvt9Os7ZNwbXfk7/dM1hgM8zsH/ybAEUwD/YZcotBKaHRf4e/TFxv9dN8gznrYA9N7nzyTPBAZ7yiV2OPvANCEnsvs8Mxv0fIHRS1ouLPNSZJzB2DQN+cgmNAP9I07xziWkD7WjigNMBSQPu2nD7/L/pAAQ/TmOczzOQ8GRIwy/Dp47z0mFQPuMHBeMfZ/J+8I/AR5vHwsdJF0cHp7/B9rgsFj/yT//p/bCnJ75RU3PeZOx1ikDwL9vk6FlT7fNuMITlS9bo/w8w78cQjjhP4k744Z9HTsXwTTEp/nTu4aFbNwPTsdhnCJK5//+Bx8RDsJ/v/wSHxWcmlZ39furJnu7xn39+JMH855rYfXqSpt/v3zs/swAf/zd5TSeQ3d+9WWORH4Dvf+ZGBhFQvyFtAMDBiuSbpmid0cnq6NSuYRRgT7L/3I92OnBetuPGJv4/9Bb0TQNA7n3wQnDj4/Emg4P7fHGx/DY/AYgRRvt3Mj88jubn/babLiI4aoDYFGD2qB0imChAlvPv5QGyEJP7Gaj+56dA0EMBfLBuU2NgQzja8L/T5/k/n9w"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 8, "EURSBSI"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 16, "VApCdiqSaiy0lZoLIimMLCSM"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 32, "qjWoVmxWzFgmmCwmGCuomknKTU+AKlwa1SY0jKSaZJwhw3YTkqOKw861xqSVhliWq1RDCmRpdKKRrfIkazQ0ZqHkasQ"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 64, "jcwlTolJLw9MSykLm8yYM5opOTM00ViRknFOmPC00pFl2cZTLR0t0zONaSWkwbiliSWWnqIW2ZLIlFXDyJ3FrcuZmrRTkuRwRVhaNWVLybRL6s5MSpaMDKoRxZqhIZgZpuI4qGStZMl1VmhjMUTU4jZmRpHJNSFBUSJaJcdmS3KSTlYzSRVbIhlkThIpygzqmslTaYnpGtk5qRmz3NFVpngyW4dkNxjSLcY0UruGNWoyJz3TqmkMzZpxMpEuSTAZpUymilZi2JhIMpVQcVSiSiAtlcy1UZ5KFZtK5gWjy9KZIqlytEyZdN9kWQrLODnXyLZ0NnBOdqxSjHaoD842YuqoaXOemDQl"),
            (FilterType::Triangle, HashAlg::VertGradient, 8, "kRsTIQwkJIw"),
            (FilterType::Triangle, HashAlg::VertGradient, 16, "dhKww4rHxrWWhDWWSRSjFEZKZKGRqZIZayQsZqHk4sQ"),
            (FilterType::Triangle, HashAlg::VertGradient, 32, "STVbIhlkThIp2gzqmslTSYnpGtEcoVmr3NBdpmAyG8IsZzjSu8Y0UjKmPUomLR3T6nEwzYpJMpmuzSYZZU7Emk4ynBh0WpdaYW0qzLUpnUo120pGBYPKhpmiqdK4Xpl0z2RZLss4OZfKtnQScG52rFLMdKgLzjZq6ipJc56YNCU"),
            (FilterType::Triangle, HashAlg::VertGradient, 64, "1DYyGcympk7TLBMnm7qcqhKpJi/lEj2LW6lqXJ0Z1K3irGm20JScpE2NzebSbs62lM+MuJtmWrrKpCKrhJnXsY2G8p2aqpCxisrMZdKpg6eqSUJdySE1amVmAmYTV+JksbMBtyW1vMSb0pWVpbGmJqxqCCqLXbcmqTplhgVJprJgmmUupVReadEUNjWkakp7U9kstagWJqtVzXRSpR0lKolPLeUSpczYCU8tTGEnRxpKGq0MZ5lR8VpoawazyEeTTV0bOaqts1kVs6pthZpb9ZpGklU5p83Lu2RWmZImiUG4WdPVRi6pElbaUSNkTmXJspZoVidXTuHpsswmqHBp4VlnHY6maWSVyVE9CibLxhEyK2lJdmvSEViuTGtOQoaKSy2yptxaWpIrGaIsVkdTcS6V5Y125vSsJY1VM2pSarIzDUXxXGoUm5Oi2MnONpiqU2EfZZ5RjKxTaottyvaYkqtCrklH+syZ0saOaCZKlZQlT5YrRVERFTXZdFpmSzGLu/JmNikZ2qxTYzUljaPKrNZxqSpXQpLijFDhyuYGmliN0TztkC6rRKWglGYRa7VmhfKctpobpk+ZbFtZOC1KxstkMnQ6PXOMKkvR5Vma6GpMaVhZGRKIrWYJm3k5ViStVs3YTMU0nbVVwZVFV6WHLTXFlVMZV4stJSel01XVGtI"),
            (FilterType::Nearest, HashAlg::Blockhash, 8, "3M0FbCH1SXo"),
            (FilterType::Nearest, HashAlg::Blockhash, 16, "Rmtwt5NMuVGZAHNooZz9P9IYk8R/t5BcqAnrPgTZ9pY"),
            (FilterType::Nearest, HashAlg::Blockhash, 32, "+JhF7rcRfDgFvjlWJBsrVxfiaTYPDVOQ509GlqOFS7dL0gFG+1tsEIc2YaxOjy2/UU7QWqXdmZaTiZ0GUt37CqRmUrOM8spCTlc2pHfVukj+Dy1JzuedykxTZBgEh/si2OhlgqulS0TKOvRP6twsFxJrVwUVaMD+gh2+J3zecM4"),
            (FilterType::Nearest, HashAlg::Blockhash, 64, "VVTBXKcKnFdaqibDc/X7xH8HZs/3W0SDJE0DI1BZyJ0zz7Xv4kg9sych6keNnokFcJyBuN+Xb4qkCs9lUYwcM7ZOh9SqOCkNj5GseqYmPL0dcOIyBD5Tz9+Fu4pXpoSVemr8PRp1r5dh2F/iaIC8CaeQE5hNAiKbSc8u5RNxPc+Pog2vrMRdczJ51nKRoJ7ajk3OV7IMBAl/9kZoT/UCYnHSFIsBNFKYJ0BeK1va9Fr/uLda0Dx3y+wXxUhpid0lSSZWsnC7fDvtG/g8DpOUwTaA42DDmyzBJ93129GgOsOPRaKodH9+kFlTbYqHgkEJRgFS387Wtphss/biT5/oFbCFZhwJF2tcPOx+mER0KZP2yRlHJfNMLBHiT/6s9G+i5yVdIjpPYk7oeJdmrOVwbjorItHOLWR5d6UTc4DGmpK8rLVB4phTkIen5wS9hWMm1XCxbOHJivCc0bt8Y6WuppjayHqU5tmHqjIXGxIfWlc1CS+wr+21SViRt8DtcQxaNGWcqmFtBuLFesLap2jOwe3qrqQnZCRheagz2YV0On5KgtUONP53YHz0XFUE430ZpATg4tE2IUPZ6lHX3mq/jYxWVZt+fjdADoduNDwboxKwy8Catfj0NhsnBDxK00fvBIX0UZT6WQxN54dK5Mc3baIuR3MkFn7asDvUx+M6nVA"),
            (FilterType::Nearest, HashAlg::Gradient, 8, "x1qyqjKLtiY"),
            (FilterType::Nearest, HashAlg::Gradient, 16, "rcZajVpURmrulRqyrZyiMqmopirWLlG3l9TjpmZSssk"),
            (FilterType::Nearest, HashAlg::Gradient, 32, "olSkMik5KjvUtCZtNFm4pzQ5E9ccnlWpvWVlqlNLIXerZiazpZWK0iS2S2k4WaY0RQWFa5PSVltKicxUZTWZL6XyqOYrs3ZqqullladSVbtUxgpm6TXmoplkbVUs2opUG2WObDBrJA51t7WmdlSvqtiUqmI3KxqppfTYcJFqN5E"),
            (FilterType::Nearest, HashAlg::Gradient, 64, "qutqgqkUVUnU9SodaWqad6U2RDWPM3OpmtSptdpR2uGMNlsorVgTpzoLtyyWG6fVzXVW7jRFzRlN1SSVp6q4ijVlaSlRtalkRVKNz6CqUClueHRNbbqJWllaqWFVIrO0o0NFnWUUyWYam3J9nbYWzThRGVRN21NrMyqFhNVUk6z2rOVIJWPNlStrzqNTVa2bpjYKDpWSXA0KZaumodTWTbYwIs3WUpTVkpqGhFdqKWY2Hlsu1ZkRS3RburQyRWVVps2onFNrzpYllmxTVyWU5knValKl2TqrjRV76fBWnFNIVaXWG0tKlVZyWtnV0q4dSebJkYxJs5vUYolVyjVZpZOTUW8bK1ZVGktlpKW1rso1tUY5dahHtWYK4y6tWEsbKVvKWawjVmWdqmazqfESVUq3q6xlFVVVqxppWj3TTdtVV7Fb6zGcKpKlZpUVt1JTMhqqKlJPplaUJkqxsoSrbFWupDKbtpBljlhLJRqraptaYzFMqdK1X6jmTlFKRsUmiW5sVVUpIm0pUrYNpcpF2K21VCpydEU7tFmtkp0Ui1IkyktrUkalVsgWolk2txRaEpJROdpy23mbnKaK2U4riWrWKm0ZWStKmXJRShYp7qpVp6m0WTfpqjU1Vta2K+ksl6pLaS41TcqNSeo1Lk2aRZS7tLSyaiarK6Yyqjivlto"),
            (FilterType::Nearest, HashAlg::Mean, 8, "TZwCstVF88A"),
            (FilterType::Nearest, HashAlg::Mean, 16, "0YnDJG8XhVtEF+P0LfAlLQ1uQwadWHDdKsngzt5xHq8"),
            (FilterType::Nearest, HashAlg::Mean, 32, "PJ7P/qvHYmuXgeMDPrsnKY4uzcRrLPkEpr3sPJJ4B2pghWOk1zFakXxiqHi4rRtwejM2zV3yI9ekIvhyU56S5VZWgeTR9PbLvuvftWLDt//YgVLPdWm9LsY2F0PVxvdQWLooxAr1Djf4KlbhcsQcBWLpLEFWXlalyMXwY3AZ7rg"),
            (FilterType::Nearest, HashAlg::Mean, 64, "TAzV/ygcdPtL78mFmW8aZsYIf+FRJXchkEkHW2qG2AEorjQXHlhzrtsVCm9vB+TfAPSY6dfdmXu3msAHi3Twnxjtld1KJK9l3h++8UIIB+vJumfu6S8Z8ti0fvP4hKYwUAMtG3Axn8Zfk1J36LMU3ZZQtdoNeh9OZW7DrDkT8ixOiy3az5cNNZjZpTccy7izxg8Ob65teAmvPMLyX72Gz5d+caOTZZXd4qFWiTjlOEy3t795V2wRCi1JjHHkgUFVD28K2GTTiLdWv9Q2WseUzjyaZhNgkiAvjwzDOrHYEFOugdR99NUaNZbC1++y5qg516GDp7EpI5tU8A+aAWn7/ZF3YnTI0fx/c0rdAkU8r4otQmBh7/AH4WOBW0mo7Q4yM0+KK7PMdG0BQLeXeGwQfXPLsRzDP10FNIvdw0nHXNIU/zOXRwYxKI+I69G9pUZ2FYWpbzILrlb0ruvYZuG/7YjanHtTN7Ntg4gboYIVapP+LWsa+S83UKOB1/fZdsDnwWFC9uEPJszESo7Kp3gF2Oj35rTqLER4N6yWny5Em1aqgOUAIEsHUCTUCXOJYxDYUzJiy8a4mvHnJrEUxF5vD44FzuP35Woa6y6/9bow7Ko8y6KNJQHPqgAyHD1Ii/nvgSUQldd7XMq466hkjLoSXegVKTgvIg4uPhPXkP7Otpg"),
            (FilterType::Nearest, HashAlg::Median, 8, "TZ4DstXl88A"),
            (FilterType::Nearest, HashAlg::Median, 16, "0YnDJG+XhV9EF+P0LfAlLQ1uQwadWHDdKsngzt5xHq8"),
            (FilterType::Nearest, HashAlg::Median, 32, "PJ7P7qvHYmuXgeMDPrsnKY4uzcRrLPkEpp3sPJJYB2pghWOk1zFakVxiqHi4rRtwcjM2zV3yI9ekIvhyU56S5VZWgeTR9NbLvuvflWLDt//YgVLPdWm9LsY2F0PUxPdQSLooRArlDjf4KlbhcsQcBWLpLEFWTlalyMXwY3AZ7rg"),
            (FilterType::Nearest, HashAlg::Median, 64, "TAyV/ygcdPtL78mFmW8aZsYIf+FRJXchkEkHW2qG2AEorjQXHlhzrtsVCm9vB+TfAPSY6dfdmXu3msAHi3Twnxjtld1KJK9F3h++8UIIB+vJuifu6S8Z8ti0fvPwhKYwUAEtG3Axn8Zfk1J36LMU3ZZQtdoNeh9OZW7DrDkT8ixOiy3az5cNNZjZpTccy7izxg4Ob65teAmPPMLyX72Gz5c+caOTZZXd4qFWiTjlOEy3t795V2wRCi1JjHHkgUFVD28K2GTTiLdWv9Q2WseUzjyaZhNgkiAvjwzDOrHYEFOugdR99NUaNZbC1++y5qg516GDh7EhI5tU8A+aAWn7/ZF3YnTI0fx/c0rdAkU8r4otQmBh7/AH4WOBW0mo7Q4yM0+KK7PMdG0BQLaXeGwQfXPLsRzDP10FFIvdwwnHXNIU/zOXRwYxKI+I69G9pUZ2FYWpazILrlb0ruvYZuG/7YjamHtTN7Ntg4gboYIVapP+LWsa+Ss3UKOB1/fZdsDnwWFC9uEPJszESo7Kp3gF2Oj35rTqLER4N6yWny5Em1aqAOUAIEsHUCTUCXOJYxDYUzJCy8a4mvHnJrEUxF5vC44FzuP35Woayy6/9bow7Ko8y6KNJQDPqgAyHD1Ii/jvgSUQldd7XMq466hkjLoSXWgVKTguIgwuPhPXkP7Otpg"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 8, "NN/slHM"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 16, "zxSRU5ayMSzWayOuqmqLTXWa"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 32, "S6R4aK1Ku02qmlf2bmWq7avb5UVaDdd6kpXUBazSalmyyWmrVJNMyjpNKmoQSslsNVPLBKNI27Y6G2bKp0YpRnXFTK4"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 64, "2iqXcRJVqdSTijPRUtTso1Iyeaq1MnKfr3Bs12qgepdjjcq2JVVFu/l5JS2nSlJUMmVlFKbO6tbJWteaKsrFLcPIsrhWsmo6VFqqLiVVUqobrWhrtbnmSpPdxKX2KKVFqkJZOWV11o6pzFKTsqzVck0qV6XYlKpiNysaqaX02HCRajeRRL3TWtqiikvNzqi0iZVlramxhMoLaSm7U02uYmxoSw0it/hSIuk0ESox0agtScVUylTuqoWlJsueo0lrtaMyc7VSKlOpa6tEWpaCmrJpFI2h2rlEWKaiYlmqS7lLL6Voo5YTVSJ7JaomuUWXd0pjrTtrVql7HcWizGGtbrpWSVa7EDWh"),
            (FilterType::Nearest, HashAlg::VertGradient, 8, "VHFeIS1RbLI"),
            (FilterType::Nearest, HashAlg::VertGradient, 16, "WwtSa6vWNbOT6l5V2hNtOdpSq6TZ6NXKSk2k7qtaTK4"),
            (FilterType::Nearest, HashAlg::VertGradient, 32, "ySq1rDJBoxrFsdMraxUnRa3K0aG0oo2VlFwnM0NOs2Wp0shStJWOJFGrLCHVIieVyJYztUiqVN23jJQs06plReaq3knV1FShW1uzmYro2NimmGlijVwj2UonL+svrSliMqX7Gc0qmEXFpJQ5KmMvxnsdxaLMYa1uulZJVrsQNaE"),
            (FilterType::Nearest, HashAlg::VertGradient, 64, "pMYyGSlxzZrrjm3aSppMwaDqZFatUKvVdGmpS1VZVTH2blSy1ZVOtk3XjFWysq52DdetkkmVUVIurqIrZ6SXENPSSUk2qqV1ydguViVp1qpyktqUVW6lWVFVi3KWdYuis6VpMqWtJG2ZmqkVprnFmmTOqniKdbe0TbbkJi1tLbR1umVsplWzyvOUFS2l2mqhU1XVWbJ3BqlFhaZWq02iK27rLWUzaY7aTVGyWet21jLKJifFJJNUlVLJ6kavzZq1a2U7taj1aEl5sRqVVZprVathgtWarKwSulUj3U6aSma8TFtVqq6k2krGVnvJTDVptS6UlB2raatUuay6qZ1rldpQiqaG7TFVjJZlyyvpalsVzEaNWCVLddmq1nIqaiUqqy2lUpmpqNGjzaguLUkV8U20be60VfXMJX2VtbjIdrhKTNUgW041TZpsK0aV4pgoW7WJTE0u5dSu0snKSSrTmnJVNaWHDuKaU1W5JqtUJzOsqh6pbdqqmVVYYppqyXatqbY2xk0124RWIi9FpSpJtCZNqrK2WJVSyGK1+ratzkouy1QlGaUommXOUCXVTLGpmWpqolOxplxLZEZWLiVqx0ZEttJqam6qSsGa0lzarlVcNU1cslmJKkU9uVRVEqV1Xd2aR1mdKrk1p7bVWS9VT/ROUpVXZjmpE1W9il12a9s"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 8, "3M0FbCH1SXo"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 16, "Rmtwt5NMuVGZAHNooZz9P9IYk8R/t5BcqAnrPgTZ9pY"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 32, "+JhF7rcRfDgFvjlWJBsrVxfiaTYPDVOQ509GlqOFS7dL0gFG+1tsEIc2YaxOjy2/UU7QWqXdmZaTiZ0GUt37CqRmUrOM8spCTlc2pHfVukj+Dy1JzuedykxTZBgEh/si2OhlgqulS0TKOvRP6twsFxJrVwUVaMD+gh2+J3zecM4"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 64, "VVTBXKcKnFdaqibDc/X7xH8HZs/3W0SDJE0DI1BZyJ0zz7Xv4kg9sych6keNnokFcJyBuN+Xb4qkCs9lUYwcM7ZOh9SqOCkNj5GseqYmPL0dcOIyBD5Tz9+Fu4pXpoSVemr8PRp1r5dh2F/iaIC8CaeQE5hNAiKbSc8u5RNxPc+Pog2vrMRdczJ51nKRoJ7ajk3OV7IMBAl/9kZoT/UCYnHSFIsBNFKYJ0BeK1va9Fr/uLda0Dx3y+wXxUhpid0lSSZWsnC7fDvtG/g8DpOUwTaA42DDmyzBJ93129GgOsOPRaKodH9+kFlTbYqHgkEJRgFS387Wtphss/biT5/oFbCFZhwJF2tcPOx+mER0KZP2yRlHJfNMLBHiT/6s9G+i5yVdIjpPYk7oeJdmrOVwbjorItHOLWR5d6UTc4DGmpK8rLVB4phTkIen5wS9hWMm1XCxbOHJivCc0bt8Y6WuppjayHqU5tmHqjIXGxIfWlc1CS+wr+21SViRt8DtcQxaNGWcqmFtBuLFesLap2jOwe3qrqQnZCRheagz2YV0On5KgtUONP53YHz0XFUE430ZpATg4tE2IUPZ6lHX3mq/jYxWVZt+fjdADoduNDwboxKwy8Catfj0NhsnBDxK00fvBIX0UZT6WQxN54dK5Mc3baIuR3MkFn7asDvUx+M6nVA"),
            (FilterType::CatmullRom, HashAlg::Gradient, 8, "VkpCRppKLDQ"),
            (FilterType::CatmullRom, HashAlg::Gradient, 16, "qjGoVkxWzFgmkpgmOJuqmE3OiWsaCs12tIyknmAYIcM"),
            (FilterType::CatmullRom, HashAlg::Gradient, 32, "jcwlxolJLwtMWz0Lm+y4MbgpPTM001iRkmBOmlCQUpMtScRTrR0t0zKJOCWhZaSErC3XmurWlNbMtEVpydmardOR3rVFUlRwZVubtatrzV5LhoxMmhLFmrExWxui4rqqLK1kyVRWaWdxVNXiNmZWnc01JUFZJmKlx2JLdpJMViM"),
            (FilterType::CatmullRom, HashAlg::Gradient, 64, "KuNqsiVe2hakzNopmVZaNO3ZYyabFXOJi6SBJVpNZpXdIIpVWUnNlSBZQ2KlgouUi+VWa9RWiUpYpXEkTIWSnkmb6U5Llq2lQ1qDiLPSEqemnlS7phJL48S2jaJpM6NVlSULqqxtq0q8yiFpJtWWzFptME2UZrPIMmeHtphSjdZyWadzJrAcMUxJZZpUzYay5Cpjix20xG1TWaGmYpQGqRspoqXIFlKcELqLlcliOm1GRVMoWZaazRZFVTQp4lDNtplhRTrJJqt2mKsNZ7mU6lmtXDZtUSr10+U6qTVSMzPSsjWKM1su3CK6c01z0mFMybDJ0a5tM4yW6hFVoiWshE5ak8moSRPWkxIya2aroMlbZCwppVhmvVKSIWusWUUTtGCKQ4xBc0Z3rswypXUabcpUpXpjeaRditGpWkxzSpvGUqvUWmVRmcVSdJbh9HKTisaWJ3VNrubYSiuTstWOIEVsJhnbskzlZYiySZpFWxO0NhHO7OKmTL3MytrkTGJD6wgv1bQmUus6VXNNti6VWHY1VraiNJcttFtY9kYZmzYlmXljUiuFVksbZkkyozS9KZIxuVCykcEVtqhabxqBiUXMa2kzG6sFUTMzRpLKqroeNWouSjazuiaRVJZkk1EjElRZmN08SO2TcDJZ21ERP1yzmCi2Ww/bGc1qc2Srzcw"),
            (FilterType::CatmullRom, HashAlg::Mean, 8, "eQ9N/ff/eOs"),
            (FilterType::CatmullRom, HashAlg::Mean, 16, "V2fQdN+VuVNcEnMs4f7Rux6cG8f/85v96InpPsE5Ls4"),
            (FilterType::CatmullRom, HashAlg::Mean, 32, "uLlPnDeTfB4dvjl+BKs5J2dzYSdrpnOzp8+akOOjBx/Y2gjX+ntoAkU/cSwHj609GHrevoXNmZeDec/LFvO/7+w30PX4k/pDj7IWvBbPmlD/n5lJtW+ev+5XMP9Nh/+qGfjBgMvuS0bHmfDHzdwMFRL5R0eGYMD9ng3Y4PydfOY"),
            (FilterType::CatmullRom, HashAlg::Mean, 64, "1dfVZ60o+HdImxdHu9/6ZR4Xx2R3P2YDdk8PY9BZTL2yx3SP8hicF08C/G+LH58VFMiB2sUXO043CM/piQ0iPJOng9kCKq0FvjQOO+eWPv7NPe0mBT5fjtt9Ok7ZNybVfk7/dM1Bgs8jsH/ybIEUwDfYYcotRKaXTf4efTF1v9fN8gzn7KA9NznzyTPFgZ7yiV2KPvANCEnsvs8Mzv0fIHTSVIuLPNSZJzB2LRN+ckmdgP5I07wziWkDrWjigNMBSQfumnC7/L/JAAQ/TmOUzzPQ4GRIwy7Bp4rz0nFUPuOHBeMfZ/J88M/AR5nGwsdJF0cHp7/J9rgsFj/yT//o/bCnJr5RU3PeZOx1ikzUL9vkytlT7fNuMITlS9bo/w8y68cQjzhf4kbs6Z9GTsXwTTA5/nTu4aFZNwOTsdhnWJK5//+B18RDsZvv3waHxWemFZ19fOrLju7xn39+ZYG855jYbWqSpl3P3zc9kwAf+zd5TSeQ3f+9WWORFYDPf8ZGFhFQvyFtAMjFmuSbpmid0cnq6MSuYQRhT7L/3Is2OmReNsPGJv4f9Bb0TANE5n3wQkTj4/Emg4P7fDG13Do/gZgQVut3cjs8jubn/babJiI4aoDYFGD2qB0mmCxIFvPu5QGyEJf5GYj+56dQ1EMBfLBuU2NgQzja8LvT5+k7n5w"),
            (FilterType::CatmullRom, HashAlg::Median, 8, "eQ9N/ff/eOs"),
            (FilterType::CatmullRom, HashAlg::Median, 16, "V2fQdN+VuVNcEnMs4f7Rux6cG8f/85v96InpPsE5Ls4"),
            (FilterType::CatmullRom, HashAlg::Median, 32, "uLlPnDeTfB4dvjl+BKs5J2dzYSdrpnOzp8+akOOjBx/Y2gjX+ntoAkU/cSwHj609GHrevoXNmZeDec/LFvO/7+w30PX4k/pDj7IWvBbPmlD/n5lJtW+ev+5XMP9Nh/+qGfjBgMvuS0bHmfDHzdwMFRL5R0eGYMD9ng3Y4PydfOY"),
            (FilterType::CatmullRom, HashAlg::Median, 64, "1dfVZ60o+HdImxdHu9/6ZR4Xx2R3P2YDdk8PY9BZTL2yx3SP8hicF08C/G+LH58VFMiB2sUXO043CM/piQ0iPJOng9kCKq0FvjQOO+eWPv7NPe0mBT5fjtt9Ok7ZNybVfk7/dM1Bgs8jsH/ybIEUwDfYYcotRKaXTf4efTF1v9fN8gzn7KA9NznzyTPFgZ7yiV2KPvANCEnsvs8Mzv0fIHTSVIuLPNSZJzB2LRN+ckmdgP5I07wziWkDrWjigNMBSQfumnC7/L/JAAQ/TmOUzzPQ4GRIwy7Bp4rz0nFUPuOHBeMfZ/J88M/AR5nGwsdJF0cHp7/J9rgsFj/yT//o/bCnJr5RU3PeZOx1ikzUL9vkytlT7fNuMITlS9bo/w8y68cQjzhf4kbs6Z9GTsXwTTA5/nTu4aFZNwOTsdhnWJK5//+B18RDsZvv3waHxWemFZ19fOrLju7xn39+ZYG855jYbWqSpl3P3zc9kwAf+zd5TSeQ3f+9WWORFYDPf8ZGFhFQvyFtAMjFmuSbpmid0cnq6MSuYQRhT7L/3Is2OmReNsPGJv4f9Bb0TANE5n3wQkTj4/Emg4P7fDG13Do/gZgQVut3cjs8jubn/babJiI4aoDYFGD2qB0mmCxIFvPu5QGyEJf5GYj+56dQ1EMBfLBuU2NgQzja8LvT5+k7n5w"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 8, "EERhBSI"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 16, "VgpCdqqSaiS0lZsbZimMbCSN"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 32, "qjWoVmxWzFgmmAwmWCuqmknKTU+IKlwa1SY0jKSaYBwhw1ZStKOKws611qSVhkiWq1RDCmQpNKKRrbI1azQ0ZqHk6tQ"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 64, "jcwlxolJLwtOWy0Lm8yYMbotPTM10ViRknFKmPC00pEl2cRTrR2t0zONaSekoayljSWXmqIW25LIlFXDyJ3FrcuRmqVXlkRwRFhaNSVLybRL7s5MSpaMJLoRxZqxIZoZpuI4qGStZMtxVmlndUTU4jZmRpnNNWFBUSJarcdiS3aSTlYjaRWbIhlkThMp2gzqmslTSYmpmlE5qRmz3JFVpnJyW4dkNRjSrcZ0UrOGNWoypz3T6mFNzZLRMpEuSTIZpUwmilYi2JhIMpVacVSiSiAtpcy1UZ5KFdtK5kWj2pKZIqlzpEyZdN9kWYrLODmXyLZUMnJudqyajHaoS8Z2cqqoSVMemLQl"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 8, "lZsTIQisJIw"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 16, "VlKQocrGzrWWhDWWSBSjTEZKJKORqZIZazQsZqXk69Q"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 32, "STebIhlkThKp2izqmslTSYmpGtEcoVmrnNBZpmIyG8MsZzrSqcY0UjK2PWomLR3VynEgzUpJMpGu3SYZZU7Wmloy3Fh0WpdaYW0qzLEpnUo120pWRYPKhpmiqdK8Xpl012RZLss4OZPKtnSScGZWrJrMZqlLzjZyqihJc56YtCU"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 64, "VDYyEcympk7SLRNnm7rcqjapJi3lEj2LW6tqXJ0ZxarirGmy1JScpG2NzebSbs62lM+MuJtmWrrKpCKrhJnXsY2G8pya6pSxisrM5VKpm6eqQVJdySE1amVmAmZTV8JksbMBtSW1vNSb0pWVpaGmJq5qCiqLXbcmqTplhhVJprJkumVupVReaNGUNjXsakp7W9kstag1JqtV3XRSJRslqolOOaWSpcxdCU8tTGElRxpKGqVMZ5lZ8Vpoa0azyEeXzV0ZOaqts1gVs6ptlZpb9ZrGklU5p8XLq+RWmZImiWm4WVPVRi6pElZbUaNkTmXJspZoVidXTuGpsswkqTFp4VlnGY6maWWVyVU9CibZhhEyKWlZNmvSEViqTGsaQobKa2UypNwaWpIrGaIsVkdTcy6V5Y1m5vSsJc1ZM2pSarIzDUXxXGoUm5Oi2snONpiqU3EeZJZRrKxSaotlyvaYkqtTrklH+s7J0sauaCZKFZQlT5YrRVGRF3XZdFomSzWLu/JmNikZ26xTIzUljaPKrNZx6SpXQpLijFrhyuYGmliJ0zztkGyrRKGklGYRa6VmhfKctpqbpk+dbEtZOC1KxstkMnQ7PXOMKkvR5VmaaGpFaVhZGRYKrSYZm3k5ViWtVsXYTOU0mbVVyZVEV6UFpTXFFVM5V4stJXetQ1XVWtY"),
            (FilterType::Gaussian, HashAlg::Blockhash, 8, "3M0FbCH1SXo"),
            (FilterType::Gaussian, HashAlg::Blockhash, 16, "Rmtwt5NMuVGZAHNooZz9P9IYk8R/t5BcqAnrPgTZ9pY"),
            (FilterType::Gaussian, HashAlg::Blockhash, 32, "+JhF7rcRfDgFvjlWJBsrVxfiaTYPDVOQ509GlqOFS7dL0gFG+1tsEIc2YaxOjy2/UU7QWqXdmZaTiZ0GUt37CqRmUrOM8spCTlc2pHfVukj+Dy1JzuedykxTZBgEh/si2OhlgqulS0TKOvRP6twsFxJrVwUVaMD+gh2+J3zecM4"),
            (FilterType::Gaussian, HashAlg::Blockhash, 64, "VVTBXKcKnFdaqibDc/X7xH8HZs/3W0SDJE0DI1BZyJ0zz7Xv4kg9sych6keNnokFcJyBuN+Xb4qkCs9lUYwcM7ZOh9SqOCkNj5GseqYmPL0dcOIyBD5Tz9+Fu4pXpoSVemr8PRp1r5dh2F/iaIC8CaeQE5hNAiKbSc8u5RNxPc+Pog2vrMRdczJ51nKRoJ7ajk3OV7IMBAl/9kZoT/UCYnHSFIsBNFKYJ0BeK1va9Fr/uLda0Dx3y+wXxUhpid0lSSZWsnC7fDvtG/g8DpOUwTaA42DDmyzBJ93129GgOsOPRaKodH9+kFlTbYqHgkEJRgFS387Wtphss/biT5/oFbCFZhwJF2tcPOx+mER0KZP2yRlHJfNMLBHiT/6s9G+i5yVdIjpPYk7oeJdmrOVwbjorItHOLWR5d6UTc4DGmpK8rLVB4phTkIen5wS9hWMm1XCxbOHJivCc0bt8Y6WuppjayHqU5tmHqjIXGxIfWlc1CS+wr+21SViRt8DtcQxaNGWcqmFtBuLFesLap2jOwe3qrqQnZCRheagz2YV0On5KgtUONP53YHz0XFUE430ZpATg4tE2IUPZ6lHX3mq/jYxWVZt+fjdADoduNDwboxKwy8Catfj0NhsnBDxK00fvBIX0UZT6WQxN54dK5Mc3baIuR3MkFn7asDvUx+M6nVA"),
            (FilterType::Gaussian, HashAlg::Gradient, 8, "RAJCFppiLDQ"),
            (FilterType::Gaussian, HashAlg::Gradient, 16, "qjWoFkxWzBgmkhgmOBqomg3OjWsICs029IwkjmQYIcM"),
            (FilterType::Gaussian, HashAlg::Gradient, 32, "jcwlRolJLw9ISzgLm+wYM7gpODMw0ViRsmFKknCQUpMlScRTJR0t0zCJOCWBZaSEhC3XmsjG1NLIlMXhydnYrdOR3rRnUFZwZ1ibtMtDzUxLxsxMmhPFkqEhmRum4RiKJK9kyHSWaWNxRFTiNkZGkck1IUFRJmAlx2JLcpZOFjM"),
            (FilterType::Gaussian, HashAlg::Gradient, 64, "KuFgsCVe2jSkzMKhmVZaNOXBYySbFXOJiaDBMVpNZpWcYIpxWUjNlTBRQ2LhgouUi+VSa/TWiUZIpWFkxIaTnkmb6UxLlo+lQxrDiLPSE6dmnlW74BML48S2DaJoMqPRnacLqqxkq0OcyyFtJtWWzFJtMG2UcrPAMmfHtpwSjZIySadzJrAcM0xIZZpUxIYy5Cpjix2ExmlzWWGC45QGqRspo6XoFlKMEL6LhckiOkxGTVOsSZa6zRZBVTQpwlDJtplhRDrJpot0mOsNZ7me4lntfDdtUSrx0eU4qTFaO7HQsjGKM1sm3GK6c01zUmNMybLB0aptM8yWy5FZomWsxF5ak8msSTPHm1Iya2a5oMkbdCwpp1kmvVKSKGusWUUzNGCKQ4xBc0Z3/s5ypWUabcpUpTpjcbRNitGpWE5xSpvOEqrUWmVxmcRSJNbh9HKTysYeJ3VNhubYyiqXM9GOIEduJhnbskzlZYyySZrHWxG8ZhPc7OKGTLzMStjkRmJDqQgu1bAmcms6VXJNtiyVeDY1VraiNIcoNFlY9mYZmzYlmXhjUisFVmsbZkky4zS9KZY4uFCyseEVNrhabhqDiUfMO2kzGqsFUTMzRpLIKroGNXouSj6zuiYRVJZkk3EiMlRZmN09SS2TcLJY21EZH1yzmCi2ew/bGc1ocGSrzcw"),
            (FilterType::Gaussian, HashAlg::Mean, 8, "fQ9N/ff/fOs"),
            (FilterType::Gaussian, HashAlg::Mean, 16, "V2fQdN+V+RF9EHEs4f7xvx7eH+f//7v/6BnpPsE7bs4"),
            (FilterType::Gaussian, HashAlg::Mean, 32, "uLlvHD+Tfh4fvjk+ALsxJ2fzYSdv53Gz588akOPzBx/K+gjH+ntoAkU/cSwHj609G3rcvof9mZeDec/PFvO/7/y30Of8k/pDnwMWvBePnnj/n5lJ//+e/+9XMP9Nx/++GPjBgMvsQ0bHmfDHzNhMF4L5TwcGQMDtng3Y4PydfOY"),
            (FilterType::Gaussian, HashAlg::Mean, 64, "wdfBZ+8A+HdIm8dH+9/4ZR4Hh0f/P2YD9g8HY9BZ/J+zx3RP8hicF08A/G+DH54VBIiB6MUfOw43CM/piR0uPJMnh/kCPD8FnjQOO6eeP/7NPb4mBz4fhvt9Os7JdwbHfk7/dMlhgM8zkH/ybAEUwD/YZ8g9AKaHRf4e/TFwv/dN8gzn7YA9c7nzyTPhAR7yiV2OP/ABDEH8/s8Mxv0eIHTS1ouDPdSZJzB+DwN+8gmPAP9I07zziXkD7WjigNMBSQPu2nD7/L/JAAQ/TuOczzOQ8WRIwz/DJ4rz0mNQPuMHBeMfZ/J+8I/AR5/nwsdJF0cH5//D5/g8Bj/yD//p/bCnJ75RU2PeZOx1gkDwL9vk6VlT7fNuMITlS9bo/w8w78UQDjgP4k784Z9HTkXwTxApvnDu4aBbPwfTsdjnAJK9//+Bx8RDsJ/v/4zHxWeWtZ19fu/Jnu6xn39+ZMH+55zYfXqS5t/v3zs/kwAf/z953SeQ3f/9WWORHYDvf8ZGBhFQvwNtAMDBAuSbp2ic4c3q6NSuYRRgT7L/3I83OnBetsPGJP4/9Bb0SANA5n3wQnDh4/Emg4P7eHCz/DY/A4wRRv9/Mj88jubn/bbbLiI4aoDYBGD2qB0ikChAEvPv5QGwEJD7Gez856dA1GMRfPD+02NgQzja8P/T5+g/n5w"),
            (FilterType::Gaussian, HashAlg::Median, 8, "fQ9N/ff/fOs"),
            (FilterType::Gaussian, HashAlg::Median, 16, "V2fQdN+V+RF9EHEs4f7xvx7eH+f//7v/6BnpPsE7bs4"),
            (FilterType::Gaussian, HashAlg::Median, 32, "uLlvHD+Tfh4fvjk+ALsxJ2fzYSdv53Gz588akOPzBx/K+gjH+ntoAkU/cSwHj609G3rcvof9mZeDec/PFvO/7/y30Of8k/pDnwMWvBePnnj/n5lJ//+e/+9XMP9Nx/++GPjBgMvsQ0bHmfDHzNhMF4L5TwcGQMDtng3Y4PydfOY"),
            (FilterType::Gaussian, HashAlg::Median, 64, "wdfBZ+8A+HdIm8dH+9/4ZR4Hh0f/P2YD9g8HY9BZ/J+zx3RP8hicF08A/G+DH54VBIiB6MUfOw43CM/piR0uPJMnh/kCPD8FnjQOO6eeP/7NPb4mBz4fhvt9Os7JdwbHfk7/dMlhgM8zkH/ybAEUwD/YZ8g9AKaHRf4e/TFwv/dN8gzn7YA9c7nzyTPhAR7yiV2OP/ABDEH8/s8Mxv0eIHTS1ouDPdSZJzB+DwN+8gmPAP9I07zziXkD7WjigNMBSQPu2nD7/L/JAAQ/TuOczzOQ8WRIwz/DJ4rz0mNQPuMHBeMfZ/J+8I/AR5/nwsdJF0cH5//D5/g8Bj/yD//p/bCnJ75RU2PeZOx1gkDwL9vk6VlT7fNuMITlS9bo/w8w78UQDjgP4k784Z9HTkXwTxApvnDu4aBbPwfTsdjnAJK9//+Bx8RDsJ/v/4zHxWeWtZ19fu/Jnu6xn39+ZMH+55zYfXqS5t/v3zs/kwAf/z953SeQ3f/9WWORHYDvf8ZGBhFQvwNtAMDBAuSbp2ic4c3q6NSuYRRgT7L/3I83OnBetsPGJP4/9Bb0SANA5n3wQnDh4/Emg4P7eHCz/DY/A4wRRv9/Mj88jubn/bbbLiI4aoDYBGD2qB0ikChAEvPv5QGwEJD7Gez856dA1GMRfPD+02NgQzja8P/T5+g/n5w"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 8, "EERABSI"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 16, "VAJCJhKYKiw0kZoZISGMJCSM"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 32, "qjWoFmxWzFgkmAwGGCqomEnKTU+IakgaxSakjKSYZIwhw3YSkqPKw86zxoSVhgiWK1RDCmQoNKKRrbIkazQ0JqDk4sQ"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 64, "jcwlxolJLQ9KSy0Lm8yYM7ApODM0kRiRknFKmPCw0pMl2cRTLR2N0zOJKSegobykjSWWnqIW3ZLIhEXDyJ3EqcuZmqRTklxwZVhaNGNJybRLwM5MSpKMBLoRxZqjIZobpuI4iCStYMl1VmljcUTU4jZmBpHJNWFBUSZKJcdiS3KWThYjeTcbIhlmThIpygzqmslTSZnpGtEZqRmT3JBZpnByG4dkdBjSLeY00rOGNWoyJh3D4iEMzZJRMpkOSTCZpUwmm0Zm2JhIMpRYcFyiSCAthsy1WZ5KFdtK5gWjy9YZIqlzlEwZJN90GY7LODmTyLY0NnBOdqwajHaoC852cuq4aXMcmDQl"),
            (FilterType::Gaussian, HashAlg::VertGradient, 8, "kJkDYQwsJIw"),
            (FilterType::Gaussian, HashAlg::VertGradient, 16, "dhKQg4rHxrWWhBSWSBRjFkQKZKGRqZIcazQoZqDm48Q"),
            (FilterType::Gaussian, HashAlg::VertGradient, 32, "WTYbIhlmThIpygzqmMlTSYnpGtEYoRmLnNBZhnAyG8YsZzjSucY0UjKmPUoiJR3TwnEwzYpZMpmu3DaZZW7Gmkwy3BhwWodaYG2qzLEpjUoV20rGBYPKxpmj6ZI8Tpl0z2QZDss4OYbKtjSScGZ2rBLMdqwLznZy6qhpcxyYNGU"),
            (FilterType::Gaussian, HashAlg::VertGradient, 64, "1jYyE8ympg7TLRMnm7qcqhKpJiflEp2LU69oWJ0ZxCvirG320JScpE+NzebSbM62lM+MuJtmUrrGpKKblJnXsY+G4pyeopCxgsjMZcKpi6eiQUZdySE1YmVmAmYTV+JksbMBpyW1vMSb05WVpbGmpqxqCDqLWbcmqTplhoVJJrJgHmUupVVeedMcNjWsakp7U9ksdagfJqtXzWRQJx8lColPPeWSpczYCU8tzGEHRxpaGqUMZ4lZ81poawazyEeTzV0bOaq8s1kNs6JthZtb+ZrGklEdp83Lu2RWmZInicO4WdPVRi6hElbbUXNkTmXJMp5oVidXTuFpssxmrHFp4VlnHY6maWSVmXE9CibJhhEyKW1ZNmvSEVquTGseYsaYay0ybNxa2pArOeIsVkZTcSaV5Y125vSsJY1FM2JSarIzDUfxXGIEu5OjmsnONpiqE2EeZJ5xjKxTSo9lyvaYkqNCrklH+syZ0saOSWbKlZQlT5YrZUERFzHZdFImSzGLufJkNqkZ24wTYzUkjbPKrNZx6SpHRprijFDhyuYGmliN0RztEi6PRKGglGYRa6VmhXKcPpobNkedbFtZOC1CxotkcnQ6HXPMKkvQ5VmaaGpMadjZGRcIrSYZm3k5ViytRs3YTOU0nbVUwZRFRyUHJTTHlVMZV5stJSelQ1HVOs4"),
        ];
        #[cfg(feature = "fast_resize_unstable")]
        let data = &[
            (FilterType::Lanczos3, HashAlg::Blockhash, 8, "3M0FbCH1SXo"),
            (FilterType::Lanczos3, HashAlg::Blockhash, 16, "Rmtwt5NMuVGZAHNooZz9P9IYk8R/t5BcqAnrPgTZ9pY"),
            (FilterType::Lanczos3, HashAlg::Blockhash, 32, "+JhF7rcRfDgFvjlWJBsrVxfiaTYPDVOQ509GlqOFS7dL0gFG+1tsEIc2YaxOjy2/UU7QWqXdmZaTiZ0GUt37CqRmUrOM8spCTlc2pHfVukj+Dy1JzuedykxTZBgEh/si2OhlgqulS0TKOvRP6twsFxJrVwUVaMD+gh2+J3zecM4"),
            (FilterType::Lanczos3, HashAlg::Blockhash, 64, "VVTBXKcKnFdaqibDc/X7xH8HZs/3W0SDJE0DI1BZyJ0zz7Xv4kg9sych6keNnokFcJyBuN+Xb4qkCs9lUYwcM7ZOh9SqOCkNj5GseqYmPL0dcOIyBD5Tz9+Fu4pXpoSVemr8PRp1r5dh2F/iaIC8CaeQE5hNAiKbSc8u5RNxPc+Pog2vrMRdczJ51nKRoJ7ajk3OV7IMBAl/9kZoT/UCYnHSFIsBNFKYJ0BeK1va9Fr/uLda0Dx3y+wXxUhpid0lSSZWsnC7fDvtG/g8DpOUwTaA42DDmyzBJ93129GgOsOPRaKodH9+kFlTbYqHgkEJRgFS387Wtphss/biT5/oFbCFZhwJF2tcPOx+mER0KZP2yRlHJfNMLBHiT/6s9G+i5yVdIjpPYk7oeJdmrOVwbjorItHOLWR5d6UTc4DGmpK8rLVB4phTkIen5wS9hWMm1XCxbOHJivCc0bt8Y6WuppjayHqU5tmHqjIXGxIfWlc1CS+wr+21SViRt8DtcQxaNGWcqmFtBuLFesLap2jOwe3qrqQnZCRheagz2YV0On5KgtUONP53YHz0XFUE430ZpATg4tE2IUPZ6lHX3mq/jYxWVZt+fjdADoduNDwboxKwy8Catfj0NhsnBDxK00fvBIX0UZT6WQxN54dK5Mc3baIuR3MkFn7asDvUx+M6nVA"),
            (FilterType::Lanczos3, HashAlg::Gradient, 8, "VCpSNptCLTU"),
            (FilterType::Lanczos3, HashAlg::Gradient, 16, "qjWoVkzWzBgmkpgmOJuomE3OjWscGt12toyknmQYIcM"),
            (FilterType::Lanczos3, HashAlg::Gradient, 32, "ncwlxolJLwtOWy0bm+y4MagpNTM1k1mRkmBKmlCQUpMtSYRTrQ0t2zaZOCWB5aSEri3XmvrWlNbMtEVpy9mardOR3rVVUnpwZVmbtatrTU5LhoxkmhLFmrFBkxui4jooJK1kyFRWaWNxVNTiNmZGnc01YUFRJ2qtxmpLNpJMViM"),
            (FilterType::Lanczos3, HashAlg::Gradient, 64, "qutqsiVe2hakxNqpmVLaNG3ZYyaTFXOJi6SFJVpNZpfUIIpVWUndlSRZQ2KlgouUm+VWa9RSqUpYpWGkTIWSnkmR6U5Ltq2lU1qDyLPSEqemnlSzphJL40S2zaJtM7NVlSWPmqxsq0q8yiFpNtWWzFptME20ZrPZMifHtphSjZIyWaZzJrQYMYxJZZpUzYay5CpjC5m0hG1TWWWmYpQGqRspo6XINlKcELqrlYliO21GRVMsWZaazZZFVSQp4lDNtpkhRTrJJqt2maoNZ6mc7kntTDZtUSr10+U6qTVSM7PQsjWqM1su3GK6c0xzUmFMybCJ0bpsM4yWqxFZoqWs5Epak9koSTPWkxI2a2aroElbZS4ppVJmvVKSKWssW0UztGKKRYxBc0ZnrEpypXUabcpUpXpjeaRNitGpWExrSpLEEq/UWmVRmcVSZJbh9HKTysYWJ3VNrubYSyuXMtWKIEdsJjnZskzlbYiySZpFW5O0dpHO7OKWTLzMytrkTGJD6whv1bAmUus6RXNNti6VUHYXVraiNJcttFtadk4buzIlmWljUiuFVksbZskyoyS9KZIxuVCyseUVtqxabxqByU3Ma2kzG6slUTOzRpLKqroeNXpuyjazuiaTVLZkk3EjklRZmN09SK2TNDJZW1ERP0yzmCi2Ww3bGc1qcWStycw"),
            (FilterType::Lanczos3, HashAlg::Mean, 8, "eQ9N/ff/aOs"),
            (FilterType::Lanczos3, HashAlg::Mean, 16, "V2fQdN+UuXNcEnMs4bLRu17cG8f/85r96InpPsE7Ls4"),
            (FilterType::Lanczos3, HashAlg::Mean, 32, "OLlvnDeTfB4dvjl+BKk5I1NzYSdrpnOzp8+ekOOjBx/Y0gjX+ntIAkU/cSwHj609HHqevoWNuZeTOc9LFvO/7awn0PX4lfpDj7IWvRbPmlD3n7lJtW+ev+5XMP9Nh/8qWfjBgMvtS0bHifDHydwMGRL5R0eGaMD9nQXa5PyZfOY"),
            (FilterType::Lanczos3, HashAlg::Mean, 64, "VdfVZ+0o+ndImRdPs9fyZZ4Xx2x2P0YTdk8PY9BZTLezx3WPchiYF08C/G+LH581JMiF2sUXO043CM/pmQ0iPJMng9kCKr0FvjQOO+eWPv7NPY0mBT5fjtttuk7ZJibVfk7/dM1Bgs8jsF/ybIEUyDfYZdotBKaTTf4efTF1nZfN8g3O7KA9dTlyyTPFiZ7yiV2KPvAlCGnsvs8MzvkfIHTSVIuPPdSYJ3B2L5N+ckmdgP5K07wziXkRrWjjgMEJSQPummC7fL/NAAQ/TmOczzPQ0GRIwy7BJ4vz0mFUPuOHReMeZ/L8+M/gR5nGwsNJF0cHp75J9vgoVj/yT+/o/bC3J75RU3eeZOx1igzQJ9v0y1lT7fNuIITlS9JI/w8w688QjTgf4kbc6Z9GTsXwTRA57HTm4KFZdwfTsdhjWJK5//+BV0RDsZvv3waHxWemNZ19bOrLjs6xl38+ZYG855zYTXKSpl3P3zM9kwAf+zdpzSeQ3f+9WWORFYDvXUbGFhFQvyFtAIjFmuSbrmi90cnurMSuYRRhL7b//Is2OnReJsPGNv6P9Jf0TJNE5n3wUgTj4/Emg4H7eDG13DI/gZgSVut3Mjt8jubn/bbbJiI4aoDYlGD2sB0mmCxAFvNv5QGyEZX5GYj856dS1kMxfLBuE2NgQzja8LvR5+k/n5w"),
            (FilterType::Lanczos3, HashAlg::Median, 8, "eQ9N/ff/aOs"),
            (FilterType::Lanczos3, HashAlg::Median, 16, "V2fQdN+UuXNcEnMs4bLRu17cG8f/85r96InpPsE7Ls4"),
            (FilterType::Lanczos3, HashAlg::Median, 32, "OLlvnDeTfB4dvjl+BKk5I1NzYSdrpnOzp8+ekOOjBx/Y0gjX+ntIAkU/cSwHj609HHqevoWNuZeTOc9LFvO/7awn0PX4lfpDj7IWvRbPmlD3n7lJtW+ev+5XMP9Nh/8qWfjBgMvtS0bHifDHydwMGRL5R0eGaMD9nQXa5PyZfOY"),
            (FilterType::Lanczos3, HashAlg::Median, 64, "VdfVZ+0o+ndImRdPs9fyZZ4Xx2x2P0YTdk8PY9BZTLezx3WPchiYF08C/G+LH581JMiF2sUXO043CM/pmQ0iPJMng9kCKr0FvjQOO+eWPv7NPY0mBT5fjtttuk7ZJibVfk7/dM1Bgs8jsF/ybIEUyDfYZdotBKaTTf4efTF1nZfN8g3O7KA9dTlyyTPFiZ7yiV2KPvAlCGnsvs8MzvkfIHTSVIuPPdSYJ3B2L5N+ckmdgP5K07wziXkRrWjjgMEJSQPummC7fL/NAAQ/TmOczzPQ0GRIwy7BJ4vz0mFUPuOHReMeZ/L8+M/gR5nGwsNJF0cHp75J9vgoVj/yT+/o/bC3J75RU3eeZOx1igzQJ9v0y1lT7fNuIITlS9JI/w8w688QjTgf4kbc6Z9GTsXwTRA57HTm4KFZdwfTsdhjWJK5//+BV0RDsZvv3waHxWemNZ19bOrLjs6xl38+ZYG855zYTXKSpl3P3zM9kwAf+zdpzSeQ3f+9WWORFYDvXUbGFhFQvyFtAIjFmuSbrmi90cnurMSuYRRhL7b//Is2OnReJsPGNv6P9Jf0TJNE5n3wUgTj4/Emg4H7eDG13DI/gZgSVut3Mjt8jubn/bbbJiI4aoDYlGD2sB0mmCxAFvNv5QGyEZX5GYj856dS1kMxfLBuE2NgQzja8LvR5+k/n5w"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 8, "EERhBSI"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 16, "VCpCVqqSai21lZobJimUbKSN"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 32, "qjWoVmzWzGgmmYwmGCqqmEnKTUeIKkwb1Wa0jKSaYA0hw1ZWlKGKxs611qS2hkiWq1RDCkRqJaKRrbY0azQ0ZqXk6tQ"),
            (FilterType::Lanczos3, HashAlg::DoubleGradient, 64, "ncxlxolZLwtOWy0Lm8yYMagtPTM1kVmRknFKmtC00pNl2cJTrQ2t03KdaSekpbSliSXXmqIW25LIlGVDyZ3FKcuZmrVX1mRwRFhaNSdLybRL7M5MSpKMJJISxZqxYZodpuI4qGStZMtRVmljcUTU5jZOBp3JNWFhUyZqrcZqS3aSTFYjaRebIlllThOpygyqmstTSYnpmlE5qViT3JFVrtoyW4dkNRrSrUZ0UrOWNWoypz3TqmlNzdLQMpFOSTIZpcwmilYm2IhYM5VadVziWmgtrcy1WZZKFdpK5kWj2pIZKqlTpEwZdN9kWYrLODnSyLZWMnJGdq2ajDapS8Z2cqooSVOembYl"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 8, "lZsTJQwsJLw"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 16, "VlaQwYrGzrWWpDSWSBSjTFRKJKORqZIdazQkJqTk6tQ"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 32, "aTebIhllThKpyizqmklTSYnpGtFcsVmrnNBZpnIyW8MsZzrSqcY0UjK2PWomLQ2VilEmzVpbMpGuTSYZ5cbGiloy3Fh0WodaYW0qzLEpnUoV20JGRZPahhkiqZK8Xol032RZLss4OZLKlnQScGZWrJqMZqlLzjZyqihJU56YtiU"),
            (FilterType::Lanczos3, HashAlg::VertGradient, 64, "1DYyEcympk7SLRMm27rcqjapJi2lEj2LW69qXJ0ZxKrirGmy1JScZWwNzebSbM62lM+MOJtmWrrKpCKrpJn3sY2G8oye6oSxgsjN5Vasm+eqSVJdySE1amVmImYTU8JksbMJtiWVqNSb0pWVpaG2pi5qCzqLWbcmqTplhhVJprJksuVupVReatGUNjXsaEp7W9kutag1IqtX3WRKJxWlKolOOaWSpMxZCW8tTGElRxpKGrVMZ5lZ8VpuawaySk+TzV2ZOaq1s1gVs6JtlZpb9ZpGklE5r8XKq+RWmZImiUm4WVPVRi6pElZbWaN0zmXJMpZoVidXTuWtuswkiTBp4XFnGa6maWWViVU1CibZkhUyKWlZMmmSE1iuTGtaQobOa2UyrNxYWpIrmbIsVldTcyaV5Y1mZqasNI1bM+pSSrIzTUXxHKsUm5Oi2snONomqE3EeVJ5ZrKxSaqtlyraYkqtTrklH+s7J0sauSSZKlZQlT5ZrRVGRF3XZdFImSzWbu/JmNilZ26xaIz0lhaPKrNY16SpVQ5LijFrhyuYWmliJ0zzpkCyvVKWklGYRS6VmhfLevpqbqk2dbEtZKCVKxotkcnQ6HXPMKmvVpFma6GpFaVjZGBZKrSIZE3kxVmWtVm3YTMU0mbVV6ZVER60XpTWNlVM5V6stJVOtQVXVetY"),
            (FilterType::Triangle, HashAlg::Blockhash, 8, "3M0FbCH1SXo"),
            (FilterType::Triangle, HashAlg::Blockhash, 16, "Rmtwt5NMuVGZAHNooZz9P9IYk8R/t5BcqAnrPgTZ9pY"),
            (FilterType::Triangle, HashAlg::Blockhash, 32, "+JhF7rcRfDgFvjlWJBsrVxfiaTYPDVOQ509GlqOFS7dL0gFG+1tsEIc2YaxOjy2/UU7QWqXdmZaTiZ0GUt37CqRmUrOM8spCTlc2pHfVukj+Dy1JzuedykxTZBgEh/si2OhlgqulS0TKOvRP6twsFxJrVwUVaMD+gh2+J3zecM4"),
            (FilterType::Triangle, HashAlg::Blockhash, 64, "VVTBXKcKnFdaqibDc/X7xH8HZs/3W0SDJE0DI1BZyJ0zz7Xv4kg9sych6keNnokFcJyBuN+Xb4qkCs9lUYwcM7ZOh9SqOCkNj5GseqYmPL0dcOIyBD5Tz9+Fu4pXpoSVemr8PRp1r5dh2F/iaIC8CaeQE5hNAiKbSc8u5RNxPc+Pog2vrMRdczJ51nKRoJ7ajk3OV7IMBAl/9kZoT/UCYnHSFIsBNFKYJ0BeK1va9Fr/uLda0Dx3y+wXxUhpid0lSSZWsnC7fDvtG/g8DpOUwTaA42DDmyzBJ93129GgOsOPRaKodH9+kFlTbYqHgkEJRgFS387Wtphss/biT5/oFbCFZhwJF2tcPOx+mER0KZP2yRlHJfNMLBHiT/6s9G+i5yVdIjpPYk7oeJdmrOVwbjorItHOLWR5d6UTc4DGmpK8rLVB4phTkIen5wS9hWMm1XCxbOHJivCc0bt8Y6WuppjayHqU5tmHqjIXGxIfWlc1CS+wr+21SViRt8DtcQxaNGWcqmFtBuLFesLap2jOwe3qrqQnZCRheagz2YV0On5KgtUONP53YHz0XFUE430ZpATg4tE2IUPZ6lHX3mq/jYxWVZt+fjdADoduNDwboxKwy8Catfj0NhsnBDxK00fvBIX0UZT6WQxN54dK5Mc3baIuR3MkFn7asDvUx+M6nVA"),
            (FilterType::Triangle, HashAlg::Gradient, 8, "VEJCRpJiLDQ"),
            (FilterType::Triangle, HashAlg::Gradient, 16, "qjWoVkxWzFgmkhgmOBuoik3PjWsYCtV29IykjmQcIcM"),
            (FilterType::Triangle, HashAlg::Gradient, 32, "jMwlRolJLQ9MSz0Lm+yYM7gpPbM001iRsnBOknCQUlMtWcRTLR1t0zCJOCWhZaSEjC3XmurWlNbMtMXpydmYrdOR3rVHUlZwZVqbtatKzU5LxoxMihOFiqEhGRum4jqoJK1kyXRWaWMxVNXiNmZGnck1NUFRJkqlx2JLcpZOFjM"),
            (FilterType::Triangle, HashAlg::Gradient, 64, "KuFisCVe2jSkzMopmVZaNO3RYySbFXOJi6TBIVpNZpWdIIpRWUnNlSBZw2LhgouUi+VWa9TWiUJYpWEkRIWSnkmZ6U5blq2lQ1rDirPSEqcmnlW7pBML48S2jaJpMaNRlaULqqxtq0qc2yFtJtWWzVptMEWUcrPIMmeHtpxSjdRySadzppAcMUxJ5ZpUxIay5Cpjix2Uxm1TWeGC45QWqRspo6XIFtKcEKqLlckiOk1GRVOoSZaazRZBVTQp4lDNtplhRTrJpqt0iKsNZ7mW6lntfBdtUSrx0eU4qTFSMzPQsjWKM1su3GK6c01z0mFMybDL0a5tM4yW6pFVoiWsxE5ak8ksSTPXm1Iya2apoMl7dGwppVhmvVKSIGusWUUTNGCKQYxBc0Z3vswypWUabcpUpXpjcaRNitGpWkxzTpvGUqrUWmVxmcRSdJbh9XKTysa+JnVNrubYyiqTstGuJEVuJhmbskzlZYiySZpFWxO0ZhHOrOKmTLzMStjkRmJDqwgv1bAmcus6VXJNtSyVeDYVVramNJcpNFsY9mYZmzYlm3ljUiuFVksbZkkyozS9KZIxuVCykeEdtqhabxqBiUfMK2kzG6sFUTMzRpLKqroeNXouSjazqiYVVZZk01EiMlRZmF09SayTcDJZ21URP1yzmCi2ew3LGc1qcnSrzcw"),
            (FilterType::Triangle, HashAlg::Mean, 8, "fQ9N/ff/eus"),
            (FilterType::Triangle, HashAlg::Mean, 16, "V2fQdN/V+RNcEnMs4f7Rv17eG+b/+7v96InpPsE5Ps4"),
            (FilterType::Triangle, HashAlg::Mean, 32, "uLlPHDeTfh4bvjl+BIs5J2dzYadrp3Gz58+akOOzBx/a2gjH+ntoAkU/cSwHj609GnrevofdmZeDec/PFvO//+y30OX8k/pDnwIWvBbPmtD/n5lJ9X+e/+9XMP/Nx/+6WPjBgMvuS0bHmfDHzNxMFRL5RweGQMDtng3Y4PydfOI"),
            (FilterType::Triangle, HashAlg::Mean, 64, "xcfFZ60I+HdIm4dHu9/4ZR4Xx0R/P2YD9k8HY9BZ/J+yx3RP+hicF08A/G+LH54VFMiB2MUXO043CM/piR0iPJMnh/kCPj8FnjQOO6eWPv7NPa0mBz5fhvt9Os7ZNwbXfk7/dM1hgM8zsH/ybAEUwD/YZcotBKaHRf4e/TFxv9dN8gznrYA9N7nzyTPBAZ7yiV2OPvANCEnsvs8Mxv0fIHRSVouLPNSZJzB2DQN+cgmNAP9I07xziXkD7WjigNMBSQPu2nD7/L/pAAQ/TmOczzOQ8GRIwy/Dp47z0mFQPuMHBeMfZ/J+8I/AR5vHwsdJF0cHp7/B9rgsFj/yT//p/bCnJ75RU3PeZOx1ikD0L9vk6FlT7fNuMITlS9bo/w8w78cQjjhP4k744Z9HTkXwTTE5/nTu4aFbNwPTsdhnGJK5//+Bx8RDsJ/v/wSHxWcmlZ39furJnu7xn39+JMH855rYfXqypt/v3zs/kwAf/zd5TSeQ3d+9WWORH4Dvf+ZGBhFQvyFtAMDFiuSbpmid0cnq6NSuYRRgT7L/3I52OnBetuPGJv4/9Bb0TQNA7n3wQnDj4/Emg4P7eHGx/DY/AYgQRvt3Mj88jubn/babLiI4aoDYFGD2qB0imChAlvPu5QGyEJf7Gej+56dA0EMBfLBuU2NgQzja8L/T5/k/n9w"),
            (FilterType::Triangle, HashAlg::Median, 8, "fQ9N/ff/eus"),
            (FilterType::Triangle, HashAlg::Median, 16, "V2fQdN/V+RNcEnMs4f7Rv17eG+b/+7v96InpPsE5Ps4"),
            (FilterType::Triangle, HashAlg::Median, 32, "uLlPHDeTfh4bvjl+BIs5J2dzYadrp3Gz58+akOOzBx/a2gjH+ntoAkU/cSwHj609GnrevofdmZeDec/PFvO//+y30OX8k/pDnwIWvBbPmtD/n5lJ9X+e/+9XMP/Nx/+6WPjBgMvuS0bHmfDHzNxMFRL5RweGQMDtng3Y4PydfOI"),
            (FilterType::Triangle, HashAlg::Median, 64, "xcfFZ60I+HdIm4dHu9/4ZR4Xx0R/P2YD9k8HY9BZ/J+yx3RP+hicF08A/G+LH54VFMiB2MUXO043CM/piR0iPJMnh/kCPj8FnjQOO6eWPv7NPa0mBz5fhvt9Os7ZNwbXfk7/dM1hgM8zsH/ybAEUwD/YZcotBKaHRf4e/TFxv9dN8gznrYA9N7nzyTPBAZ7yiV2OPvANCEnsvs8Mxv0fIHRSVouLPNSZJzB2DQN+cgmNAP9I07xziXkD7WjigNMBSQPu2nD7/L/pAAQ/TmOczzOQ8GRIwy/Dp47z0mFQPuMHBeMfZ/J+8I/AR5vHwsdJF0cHp7/B9rgsFj/yT//p/bCnJ75RU3PeZOx1ikD0L9vk6FlT7fNuMITlS9bo/w8w78cQjjhP4k744Z9HTkXwTTE5/nTu4aFbNwPTsdhnGJK5//+Bx8RDsJ/v/wSHxWcmlZ39furJnu7xn39+JMH855rYfXqypt/v3zs/kwAf/zd5TSeQ3d+9WWORH4Dvf+ZGBhFQvyFtAMDFiuSbpmid0cnq6NSuYRRgT7L/3I52OnBetuPGJv4/9Bb0TQNA7n3wQnDj4/Emg4P7eHGx/DY/AYgQRvt3Mj88jubn/babLiI4aoDYFGD2qB0imChAlvPu5QGyEJf7Gej+56dA0EMBfLBuU2NgQzja8L/T5/k/n9w"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 8, "EURSBSI"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 16, "VApCdiqSaiy0lZoLIimMbCSM"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 32, "qjWoVmxWzFgmmCwmWCuomknKTU+AKlwa1SY0jKSaZJwhw3YTkuOKw861xqSUhliWq1RDCmRpdKKRrfIkazQ0ZqHk6sQ"),
            (FilterType::Triangle, HashAlg::DoubleGradient, 64, "jcwlTolJLw9MSykLm8yYM5opOTM00ViRknFOmPC00pFl2cZTLR0t0zONaSWkwbiliSWWnqIW2ZLIlFXDyJ3FrcuRmrRTkuxwRVhaNWVLyaRL6s5MSpaMDKoRxZqhIZgZpuI4qGStZMl1VmhjMUTU4jZmRpHJNSFBUSJaJcdiS3KSTlYzSRVbIhlkThIpygzqmslTaYnpGtk5qRm73NFVpnQyW6dkNxjSrcY0UrOGNWoyJx3TqmkMzZpxMpEuSTAZpUymilZi2JhIMpVQcVSiSjAtl8y1UZ5KFZtK5gWjy9KZIqlytEyZdN9kWQrLODnXyLZ0NnJOdqxSjHaoD842YuqoaXOemDQl"),
            (FilterType::Triangle, HashAlg::VertGradient, 8, "kJsTIQwkJIw"),
            (FilterType::Triangle, HashAlg::VertGradient, 16, "dhKww4rHxrWWhDSWSRSjFEZKZKGRqZIZayQsZqHk4sQ"),
            (FilterType::Triangle, HashAlg::VertGradient, 32, "STVbIhlkThIp2gzqmslTSYnpGtEcoVmr3NBdpmAyG8IsZxjSq8Y0UjKmPUomLR3T6nEwzYpJMpmu3SYZ5U7Emk4ynFh0WpdaYW2qzLUpnUo120pGBYPKhpmiqdK4Xpl0z2RZLss4OZfKtnQTcG52rFKMdKgLzjZq6ipJc56YNGU"),
            (FilterType::Triangle, HashAlg::VertGradient, 64, "1DYyGcympk7TLRMnm7qcqhKpJi/lEj2LW6lqXJ0Z1C3irGm20JScpE2NzebSbs62lM+MuJtmWrrCpCKrhJHXsYWG8pyaqpCxisrMZcKpg6eqSUJdySE1amVmAmYTV+JksbMBt6W1vMTb0pWVpaGmJqxqCCqLXbcmqTplhgVJprJgmmUupVVeadEUNjXkakp7U9kstagWJqtVzXRSpRUlKolPKeUSrczYCU8tTGEnRxpKGqUMZ5lR8VpoawazyEWTTV0bOaqts1kNs6pthZpb9ZpGklU5p83Lu2RWmZImiUG4WdPVRi6pElbaUSNkTmXJspZoVidXTuHpsswmqHBp4VlnHY6maWSVyVU9CibLxhEyK2lJdmvSEViuTGtOQoaKSy2yptxYWpIrGaIsVkdTcS6V5Y1m5vSsJY1VM2pSarIzDUfxXGoUm5Oi2MnONpiqU2EfZZ5RjKxSaottyvaYkqtCrklH+syZ0saOaCZKlZQlT5YrRVERF3XZdFpmSzGLu/JmNikZ26xTYzUljaPKrNZxqSpXQpLijFDhyuYGmliN0TztkC6rVKWglGYRa7VmhfKctpobpk+ZbFtZOC1KxstkMnQ6PXOMKkvR5Vma6GpMaVhZGRKIrWYJm3k5ViStVs3YTMU0nbVVwZVFV6WHJTXFFVMZV4stJSet01XVGtI"),
            (FilterType::Nearest, HashAlg::Blockhash, 8, "3M0FbCH1SXo"),
            (FilterType::Nearest, HashAlg::Blockhash, 16, "Rmtwt5NMuVGZAHNooZz9P9IYk8R/t5BcqAnrPgTZ9pY"),
            (FilterType::Nearest, HashAlg::Blockhash, 32, "+JhF7rcRfDgFvjlWJBsrVxfiaTYPDVOQ509GlqOFS7dL0gFG+1tsEIc2YaxOjy2/UU7QWqXdmZaTiZ0GUt37CqRmUrOM8spCTlc2pHfVukj+Dy1JzuedykxTZBgEh/si2OhlgqulS0TKOvRP6twsFxJrVwUVaMD+gh2+J3zecM4"),
            (FilterType::Nearest, HashAlg::Blockhash, 64, "VVTBXKcKnFdaqibDc/X7xH8HZs/3W0SDJE0DI1BZyJ0zz7Xv4kg9sych6keNnokFcJyBuN+Xb4qkCs9lUYwcM7ZOh9SqOCkNj5GseqYmPL0dcOIyBD5Tz9+Fu4pXpoSVemr8PRp1r5dh2F/iaIC8CaeQE5hNAiKbSc8u5RNxPc+Pog2vrMRdczJ51nKRoJ7ajk3OV7IMBAl/9kZoT/UCYnHSFIsBNFKYJ0BeK1va9Fr/uLda0Dx3y+wXxUhpid0lSSZWsnC7fDvtG/g8DpOUwTaA42DDmyzBJ93129GgOsOPRaKodH9+kFlTbYqHgkEJRgFS387Wtphss/biT5/oFbCFZhwJF2tcPOx+mER0KZP2yRlHJfNMLBHiT/6s9G+i5yVdIjpPYk7oeJdmrOVwbjorItHOLWR5d6UTc4DGmpK8rLVB4phTkIen5wS9hWMm1XCxbOHJivCc0bt8Y6WuppjayHqU5tmHqjIXGxIfWlc1CS+wr+21SViRt8DtcQxaNGWcqmFtBuLFesLap2jOwe3qrqQnZCRheagz2YV0On5KgtUONP53YHz0XFUE430ZpATg4tE2IUPZ6lHX3mq/jYxWVZt+fjdADoduNDwboxKwy8Catfj0NhsnBDxK00fvBIX0UZT6WQxN54dK5Mc3baIuR3MkFn7asDvUx+M6nVA"),
            (FilterType::Nearest, HashAlg::Gradient, 8, "x1qyqjKLtiY"),
            (FilterType::Nearest, HashAlg::Gradient, 16, "rcZajVpURmrulRqyrZyiMqmopirWLlG3l9TjpmZSssk"),
            (FilterType::Nearest, HashAlg::Gradient, 32, "olSkMik5KjvUtCZtNFm4pzQ5E9ccnlWpvWVlqlNLIXerZiazpZWK0iS2S2k4WaY0RQWFa5PSVltKicxUZTWZL6XyqOYrs3ZqqullladSVbtUxgpm6TXmoplkbVUs2opUG2WObDBrJA51t7WmdlSvqtiUqmI3KxqppfTYcJFqN5E"),
            (FilterType::Nearest, HashAlg::Gradient, 64, "qutqgqkUVUnU9SodaWqad6U2RDWPM3OpmtSptdpR2uGMNlsorVgTpzoLtyyWG6fVzXVW7jRFzRlN1SSVp6q4ijVlaSlRtalkRVKNz6CqUClueHRNbbqJWllaqWFVIrO0o0NFnWUUyWYam3J9nbYWzThRGVRN21NrMyqFhNVUk6z2rOVIJWPNlStrzqNTVa2bpjYKDpWSXA0KZaumodTWTbYwIs3WUpTVkpqGhFdqKWY2Hlsu1ZkRS3RburQyRWVVps2onFNrzpYllmxTVyWU5knValKl2TqrjRV76fBWnFNIVaXWG0tKlVZyWtnV0q4dSebJkYxJs5vUYolVyjVZpZOTUW8bK1ZVGktlpKW1rso1tUY5dahHtWYK4y6tWEsbKVvKWawjVmWdqmazqfESVUq3q6xlFVVVqxppWj3TTdtVV7Fb6zGcKpKlZpUVt1JTMhqqKlJPplaUJkqxsoSrbFWupDKbtpBljlhLJRqraptaYzFMqdK1X6jmTlFKRsUmiW5sVVUpIm0pUrYNpcpF2K21VCpydEU7tFmtkp0Ui1IkyktrUkalVsgWolk2txRaEpJROdpy23mbnKaK2U4riWrWKm0ZWStKmXJRShYp7qpVp6m0WTfpqjU1Vta2K+ksl6pLaS41TcqNSeo1Lk2aRZS7tLSyaiarK6Yyqjivlto"),
            (FilterType::Nearest, HashAlg::Mean, 8, "TZwCstVF88A"),
            (FilterType::Nearest, HashAlg::Mean, 16, "0YnDJG8XhVtEF+P0LfAlLQ1uQwadWHDdKsngzt5xHq8"),
            (FilterType::Nearest, HashAlg::Mean, 32, "PJ7P/qvHYmuXgeMDPrsnKY4uzcRrLPkEpr3sPJJ4B2pghWOk1zFakXxiqHi4rRtwejM2zV3yI9ekIvhyU56S5VZWgeTR9PbLvuvftWLDt//YgVLPdWm9LsY2F0PVxvdQWLooxAr1Djf4KlbhcsQcBWLpLEFWXlalyMXwY3AZ7rg"),
            (FilterType::Nearest, HashAlg::Mean, 64, "TAzV/ygcdPtL78mFmW8aZsYIf+FRJXchkEkHW2qG2AEorjQXHlhzrtsVCm9vB+TfAPSY6dfdmXu3msAHi3Twnxjtld1KJK9l3h++8UIIB+vJumfu6S8Z8ti0fvP4hKYwUAMtG3Axn8Zfk1J36LMU3ZZQtdoNeh9OZW7DrDkT8ixOiy3az5cNNZjZpTccy7izxg8Ob65teAmvPMLyX72Gz5d+caOTZZXd4qFWiTjlOEy3t795V2wRCi1JjHHkgUFVD28K2GTTiLdWv9Q2WseUzjyaZhNgkiAvjwzDOrHYEFOugdR99NUaNZbC1++y5qg516GDp7EpI5tU8A+aAWn7/ZF3YnTI0fx/c0rdAkU8r4otQmBh7/AH4WOBW0mo7Q4yM0+KK7PMdG0BQLeXeGwQfXPLsRzDP10FNIvdw0nHXNIU/zOXRwYxKI+I69G9pUZ2FYWpbzILrlb0ruvYZuG/7YjanHtTN7Ntg4gboYIVapP+LWsa+S83UKOB1/fZdsDnwWFC9uEPJszESo7Kp3gF2Oj35rTqLER4N6yWny5Em1aqgOUAIEsHUCTUCXOJYxDYUzJiy8a4mvHnJrEUxF5vD44FzuP35Woa6y6/9bow7Ko8y6KNJQHPqgAyHD1Ii/nvgSUQldd7XMq466hkjLoSXegVKTgvIg4uPhPXkP7Otpg"),
            (FilterType::Nearest, HashAlg::Median, 8, "TZ4DstXl88A"),
            (FilterType::Nearest, HashAlg::Median, 16, "0YnDJG+XhV9EF+P0LfAlLQ1uQwadWHDdKsngzt5xHq8"),
            (FilterType::Nearest, HashAlg::Median, 32, "PJ7P7qvHYmuXgeMDPrsnKY4uzcRrLPkEpp3sPJJYB2pghWOk1zFakVxiqHi4rRtwcjM2zV3yI9ekIvhyU56S5VZWgeTR9NbLvuvflWLDt//YgVLPdWm9LsY2F0PUxPdQSLooRArlDjf4KlbhcsQcBWLpLEFWTlalyMXwY3AZ7rg"),
            (FilterType::Nearest, HashAlg::Median, 64, "TAyV/ygcdPtL78mFmW8aZsYIf+FRJXchkEkHW2qG2AEorjQXHlhzrtsVCm9vB+TfAPSY6dfdmXu3msAHi3Twnxjtld1KJK9F3h++8UIIB+vJuifu6S8Z8ti0fvPwhKYwUAEtG3Axn8Zfk1J36LMU3ZZQtdoNeh9OZW7DrDkT8ixOiy3az5cNNZjZpTccy7izxg4Ob65teAmPPMLyX72Gz5c+caOTZZXd4qFWiTjlOEy3t795V2wRCi1JjHHkgUFVD28K2GTTiLdWv9Q2WseUzjyaZhNgkiAvjwzDOrHYEFOugdR99NUaNZbC1++y5qg516GDh7EhI5tU8A+aAWn7/ZF3YnTI0fx/c0rdAkU8r4otQmBh7/AH4WOBW0mo7Q4yM0+KK7PMdG0BQLaXeGwQfXPLsRzDP10FFIvdwwnHXNIU/zOXRwYxKI+I69G9pUZ2FYWpazILrlb0ruvYZuG/7YjamHtTN7Ntg4gboYIVapP+LWsa+Ss3UKOB1/fZdsDnwWFC9uEPJszESo7Kp3gF2Oj35rTqLER4N6yWny5Em1aqAOUAIEsHUCTUCXOJYxDYUzJCy8a4mvHnJrEUxF5vC44FzuP35Woayy6/9bow7Ko8y6KNJQDPqgAyHD1Ii/jvgSUQldd7XMq466hkjLoSXWgVKTguIgwuPhPXkP7Otpg"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 8, "NN/slHM"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 16, "zxSRU5ayMSzWayOuqmqLTXWa"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 32, "S6R4aK1Ku02qmlf2bmWq7avb5UVaDdd6kpXUBazSalmyyWmrVJNMyjpNKmoQSslsNVPLBKNI27Y6G2bKp0YpRnXFTK4"),
            (FilterType::Nearest, HashAlg::DoubleGradient, 64, "2iqXcRJVqdSTijPRUtTso1Iyeaq1MnKfr3Bs12qgepdjjcq2JVVFu/l5JS2nSlJUMmVlFKbO6tbJWteaKsrFLVLLsopWsmo6VFqqLiVVUqobrWhrtbnmSpPdxKX2KKVFqkJZOWV11o6pzFKTsqzVck0qV6XYlKpiNysaqaX02HCRajeRRL3SWtoii0vNzqi0iZVlrakxhcoL6Sm7U02uYmxoSw0iN/lSIuk1ESqx0agtycVUytTuqoWlJsueo0lrtaMyc7XSKlOpa6pEWpaCmrJpFI2hWrlEWKaiYlmqS7lLr6Voo5YSVSJ7JaomuUWXd0pjrTtrVql7HcWizGGtbrpWSVa7kDSh"),
            (FilterType::Nearest, HashAlg::VertGradient, 8, "VHFeIS1RbLI"),
            (FilterType::Nearest, HashAlg::VertGradient, 16, "WwtSa6vWNbOT6l5V2hNtOdpSq6TZ6NXKSk2k7qtaTK4"),
            (FilterType::Nearest, HashAlg::VertGradient, 32, "ySq1rDJBohrFsdMra5UmRa1K0aG0Io2VlNwmM0POsmWpUslStBWPJFGrLSHVoieVyJYytUiqVd23DJUs06plReaq3knV1FShW1uzmYro2dimGGlijVwj2UonLusvrShiMqX6Gc2qmEXFJJQ5KmMvxnsdxaLMYa1uulZJVruQNKE"),
            (FilterType::Nearest, HashAlg::VertGradient, 64, "pMYyGSlxzZrrjm3aSppMwaDqZFatUKvVdGmpS1VZVTH2blSy1ZVOtk3XjFWysq52DdetkkmVUVIurqIrZ6SXENPSSUk2qqV1ydguViVp1qpyktqUVW6lWVFVi3KWdYuis6VpMqWtJG2ZmqkVprnFmmTOqniKdbe0TbbkJi1tLbR1umVsplWzyvOUFS2l2mqhU1XVWbJ3BqlFhaZWq02iK27rLWUzaY7aTVGyWet21jLKJifFJJNUlVLJ6kavzZq1a2U7taj1aEl5sRqVVZprVathgtWarKwSulUj3U6aSma8TFtVqq6k2krGVnvJTDVptS6UlB2raatUuay6qZ1rldpQiqaG7TFVjJZlyyvpalsVzEaNWCVLddmq1nIqaiUqqy2lUpmpqNGjzaguLUkV8U20be60VfXMJX2VtbjIdrhKTNUgW041TZpsK0aV4pgoW7WJTE0u5dSu0snKSSrTmnJVNaWHDuKaU1W5JqtUJzOsqh6pbdqqmVVYYppqyXatqbY2xk0124RWIi9FpSpJtCZNqrK2WJVSyGK1+ratzkouy1QlGaUommXOUCXVTLGpmWpqolOxplxLZEZWLiVqx0ZEttJqam6qSsGa0lzarlVcNU1cslmJKkU9uVRVEqV1Xd2aR1mdKrk1p7bVWS9VT/ROUpVXZjmpE1W9il12a9s"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 8, "3M0FbCH1SXo"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 16, "Rmtwt5NMuVGZAHNooZz9P9IYk8R/t5BcqAnrPgTZ9pY"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 32, "+JhF7rcRfDgFvjlWJBsrVxfiaTYPDVOQ509GlqOFS7dL0gFG+1tsEIc2YaxOjy2/UU7QWqXdmZaTiZ0GUt37CqRmUrOM8spCTlc2pHfVukj+Dy1JzuedykxTZBgEh/si2OhlgqulS0TKOvRP6twsFxJrVwUVaMD+gh2+J3zecM4"),
            (FilterType::CatmullRom, HashAlg::Blockhash, 64, "VVTBXKcKnFdaqibDc/X7xH8HZs/3W0SDJE0DI1BZyJ0zz7Xv4kg9sych6keNnokFcJyBuN+Xb4qkCs9lUYwcM7ZOh9SqOCkNj5GseqYmPL0dcOIyBD5Tz9+Fu4pXpoSVemr8PRp1r5dh2F/iaIC8CaeQE5hNAiKbSc8u5RNxPc+Pog2vrMRdczJ51nKRoJ7ajk3OV7IMBAl/9kZoT/UCYnHSFIsBNFKYJ0BeK1va9Fr/uLda0Dx3y+wXxUhpid0lSSZWsnC7fDvtG/g8DpOUwTaA42DDmyzBJ93129GgOsOPRaKodH9+kFlTbYqHgkEJRgFS387Wtphss/biT5/oFbCFZhwJF2tcPOx+mER0KZP2yRlHJfNMLBHiT/6s9G+i5yVdIjpPYk7oeJdmrOVwbjorItHOLWR5d6UTc4DGmpK8rLVB4phTkIen5wS9hWMm1XCxbOHJivCc0bt8Y6WuppjayHqU5tmHqjIXGxIfWlc1CS+wr+21SViRt8DtcQxaNGWcqmFtBuLFesLap2jOwe3qrqQnZCRheagz2YV0On5KgtUONP53YHz0XFUE430ZpATg4tE2IUPZ6lHX3mq/jYxWVZt+fjdADoduNDwboxKwy8Catfj0NhsnBDxK00fvBIX0UZT6WQxN54dK5Mc3baIuR3MkFn7asDvUx+M6nVA"),
            (FilterType::CatmullRom, HashAlg::Gradient, 8, "VkpCRppKLDQ"),
            (FilterType::CatmullRom, HashAlg::Gradient, 16, "qjWoVkxWzBgmk5gmOJuqmE3OjWsaCs12toykjmAYIcM"),
            (FilterType::CatmullRom, HashAlg::Gradient, 32, "jcwlRolJLwtMWy0Lm+y4MagpPTM001iRkmBOmlCQUpMtScRTLR0t0zKJeCWhZaSErC3XmurWlNbMtEVpydmardOR3rVFUlRwZVubtatrzV5LhoxMmhLFurExWxum4rqqLK1kyVRWaWdxVNXiNmZWnc01IUFZJmKtxyJLdpJOViM"),
            (FilterType::CatmullRom, HashAlg::Gradient, 64, "KuFqsiVe2hakzNopmVZaNO3ZYyabFXOJi6SBJVpNZpXdIIpVWUnNlSBZQ2KlgouUi+VWa9RWiUpYpXEkTIWSnkmb6U5Ltq2lQ1rDiLPSEqemnlS7phJL48S2jaJpM7NVlSULqqxtq0q8yiFpJtWWzFptME2UZrPIMmfHtpxajdZyWadzJrAcMUxJZZpUzYay5Cpjix20xG1TWaGmYpQGqRspoqXIFlKcELqLlcliOm1GRVMoWZaazRZFVTQp4lDNtplhRTrJJqt2mKsNZ7mU6lmtXDZtUSr10+U6qTVSMzPSsjWKM1su3CK6c01z0mFMybDJ0a5tM4yW6hFVoiWshE5ak8moSRPXkxIya2aroMlbZGwppVlmvVKSIWusWUUTtGCKQ4xBc0Z3rswypXUabcpUpXpjeaRdqtGpWkxzSprGUqvUWmVRmcVSdJbh9HKTisaWJ3VNrubYyiuTstWOIEVsJhnbskzlZYiySZpFWxO0dhHO7OKmTLzMytrkTGJD6wgv1bUmUus6VXJNti6VWHY1VraiNJcttFtY9kYZmzYlmXljUiuFVksbZkkyozS9KZIxuVCykcEVtqhabxqBiUXMa2kzG6sFUTMzRpLKqroeNWouSjazuiaRVJZkk1EjElRZmN08SO2TcDJZ21URP1yzmCi2Ww/bGc1qc2Srzcw"),
            (FilterType::CatmullRom, HashAlg::Mean, 8, "eQ9N/ff/eOs"),
            (FilterType::CatmullRom, HashAlg::Mean, 16, "V2fQdN+UuVNcEnMs4f7Rux7cG8f/85v96InpPsE5Ls4"),
            (FilterType::CatmullRom, HashAlg::Mean, 32, "uLlPnDeTfB4dvjl+BKs5J2dzYSdrpnOzp8+akOOjBx/Y2gjX+ntIAkU/cSwHj609GHrevoXNmZeDec/LFvO/7+w30PX4k/pDj7IWvBbPmlD/n5lJtW+ev+5XMP9Nh/+qGfjBgMvuS0bHifDHzdwMFRL5RweGYMDtng3Y4PydfOY"),
            (FilterType::CatmullRom, HashAlg::Mean, 64, "1dfVZ60o+HdImxdHu9/6ZR4Xx2R3P2YDdk8PY9BZTL2yx3TP8hicF08C/G+LH58VFMiB2sUXO043CM/piQ0iPJOnh9kCKq0FvjQOO+eWPv7NPe0mBT5fjtt9Ok7ZNybVfk7/dM1Bgs8jsH/ybIEUwDfYYcptRKaXTf4efTF1v9fN8gzn7KA9NznzyTPFgZ7yiV2KPvANCEnsvs8MzvkfIHTSVIuLPNSZJzB2LRN+ckmdgP5I07wziWkDrWjjgNMBSQfumnC7/L/JAAQ/TmOUzzPQ4GRIwy7Bp4rz0nFUPuOHBeMfZ/J88M/AR5nHwsdJF0cHp7/J9rgsFj/yT//o/bCnJr5RU3PeZOx1ikzUL9vkytlT7fNuMITlS9bo/w8y68cQjzhf4kbo6Z9HTsXwTTA5/nTu4aFZdwPTsdhnWJK5//+B18RDsZvv3waHxWemFZ19fOrLju7xn39+ZYG855jYbWqSpl3P3zc9kwAf+zd5TSeQ3f+9WWORFYDPf8ZGFhFQvyFtAMjFmuSbpmid0cnq6MSuYQRhT7L/3Is2OnReNsPGJv4f9Bb0TANE5n3wQkTj4/Emg4P7fDG13Do/gZgQVut3cjs8jubn/babJiI4aoDYFGD2qB0mmCxIFvPu5QGyEJf5GYj+56dQ1EMBfLBuU2NgQzja8LvT5+k7n5w"),
            (FilterType::CatmullRom, HashAlg::Median, 8, "eQ9N/ff/eOs"),
            (FilterType::CatmullRom, HashAlg::Median, 16, "V2fQdN+UuVNcEnMs4f7Rux7cG8f/85v96InpPsE5Ls4"),
            (FilterType::CatmullRom, HashAlg::Median, 32, "uLlPnDeTfB4dvjl+BKs5J2dzYSdrpnOzp8+akOOjBx/Y2gjX+ntIAkU/cSwHj609GHrevoXNmZeDec/LFvO/7+w30PX4k/pDj7IWvBbPmlD/n5lJtW+ev+5XMP9Nh/+qGfjBgMvuS0bHifDHzdwMFRL5RweGYMDtng3Y4PydfOY"),
            (FilterType::CatmullRom, HashAlg::Median, 64, "1dfVZ60o+HdImxdHu9/6ZR4Xx2R3P2YDdk8PY9BZTL2yx3TP8hicF08C/G+LH58VFMiB2sUXO043CM/piQ0iPJOnh9kCKq0FvjQOO+eWPv7NPe0mBT5fjtt9Ok7ZNybVfk7/dM1Bgs8jsH/ybIEUwDfYYcptRKaXTf4efTF1v9fN8gzn7KA9NznzyTPFgZ7yiV2KPvANCEnsvs8MzvkfIHTSVIuLPNSZJzB2LRN+ckmdgP5I07wziWkDrWjjgNMBSQfumnC7/L/JAAQ/TmOUzzPQ4GRIwy7Bp4rz0nFUPuOHBeMfZ/J88M/AR5nHwsdJF0cHp7/J9rgsFj/yT//o/bCnJr5RU3PeZOx1ikzUL9vkytlT7fNuMITlS9bo/w8y68cQjzhf4kbo6Z9HTsXwTTA5/nTu4aFZdwPTsdhnWJK5//+B18RDsZvv3waHxWemFZ19fOrLju7xn39+ZYG855jYbWqSpl3P3zc9kwAf+zd5TSeQ3f+9WWORFYDPf8ZGFhFQvyFtAMjFmuSbpmid0cnq6MSuYQRhT7L/3Is2OnReNsPGJv4f9Bb0TANE5n3wQkTj4/Emg4P7fDG13Do/gZgQVut3cjs8jubn/babJiI4aoDYFGD2qB0mmCxIFvPu5QGyEJf5GYj+56dQ1EMBfLBuU2NgQzja8LvT5+k7n5w"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 8, "EERhBSI"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 16, "VApCdqqSaiS0lZobZimMbCSN"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 32, "qjWoVmxWzFgmmAwmWCuqmknKTU+IKlwa1Sa0jKSaYBwhw1YSsqOKws611qSVhkiWq1RDCmQpNKKRrbI0azQ0ZqHk6tQ"),
            (FilterType::CatmullRom, HashAlg::DoubleGradient, 64, "jcwlxolJLwtOWy0Lm8yYMbotPTM00ViRknFKmPC00pEl2cRTLR2t0zONaSekoayljSWXmqIW25LIlFXDyJ3FrcuRmrVXlkRwRFhaNSVLybRL7s5MSpaMJLoTxZqxIZoZpuI4qGStZMt1VmljdUTU4jZmRpnNNWFBUSJarcdqS3aSTlYjWRebIhlkThMp2gzqmslTSYmpmlE5qVmz3JFVpnJyW4dkNRjSrcZ0UrOGNWoypz3T6mFNzZLRMpEuSTIZpUwmilYi2JhIMpVacVSiSiAtpcy1UZ5KFdtK5kWj2pKZIqlzpEyZdN9kWYrLODmXyLZUNnJudqyajHaoT8Z2cqqoSXMemLQl"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 8, "lZsTIQisJJw"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 16, "VlKwocrGzrWWhDWWSBSjTEZKJKORqZIZazQsZqTk6tQ"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 32, "SRebIhlkThKp2izqmklTSYmpGtEcoVmr3NBZpmIyG8MsZzrSqcY0UjK2PWomLR3VynEgzUpJMpGu3SaZZU7Wmloy3Fh0WpdaYW0qzLUpnUo120pWRYPKhpmiqdK8Xpl012RZLssoOZPKtnSScGZWrJKMZqlLzjZyqihJc56YtCU"),
            (FilterType::CatmullRom, HashAlg::VertGradient, 64, "VDYyEcympk7SLRNnm7rcqjapJi3lEj2LW6tqXJ0ZxarirGmy1JScpG2NzebSbs62lM+MOJtmWrrapCKrhJnXsY2G8pya6pSxisrMZVKpm6eqQVJdySE1amVmAmYTV8JksbMBtSW1vNSb0pWVpaGmJq5qCCqLXbcmqTplhhVJprJkumVupVReaNGUNjXsakp7W9kstag1JqtX3XRSJRslqolOOaWSpcxdCU8tTGElRxpKGqVMZ5lZ8VpoawazyEeXzV0ZOaqts1gVs6ptlZpb9ZrGklU5p8XLq+RWmZImiWm4WVPVRi6pElZbUaNkTmXJspZoVidXTuGpsswkqTBp4VlnHY6maWWVyVU9CibZhhEyKWlZNmvSEViqTGsaQobKa2UypNwaWpIrmaIsVkdTcy6V5Y1m5vSsJc1ZM2pSarIzDUXxXGoUm5Oi2snONpiqU3EeZJZRrKxSaotlyraYkqtTrklH+s7J0sauaCZKFZQlT5YrRVGRF3XZdFomSzWLu/JmNikZ26xTIzUljaPKrNZ16SpXQpLijFrhyuYGmliJ0zztkCyrRKGklGYRa6VmhfKctpqbpk6dbEtZOC1KxstkMnQ7PXOMKkvR5VmaaGpFaVhZGRIKrSYZm3k5ViWtVsXYTOU0mbVVyZVEV6UFpTXFFVM5V4stJXetQ1XVWtI"),
            (FilterType::Gaussian, HashAlg::Blockhash, 8, "3M0FbCH1SXo"),
            (FilterType::Gaussian, HashAlg::Blockhash, 16, "Rmtwt5NMuVGZAHNooZz9P9IYk8R/t5BcqAnrPgTZ9pY"),
            (FilterType::Gaussian, HashAlg::Blockhash, 32, "+JhF7rcRfDgFvjlWJBsrVxfiaTYPDVOQ509GlqOFS7dL0gFG+1tsEIc2YaxOjy2/UU7QWqXdmZaTiZ0GUt37CqRmUrOM8spCTlc2pHfVukj+Dy1JzuedykxTZBgEh/si2OhlgqulS0TKOvRP6twsFxJrVwUVaMD+gh2+J3zecM4"),
            (FilterType::Gaussian, HashAlg::Blockhash, 64, "VVTBXKcKnFdaqibDc/X7xH8HZs/3W0SDJE0DI1BZyJ0zz7Xv4kg9sych6keNnokFcJyBuN+Xb4qkCs9lUYwcM7ZOh9SqOCkNj5GseqYmPL0dcOIyBD5Tz9+Fu4pXpoSVemr8PRp1r5dh2F/iaIC8CaeQE5hNAiKbSc8u5RNxPc+Pog2vrMRdczJ51nKRoJ7ajk3OV7IMBAl/9kZoT/UCYnHSFIsBNFKYJ0BeK1va9Fr/uLda0Dx3y+wXxUhpid0lSSZWsnC7fDvtG/g8DpOUwTaA42DDmyzBJ93129GgOsOPRaKodH9+kFlTbYqHgkEJRgFS387Wtphss/biT5/oFbCFZhwJF2tcPOx+mER0KZP2yRlHJfNMLBHiT/6s9G+i5yVdIjpPYk7oeJdmrOVwbjorItHOLWR5d6UTc4DGmpK8rLVB4phTkIen5wS9hWMm1XCxbOHJivCc0bt8Y6WuppjayHqU5tmHqjIXGxIfWlc1CS+wr+21SViRt8DtcQxaNGWcqmFtBuLFesLap2jOwe3qrqQnZCRheagz2YV0On5KgtUONP53YHz0XFUE430ZpATg4tE2IUPZ6lHX3mq/jYxWVZt+fjdADoduNDwboxKwy8Catfj0NhsnBDxK00fvBIX0UZT6WQxN54dK5Mc3baIuR3MkFn7asDvUx+M6nVA"),
            (FilterType::Gaussian, HashAlg::Gradient, 8, "RAJCFppiLDQ"),
            (FilterType::Gaussian, HashAlg::Gradient, 16, "qjWoFkxWzBgmkhgmOBqoig3OjWsICs02tIwkjmQYIcM"),
            (FilterType::Gaussian, HashAlg::Gradient, 32, "jcwlRolJLw/ISzwLm+wYM7gpOTMw0ViRsmFOknCQUpMpScRTJR0t0zCJOCWBZaSEhC3XmsjG1NLIlMXhydnYrdOR3rRnUlZwZ1ibtMtDzVxLwoxMmhHFkrEhmRum4RiIJK1kyHQWaWNxRFTiNkZGkck1IUFRJmAlx2JLcpZOFjM"),
            (FilterType::Gaussian, HashAlg::Gradient, 64, "KuFgsCVe2jSkzMKhmVZaNOXBYySbFXOJiaDBMVpNZpWdYIpxWUnNlTBRR2LhgouUi+VSa/TWiUZIpWFkxIWTnkmb6UxLlo+lQxrDiLPSEqdmnlW74BML48S2DaJoM6PRnacLqqxkq0OcyyFtJtWWzFJtMG2UcrPAMmfHtpwSjZIySadzJLAMM0xIZZpUxIYy5Cpjix2ExmlTWWGC45QGqRspo6XoFlKMEL6LhckiOkxGTVOsSZa6zRZBVTQpwlDJtplhRD7Jpot0mOsNZzme4lntfDdtUSrx0eU4qTFaO7HQsjGKM1sm3GK6c01zUmNMybLB0aptM8yWy5FRomWsxF5ak8msSTPHm1Iya2a5oMkbZCwpp1kmvVKSKGusWUUzNGCKQ4xBc0Z3/sxypWUabcpUpTpjcbRNitGpWExxSpvOEqrUWmVxmcRSJNbh9HKTysYeJ3VNhubYyioXs9GOoEduJhnbskzlZYyySZrHWxG8ZhPc7OKGTLzMStjkRmJDqQgu1bAmcms6VXJNtiyVeDY1VraiNIcoNFlY9mYZmzYlmXhjUisFVmsbZkky4zS9KZY4uFCyscEVtqhabhqDgUfMO2kzGqsFUTMzRpLIKroGNXouSj6zuiYRVJZkk3EiMlRZmN09SS2TcLJY21EZH1yzmCi2ew/bGc1ocGSLzcw"),
            (FilterType::Gaussian, HashAlg::Mean, 8, "fQ9N/ff/fOs"),
            (FilterType::Gaussian, HashAlg::Mean, 16, "V2fQdN+V+RH9EnEs4f7xvx7eH+f/+7//6InpPsE5bs4"),
            (FilterType::Gaussian, HashAlg::Mean, 32, "uLlvHD+Tfh4fvjk+ALs5J2fzYSdv53Gz588ekOPzBx/K+gnH+ntoAkU/cSwHj609G3rcvof9mZeDec/PFvO/7/y30Of8k/pDnwMWvBePnvj/n5lJ//+e/+9XMP/Nx/++GPjBgMvsQ0bHmfDHyNhMF4L5TwcGQMD9ng3Y4PydfOY"),
            (FilterType::Gaussian, HashAlg::Mean, 64, "wdfBZ+8A+HdIm4dH+9/4ZR4Hh0f/P2YD9g8HY9BZ/J+zx3RP8hjcF08A/G+DH54VBIiB6MUfOw43CM/piR0uPJMnh/kCPD8FnjQOO6eeP/7NPb4mBz4fhvt9Os7JdwbHfk7/dMlhgM8zkH/ybAEUwD/YZ8g9AKaHRf4e/TFw//dN8gzn7YA9c7nzyTPhAR7yiV2OP/ABDEH8/s8Mxv0eAHTS1ouDPdSZJzB+DwN+8gmPAP9I07zziXkD7WjigNMBSQPuWnD7/L/JAIQ/TmOczzOA4WRIwz/DJ4rz0mNQPuMHBeMfZ/J+8I/AR5/nwsdJF0cH5//D5/g8Bj/yT//p/bCnJ75RU2PeZOx1gkDwL9vk6FlT7fNuMITlS9bo/w8w78UQDjgP4k744Z9HTkXwTzApvnDu4aBbPwfTsdjnAJK9//+Bx8RDsJ/v/4zHxWeWtZ39fu/Jnu6xn39+ZMH+55zYf3qS5t/v3zk/kwAf/z953SeQ3f/9WWORHYDvf8ZGBhFQvyNtAMDBAuSbp2ic4c3q6NSuYRRgT7L/3I83OnBetsPGJP4/9Bb0SANA5n3wQnDh4/Emg4P7eHCz/DY/A4gQRv9/Mj88jubn/babLiI4aoDYFGD2qB0ikChAFvPv5QGwEJD7Gez856dA1GMRfPD+02NgQzja8P/T5/g/n5w"),
            (FilterType::Gaussian, HashAlg::Median, 8, "fQ9N/ff/fOs"),
            (FilterType::Gaussian, HashAlg::Median, 16, "V2fQdN+V+RH9EnEs4f7xvx7eH+f/+7//6InpPsE5bs4"),
            (FilterType::Gaussian, HashAlg::Median, 32, "uLlvHD+Tfh4fvjk+ALs5J2fzYSdv53Gz588ekOPzBx/K+gnH+ntoAkU/cSwHj609G3rcvof9mZeDec/PFvO/7/y30Of8k/pDnwMWvBePnvj/n5lJ//+e/+9XMP/Nx/++GPjBgMvsQ0bHmfDHyNhMF4L5TwcGQMD9ng3Y4PydfOY"),
            (FilterType::Gaussian, HashAlg::Median, 64, "wdfBZ+8A+HdIm4dH+9/4ZR4Hh0f/P2YD9g8HY9BZ/J+zx3RP8hjcF08A/G+DH54VBIiB6MUfOw43CM/piR0uPJMnh/kCPD8FnjQOO6eeP/7NPb4mBz4fhvt9Os7JdwbHfk7/dMlhgM8zkH/ybAEUwD/YZ8g9AKaHRf4e/TFw//dN8gzn7YA9c7nzyTPhAR7yiV2OP/ABDEH8/s8Mxv0eAHTS1ouDPdSZJzB+DwN+8gmPAP9I07zziXkD7WjigNMBSQPuWnD7/L/JAIQ/TmOczzOA4WRIwz/DJ4rz0mNQPuMHBeMfZ/J+8I/AR5/nwsdJF0cH5//D5/g8Bj/yT//p/bCnJ75RU2PeZOx1gkDwL9vk6FlT7fNuMITlS9bo/w8w78UQDjgP4k744Z9HTkXwTzApvnDu4aBbPwfTsdjnAJK9//+Bx8RDsJ/v/4zHxWeWtZ39fu/Jnu6xn39+ZMH+55zYf3qS5t/v3zk/kwAf/z953SeQ3f/9WWORHYDvf8ZGBhFQvyNtAMDBAuSbp2ic4c3q6NSuYRRgT7L/3I83OnBetsPGJP4/9Bb0SANA5n3wQnDh4/Emg4P7eHCz/DY/A4gQRv9/Mj88jubn/babLiI4aoDYFGD2qB0ikChAFvPv5QGwEJD7Gez856dA1GMRfPD+02NgQzja8P/T5/g/n5w"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 8, "EERABSI"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 16, "VAJCJjKYKiw0kZoZISGMLCSM"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 32, "qjWoFmxWzFgkmAwGGCqomknKDU+IalgaxSakjKSYZIwhw3YSkoOKw86zxoaVhgiWK1RDCmQodKKRrbIkazQ0JqDm48Q"),
            (FilterType::Gaussian, HashAlg::DoubleGradient, 64, "jcwlxolJLQ9KSz0L28yYM5ApODM0kRiRknFKmPC00pMl2cRTLR0P0zOJKSegobykjSWWnqIW3ZLIhEXDyJ3EqcuZmqRTklxwZFhaNGNJy7RLwM5MSpKMBLoRxZqjIZobpuI4iCStYMl1VmljcUTU4jZmBpHJNWFBUSZKJcdiS3KWTBYjeTYbIhlmThIpygzqmslTSZnpGtEZqRmT3NBZhnByG4dkdBjSLeY00rOGNWoyJh3D4iEMzZJRMpkOSTCZpUw2m0Zm2JhYMpRYcFyiSCAthsy1WZ5KFZtK5gWjy5YZIqlzlEwZJN90GY7LODmTyLY0tnBOdqwazHaoC852cuqoaXMcmDRl"),
            (FilterType::Gaussian, HashAlg::VertGradient, 8, "kJkDYQQsJIw"),
            (FilterType::Gaussian, HashAlg::VertGradient, 16, "dhKQg4rHxrWWhBSWSBRjFkRKZKGRqZIcayQoZqDm48Q"),
            (FilterType::Gaussian, HashAlg::VertGradient, 32, "eTYbIhlmThIpygzqmMlTSYnpGtEcoRmL3NBZhnAyG8IsZzjSucY0UjKGPUoiJR3TwnEwzYpZMpmu3DKZZW7Gmkwy3BhwWodaYG2qzLEpjUoV20rGBYPKxpmj6ZI8Tpl0z2QZDss4OYbKtjSScG52rBLMdqwLznZy6qhocxyYNGU"),
            (FilterType::Gaussian, HashAlg::VertGradient, 64, "3jYyE8ympg7TLRMn27qcqhKpJiflEp2LU69kWJ0ZxCvirG320JScpE+NzebSbM62lM+MuJtmUrrEpKKbhJnXsY+G4pyeopCxgsjMZcKpi6eiQUZdySE1YmVmAmYTV+JksbMBpyW1vMSb0xWVpbGmpqxqCDqLWbcmqTplhoVJJrJgHmVupVVeedMcNjWsakp7U9ksNagfJqtXzWRQJx8kColPPeUSpczYCU8tzGEFRxpaGq0MZ4lZ81poawazyEeTTV0bOaq8s1kds6JthZtb+ZrGklEZp83Lu2RWmZInicO4WdPVRi6hElbbUXNkTmXJMp5oVidXTuFpssxmrHFp4VlnHY6maWSViXE9CibJhhEyKW1ZNmuSEVquTGseYsbYay0yLNxa2pArOeIsVkZTcSeV5Y125vSsJY1FM2JSarIzDUfxXGIEu5OjmsnONpiqE2EeZJ5xjKxTSo9lyvaYkqNCrklH+syZ0saOSWbKlZQlT5YrZUERFzHZdFJmSzGLufJkNqkZ24wTYzUkjaPKrNZx6ShHRprijFDhyuYGmliN0RztEi6PRKGwlGYRa6VmhXKcPpobNkedbFtZOC1CxotkcnQ6HXPMKkvQ5Vma6GpMadjZGRcIrWYZm3k5ViytRs3YTOU0nTVUwZVFRyWHJTTHlVMZV5stJSelQ1FVOtY"),
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
