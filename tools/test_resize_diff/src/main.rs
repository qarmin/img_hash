use image::{imageops, DynamicImage, GenericImageView};

use clap::Parser;
use fast_image_resize::{ResizeAlg, ResizeOptions, Resizer};
use image::imageops::FilterType;
use log::{error, info};
use rayon::prelude::*;
use rand::random;

const RESIZE_SIZES: &[u32] = &[8, 64];//, 512];

// Usage example
// For images with alpha channel(some have really big differences):
// cargo run --release -- -i /home/rafal/ -o /home/rafal/test/F -d 1500000
// And without
// cargo run --release -- -i /home/rafal/ -o /home/rafal/test/F -d 1500 --no_alpha

#[derive(Parser)]
#[command(
    name = "Image Resizer",
    version = "1.0",
    about = "Resizes images and checks pixel differences"
)]
struct Cli {
    /// Sets the input folder
    #[arg(short, long, value_name = "INPUT")]
    input: String,

    /// Sets the output folder
    #[arg(short, long, value_name = "OUTPUT")]
    output: String,

    /// Sets diff, that is the maximum difference between the two images
    #[arg(short, long, value_name = "DIFF")]
    diff: u32,

    /// Ignore images with alpha channel
    #[arg(short, long)]
    no_alpha: bool,
}

fn main() {
    handsome_logger::init().unwrap();

    let args = Cli::parse();

    let input_folder = args.input;
    let output_folder = args.output;
    let max_diff = args.diff;
    let no_alpha = args.no_alpha;

    let files = collect_files(&input_folder);

    info!("Found {} files", files.len());

    let _ = std::fs::create_dir_all(&output_folder);

    files.into_par_iter().for_each(|file| {
        test_image_resize(&file, &output_folder, max_diff, no_alpha);
    });
}

fn collect_files(input: &str) -> Vec<String> {
    let mut paths = vec![];
    for file in jwalk::WalkDir::new(input)
        .max_depth(1000)
        .into_iter()
        .flatten()
    {
        let path_str = file.file_name.to_string_lossy();
        if path_str.ends_with(".jpg") || path_str.ends_with(".png") {
            paths.push(file.path().to_string_lossy().to_string());
        }
    }
    paths
}

fn test_image_resize(file: &str, output: &str, max_diff: u32, no_alpha: bool) {
    let base_image = match image::open(file) {
        Ok(img) => img,
        Err(e) => {
            info!("Error while reading image: {:?}", e);
            return;
        }
    };
    if no_alpha && base_image.color().has_alpha() {
        return;
    }
    // info!("Testing image: {}", file);

    let filters = &[
        (
            FilterType::Lanczos3,
            ResizeAlg::Convolution(fast_image_resize::FilterType::Lanczos3),
        ),
        (
            FilterType::Gaussian,
            ResizeAlg::Convolution(fast_image_resize::FilterType::Gaussian),
        ),
        (
            FilterType::CatmullRom,
            ResizeAlg::Convolution(fast_image_resize::FilterType::CatmullRom),
        ),
        (
            FilterType::Triangle,
            ResizeAlg::Convolution(fast_image_resize::FilterType::Bilinear),
        ),
        (FilterType::Nearest, ResizeAlg::Nearest),
    ];


    let random_number: u64 = random();

    for (img, fast_img) in filters {
        for size in RESIZE_SIZES {
            let size = *size;
            let algorithm = match img {
                FilterType::Lanczos3 => "Lanczos3",
                FilterType::Gaussian => "Gaussian",
                FilterType::CatmullRom => "CatmullRom",
                FilterType::Triangle => "Triangle",
                FilterType::Nearest => "Nearest",
            };

            let n = format!("{random_number}_{algorithm}_{size}");

            // let resized_img = base_image.resize_exact(size, size, *img);
            let resized_img: DynamicImage = imageops::resize(&base_image, size, size, *img).into();

            let resize_options = ResizeOptions::new().resize_alg(fast_img.clone());
            let mut fast_resized_img = DynamicImage::new(size, size, base_image.color());
            Resizer::new()
                .resize(&base_image, &mut fast_resized_img, &resize_options)
                .unwrap();

            let diff = pixel_difference(&resized_img, &fast_resized_img, algorithm);
            if let Some(diff) = diff {
                if diff > max_diff {
                    resized_img.save(format!("{output}/{n}_normal.png")).unwrap();
                    fast_resized_img.save(format!("{output}/{n}_fast.png")).unwrap();
                    base_image.save(format!("{output}/{n}_base_diff_{diff}.png")).unwrap();

                    info!("File {file} - difference: {diff}, {n}");
                }
            }
        }
    }
}

fn has_alpha_channel_used(img: &DynamicImage) -> bool {
    img.pixels().any(|p| p.2.0[3] != 255)
}

fn pixel_difference(img1: &DynamicImage, img2: &DynamicImage, algorithm: &str) -> Option<u32> {
    let (width1, height1) = img1.dimensions();
    let (width2, height2) = img2.dimensions();

    if width1 != width2 || height1 != height2 {
        error!(
            "Images dimensions do not match ({}, {}) vs ({}, {}) - algorithm: {}",
            width1, height1, width2, height2, algorithm
        );
        return None;
    }

    let mut diff = 0;
    for y in 0..height1 {
        for x in 0..width1 {
            let p1 = img1.get_pixel(x, y);
            let p2 = img2.get_pixel(x, y);
            let pixel_diff = (p1[0] as i32 - p2[0] as i32).abs() as u32
                + (p1[1] as i32 - p2[1] as i32).abs() as u32
                + (p1[2] as i32 - p2[2] as i32).abs() as u32;
            diff += pixel_diff;

            // Log pixel differences for debugging
            // if pixel_diff > 0 {
            //     info!(
            //         "Pixel difference at ({}, {}): {:?} vs {:?} - diff: {}",
            //         x, y, p1, p2, pixel_diff
            //     );
            // }
        }
    }
    Some(diff)
}