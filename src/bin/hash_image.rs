#![allow(clippy::indexing_slicing)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::print_stdout)]
//! Hash an image and print the Base64 value

use std::env;

use image_hasher::HasherConfig;

fn main() -> Result<(), String> {
    let mut args = env::args().collect::<Vec<_>>();
    assert!(args.len() >= 2);

    let hash_only = args.contains(&"--hash-only".to_string());
    if hash_only {
        args.retain(|x| x != "--hash-only");
    }

    let image = image::open(&args[1]).map_err(|e| format!("failed to open {}: {}", &args[1], e))?;

    let hash = HasherConfig::new()
        .hash_size(8, 8)
        .to_hasher()
        .hash_image(&image);

    #[expect(clippy::format_collect)]
    let hash_str = hash
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();

    if hash_only {
        println!("{}", hash_str);
    } else {
        println!("{}: {}", &args[1], hash_str);
    }

    Ok(())
}
