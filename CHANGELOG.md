# 3.0.0 - 29.12.2024
- Minimal Rust version set to 1.70.0(required by newer image-rs, update needed by fast_image_resize)
- Fix difference of gaussians prefilter - [#21](https://github.com/qarmin/img_hash/pull/21)
- Fixed u8 overflow in median hash - [#b8b](https://github.com/qarmin/img_hash/commit/b8bfa7c8e5fb48840706eb1d1e101c9af8328328)
- Added to CI, hash stability check to avoid accidental changes - [#23](https://github.com/qarmin/img_hash/pull/23)
- Added new unstable feature(it may be removed in later versions or change behavior) `fast_image_resize` to speed up image resizing(gives slightly different results than image-rs) - [#24](https://github.com/qarmin/img_hash/pull/24)

# 2.0.0 - 11.03.2024
- Update to image 0.25
- Minimal Rust version set to 1.67.1
- Added median hash - [#13](https://github.com/qarmin/img_hash/pull/13)
- Added bit ordering - [#14](https://github.com/qarmin/img_hash/pull/14)
- Added into_inner() function on ImageHash object - [#11](https://github.com/qarmin/img_hash/pull/11)

# 1.2.0 - 02.06.2023
- Update base64 to 0.21.0 - [#8](https://github.com/qarmin/img_hash/pull/8)

# 1.1.2 - 26.11.2022
- Revert base64 version to 0.13.1
- Set minimal Rust version to 1.61

# 1.1.1 - 20.11.2022
- Rustdct fix, criterion/benchmark update, tests fixes - [#3](https://github.com/qarmin/img_hash/pull/3)

# 1.1 - 20.10.2022
- Added CI, formatted code - [#2](https://github.com/qarmin/img_hash/pull/2)
- Update to rustdct 0.7 - [#1](https://github.com/qarmin/img_hash/pull/1)

# 1.0 - 02.04.2022
- First version without any logic changes with updated dependencies - [47e](47e4e243f79e170291580e2fb914b53b749cead6)
- Some clippy changes - [8da](8da30ed6e46697fa1ab99a664b579e51e62dc6ae)
