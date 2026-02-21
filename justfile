bench:
    cargo +nightly bench

format:
    cargo +nightly fmt

init:
    cargo install nightly
    cargo install stable

fix:
    cargo +nightly fmt
    cargo clippy --fix --allow-dirty --allow-staged --all-targets
    cargo clippy --fix --allow-dirty --allow-staged --all-targets --all-features
    cargo +nightly fmt

fixn:
    RUSTUP_TOOLCHAIN=nightly just fix

test:
    cargo +nightly test
    cargo +nightly test --feature

upgrade:
    cargo +nightly -Z unstable-options update --breaking
    cargo update