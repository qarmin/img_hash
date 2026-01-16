bench:
    cargo +nightly bench

format:
    cargo +nightly fmt

init:
    cargo install nightly
    cargo install stable

fix:
    cargo +nightly fmt
    cargo clippy --fix --allow-dirty --allow-staged
    cargo clippy --fix --allow-dirty --allow-staged --features fast_resize_unstable
    cargo +nightly fmt

test:
    cargo +nightly test
    cargo +nightly test --feature

upgrade:
    cargo +nightly -Z unstable-options update --breaking
    cargo update