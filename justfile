bench:
    cargo +nightly bench

format:
    cargo +nightly fmt

init:
    cargo install nightly
    cargo install stable

fix:
    cargo +nightly fmt
    cargo clippy --fix --allow-dirty --allow-staged ---Wclippy::bool_to_int_with_if -Wclippy::expl_impl_clone_on_copy -Wclippy::explicit_into_iter_loop -Wclippy::explicit_iter_loop -Wclippy::filter_map_next -Wclippy::flat_map_option -Wclippy::float_cmp -Wclippy::from_iter_instead_of_collect -Wclippy::ignored_unit_patterns -Wclippy::implicit_clone -Wclippy::index_refutable_slice -Wclippy::invalid_upcast_comparisons -Wclippy::iter_filter_is_ok -Wclippy::iter_filter_is_some -Wclippy::large_stack_arrays -Wclippy::large_types_passed_by_value -Wclippy::macro_use_imports -Wclippy::manual_assert
    cargo +nightly fmt

test:
    cargo +nightly test
    cargo +nightly test --feature

upgrade:
    cargo +nightly -Z unstable-options update --breaking
    cargo update