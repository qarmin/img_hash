bench:
    cargo +nightly bench

format:
    cargo +nightly fmt

init:
    cargo install nightly
    cargo install stable

fix:
    cargo +nightly fmt
    cargo clippy --fix --allow-dirty --allow-staged -- -Wclippy::bool_to_int_with_if -Wclippy::expl_impl_clone_on_copy -Wclippy::explicit_into_iter_loop -Wclippy::explicit_iter_loop -Wclippy::filter_map_next -Wclippy::flat_map_option -Wclippy::float_cmp -Wclippy::from_iter_instead_of_collect -Wclippy::ignored_unit_patterns -Wclippy::implicit_clone -Wclippy::index_refutable_slice -Wclippy::invalid_upcast_comparisons -Wclippy::iter_filter_is_ok -Wclippy::iter_filter_is_some -Wclippy::large_stack_arrays -Wclippy::large_types_passed_by_value -Wclippy::macro_use_imports -Wclippy::manual_assert -Wclippy::manual_instant_elapsed  -Wclippy::manual_is_power_of_two  -Wclippy::manual_is_variant_and  -Wclippy::manual_let_else  -Wclippy::manual_ok_or  -Wclippy::map_unwrap_or  -Wclippy::match_bool  -Wclippy::match_on_vec_items  -Wclippy::match_same_arms  -Wclippy::match_wildcard_for_single_variants  -Wclippy::missing_panics_doc  -Wclippy::mut_mut  -Wclippy::needless_bitwise_bool  -Wclippy::needless_continue  -Wclippy::needless_for_each  -Wclippy::needless_pass_by_value  -Wclippy::option_as_ref_cloned  -Wclippy::range_minus_one  -Wclippy::range_plus_one  -Wclippy::redundant_else  -Wclippy::ref_binding_to_reference  -Wclippy::ref_option  -Wclippy::ref_option_ref  -Wclippy::same_functions_in_if_condition  -Wclippy::semicolon_if_nothing_returned  -Wclippy::stable_sort_primitive  -Wclippy::str_split_at_newline  -Wclippy::string_add_assign  -Wclippy::uninlined_format_args  -Wclippy::unnecessary_box_returns  -Wclippy::unnecessary_join   -Wclippy::unnecessary_wraps  -Wclippy::unnested_or_patterns  -Wclippy::used_underscore_binding  -Wclippy::used_underscore_items   -Aclippy::match_same_arms
    cargo clippy --fix --allow-dirty --allow-staged --features fast_resize_unstable -- -Wclippy::bool_to_int_with_if -Wclippy::expl_impl_clone_on_copy -Wclippy::explicit_into_iter_loop -Wclippy::explicit_iter_loop -Wclippy::filter_map_next -Wclippy::flat_map_option -Wclippy::float_cmp -Wclippy::from_iter_instead_of_collect -Wclippy::ignored_unit_patterns -Wclippy::implicit_clone -Wclippy::index_refutable_slice -Wclippy::invalid_upcast_comparisons -Wclippy::iter_filter_is_ok -Wclippy::iter_filter_is_some -Wclippy::large_stack_arrays -Wclippy::large_types_passed_by_value -Wclippy::macro_use_imports -Wclippy::manual_assert -Wclippy::manual_instant_elapsed  -Wclippy::manual_is_power_of_two  -Wclippy::manual_is_variant_and  -Wclippy::manual_let_else  -Wclippy::manual_ok_or  -Wclippy::map_unwrap_or  -Wclippy::match_bool  -Wclippy::match_on_vec_items  -Wclippy::match_same_arms  -Wclippy::match_wildcard_for_single_variants  -Wclippy::missing_panics_doc  -Wclippy::mut_mut  -Wclippy::needless_bitwise_bool  -Wclippy::needless_continue  -Wclippy::needless_for_each  -Wclippy::needless_pass_by_value  -Wclippy::option_as_ref_cloned  -Wclippy::range_minus_one  -Wclippy::range_plus_one  -Wclippy::redundant_else  -Wclippy::ref_binding_to_reference  -Wclippy::ref_option  -Wclippy::ref_option_ref  -Wclippy::same_functions_in_if_condition  -Wclippy::semicolon_if_nothing_returned  -Wclippy::stable_sort_primitive  -Wclippy::str_split_at_newline  -Wclippy::string_add_assign  -Wclippy::uninlined_format_args  -Wclippy::unnecessary_box_returns  -Wclippy::unnecessary_join   -Wclippy::unnecessary_wraps  -Wclippy::unnested_or_patterns  -Wclippy::used_underscore_binding  -Wclippy::used_underscore_items   -Aclippy::match_same_arms
    cargo +nightly fmt

test:
    cargo +nightly test
    cargo +nightly test --feature

upgrade:
    cargo +nightly -Z unstable-options update --breaking
    cargo update