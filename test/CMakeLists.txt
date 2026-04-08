# Helper macro: define a test executable, link it, and register with CTest
macro(shilmandb_add_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE shilmandb_core GTest::gtest_main)
    gtest_discover_tests(${TEST_NAME})
endmacro()

# Uncomment each line as you create the corresponding test file:

# Types tests
shilmandb_add_test(value_test types/value_test.cpp)
shilmandb_add_test(tuple_test types/tuple_test.cpp)

# Storage tests
# shilmandb_add_test(disk_manager_test storage/disk_manager_test.cpp)
# shilmandb_add_test(slotted_page_test storage/slotted_page_test.cpp)
# shilmandb_add_test(table_heap_test storage/table_heap_test.cpp)

# Buffer tests
# shilmandb_add_test(lru_eviction_policy_test buffer/lru_eviction_policy_test.cpp)
# shilmandb_add_test(buffer_pool_manager_test buffer/buffer_pool_manager_test.cpp)

# Index tests
# shilmandb_add_test(btree_index_test index/btree_index_test.cpp)

# Parser tests
# shilmandb_add_test(lexer_test parser/lexer_test.cpp)
# shilmandb_add_test(parser_test parser/parser_test.cpp)

# Planner tests
# shilmandb_add_test(planner_test planner/planner_test.cpp)

# Executor tests
# shilmandb_add_test(seq_scan_executor_test executor/seq_scan_executor_test.cpp)
# shilmandb_add_test(filter_executor_test executor/filter_executor_test.cpp)
# shilmandb_add_test(hash_join_executor_test executor/hash_join_executor_test.cpp)
# shilmandb_add_test(sort_executor_test executor/sort_executor_test.cpp)
# shilmandb_add_test(aggregate_executor_test executor/aggregate_executor_test.cpp)
# shilmandb_add_test(projection_executor_test executor/projection_executor_test.cpp)

# Integration tests
# shilmandb_add_test(end_to_end_test integration/end_to_end_test.cpp)
# shilmandb_add_test(sqlite_comparison_test integration/sqlite_comparison_test.cpp)
