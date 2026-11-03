add_executable(load_tpch load_tpch.cpp)
target_link_libraries(load_tpch PRIVATE shilmandb_core)

add_executable(generate_join_training_data generate_join_training_data.cpp)
target_link_libraries(generate_join_training_data PRIVATE shilmandb_core)

add_executable(join_headroom join_headroom.cpp)
target_link_libraries(join_headroom PRIVATE shilmandb_core)

add_executable(eviction_sim eviction_sim.cpp)
target_include_directories(eviction_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE)
    target_link_libraries(eviction_sim PRIVATE "-framework Accelerate")
endif()
