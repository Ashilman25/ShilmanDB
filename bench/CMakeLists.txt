# Every tool lands in build/bench/. Includes are relative to bench/, like src/.

add_executable(load_tpch tpch/load_tpch.cpp)
target_link_libraries(load_tpch PRIVATE shilmandb_core)

add_executable(generate_join_training_data join/generate_join_training_data.cpp)
target_link_libraries(generate_join_training_data PRIVATE shilmandb_core)

add_executable(join_headroom join/join_headroom.cpp)
target_link_libraries(join_headroom PRIVATE shilmandb_core)

add_executable(eviction_sim eviction/sim/eviction_sim.cpp)
add_executable(eviction_samples eviction/sim/eviction_samples.cpp)
add_executable(eviction_switch eviction/sim/eviction_switch.cpp)

foreach(tool load_tpch generate_join_training_data join_headroom eviction_sim eviction_samples eviction_switch)
    target_include_directories(${tool} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

if(APPLE)
    foreach(tool eviction_sim eviction_samples eviction_switch)
        target_link_libraries(${tool} PRIVATE "-framework Accelerate")
    endforeach()
endif()
