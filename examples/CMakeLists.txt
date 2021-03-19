add_executable(for_each for_each.cpp) 
target_link_libraries(for_each PRIVATE AsyncMPI)

add_executable(bulk_finally bulk_finally.cpp) 
target_link_libraries(bulk_finally PRIVATE AsyncMPI)

add_executable(bulk_sequence bulk_sequence.cpp) 
target_link_libraries(bulk_sequence PRIVATE AsyncMPI)