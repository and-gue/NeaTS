cmake_minimum_required(VERSION 3.22)
project(NeaTS)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -mfpmath=sse -mavx512f -fno-math-errno -ldl")
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -D_FILE_OFFSET_BITS=64 -m64 -O9 -funroll-loops -lm -lrt -ldl")

add_library(sux INTERFACE)
target_include_directories(sux INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib/sux)

add_library(sdsl INTERFACE)
target_include_directories(sdsl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib/sdsl-lite/include)

add_executable(LeaTS LeaTS.cpp)
target_link_libraries(LeaTS PRIVATE sux sdsl)

add_executable(DecompressorSIMD main_simd.cpp)
target_link_libraries(DecompressorSIMD PRIVATE sux sdsl)

add_executable(SNeaTS SNeaTS.cpp)
target_link_libraries(SNeaTS PRIVATE sux sdsl)

add_executable(NeaTSL NeaTSL.cpp)
target_link_libraries(NeaTSL PRIVATE sux sdsl)

add_executable(LossyNeaTS lossyNeaTS.cpp)
target_link_libraries(LossyNeaTS PRIVATE sux sdsl)

include_directories(/usr/local/include/squash-0.7)
link_directories(/usr/local/lib)

add_executable(Benchmark benchmark/neats_lossless.cpp
        include/NeaTSL.hpp
        include/algorithms.hpp
        include/coefficients_space.hpp)
target_link_libraries(Benchmark sdsl squash0.7 sux)

add_executable(NeatsPGM include/simple.cpp)
target_link_libraries(NeatsPGM sdsl sux)