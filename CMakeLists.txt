cmake_minimum_required(VERSION 3.23)
project(NeaTS)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mfpmath=sse -ffloat-store -fno-math-errno -mavx512f")
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)


include_directories(include)
include_directories(lib/boost_1_83_0/)
include_directories(lib/vectorclass)

add_library(sux INTERFACE)
target_include_directories(sux INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib/sux)

add_subdirectory(lib/sdsl-lite)

include_directories(/usr/local/include/squash-0.7)
link_directories(/usr/local/lib)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
add_executable(main main.cpp
        include/utils.hpp
        include/NeaTSL.hpp
        include/algorithms.hpp)
target_link_libraries(main sdsl sux)

add_executable(Google_Tests_run test/test.cpp
        include/utils.hpp
        include/NeaTSL.hpp
        include/algorithms.hpp
        include/float_coefficient_space.hpp
)
target_link_libraries(Google_Tests_run gtest gtest_main)
target_link_libraries(Google_Tests_run sdsl sux)

add_executable(bench benchmark/neats_lossless.cpp
        include/utils.hpp
        include/NeaTSL.hpp
        include/algorithms.hpp
        include/float_coefficient_space.hpp)
target_link_libraries(bench sdsl squash0.7 sux)