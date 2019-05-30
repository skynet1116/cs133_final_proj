cmake_minimum_required(VERSION 3.5)
project(cs133final)

set(CS133_FINAL_VERSION_MAJOR 1)
set(CS133_FINAL_VERSION_MINOR 0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

include(CTest)
include(${PROJECT_SOURCE_DIR}/cmake/misc.cmake)
include_directories(${PROJECT_SOURCE_DIR}/include)

configure_file(${PROJECT_SOURCE_DIR}/cmake/get_eigen.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/eigen-download/CMakeLists.txt)
setup_eigen()

set(CS133_FINAL_LIB ${PROJECT_NAME})
set(CS133_FINAL_SRC
        ${PROJECT_SOURCE_DIR}/src/Network.cpp)
add_library(${CS133_FINAL_LIB} ${CS133_FINAL_SRC})
target_link_libraries(${CS133_FINAL_LIB} Eigen3::Eigen)

add_executable(linear_test ${PROJECT_SOURCE_DIR}/test/test_linear.cpp)
target_link_libraries(linear_test ${CS133_FINAL_LIB})

add_executable(conv_test ${PROJECT_SOURCE_DIR}/test/test_conv.cpp)
target_link_libraries(conv_test ${CS133_FINAL_LIB})


add_executable(cnn ${PROJECT_SOURCE_DIR}/test/unit_test_conv.cpp)
target_link_libraries(cnn ${CS133_FINAL_LIB})
add_executable(unit_test_max_pool ${PROJECT_SOURCE_DIR}/test/unit_test_max_pool.cpp)
target_link_libraries(unit_test_max_pool ${CS133_FINAL_LIB})
