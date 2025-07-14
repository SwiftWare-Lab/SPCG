cmake_minimum_required(VERSION 3.0)
project(conjugateGradientPrecond CXX)

find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(conjugateGradientPrecond main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(conjugateGradientPrecond PUBLIC OpenMP::OpenMP_CXX ${EIGEN3_LIBRARIES})
else()
    target_link_libraries(conjugateGradientPrecond ${EIGEN3_LIBRARIES})
endif()
