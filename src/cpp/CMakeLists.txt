cmake_minimum_required(VERSION 3.10)
project(AI_DLC_labs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

find_package(Armadillo REQUIRED)

# Исполняемый файл для K-Means
add_executable(kmeans my_kmeans.cpp)
target_link_libraries(kmeans ${ARMADILLO_LIBRARIES})
target_include_directories(kmeans PRIVATE ${ARMADILLO_INCLUDE_DIRS})