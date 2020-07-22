cmake_minimum_required(VERSION 3.1)
project(DRL_graph_exploration)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})
include_directories("${GTSAM_INCLUDE_DIR}/gtsam/3rdparty/Eigen")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # Force to use Python 2.7/3.5
    # set(PYTHON_EXECUTABLE /usr/local/bin/python)
    set(PYTHON_EXECUTABLE /usr/local/bin/python3)
else ()
    # set(PYTHON_EXECUTABLE /usr/bin/python)
    set(PYTHON_EXECUTABLE /usr/bin/python3)
endif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

find_package(pybind11 REQUIRED)

include_directories(include)

set(SOURCE
    src/halton/halton.cpp
    src/em_exploration/Utils.cpp
    src/em_exploration/Distance.cpp
    src/em_exploration/Simulator2D.cpp
    src/em_exploration/SLAM2D.cpp
    src/em_exploration/FastMarginals.cpp
    src/em_exploration/VirtualMap.cpp
    src/em_exploration/OccupancyMap.cpp
    src/em_exploration/Planner2D.cpp)

pybind11_add_module(ss2d src/SS2D.cpp ${SOURCE})
target_link_libraries(ss2d PRIVATE gtsam flann)

pybind11_add_module(planner2d src/Planner2D.cpp ${SOURCE})
target_link_libraries(planner2d PRIVATE gtsam flann)

install(TARGETS ss2d planner2d LIBRARY DESTINATION lib)
