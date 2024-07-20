cmake_minimum_required(VERSION 3.16)
project(guts_utils_py)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(guts_utils REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(
  ${guts_utils_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

link_directories(${guts_utils_LIBRARY_DIR})

pybind11_add_module(guts_utils src/guts_utils_py/guts_utils.cpp)
target_link_libraries(guts_utils
  PRIVATE ${PYTHON_LIBRARIES}
  PRIVATE ${guts_utils_LIBRARIES})
set_target_properties(guts_utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

pybind11_add_module(grid src/guts_utils_py/grid.cpp)
target_link_libraries(grid PRIVATE ${PYTHON_LIBRARIES})
set_target_properties(grid PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

install(TARGETS
  DESTINATION ${CMAKE_INSTALL_PREFIX})
