cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(load_torch_model load_torch_model.cpp)
target_link_libraries(load_torch_model "${TORCH_LIBRARIES}")
set_property(TARGET load_torch_model PROPERTY CXX_STANDARD 14)