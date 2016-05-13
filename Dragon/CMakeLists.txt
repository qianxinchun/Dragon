PROJECT(Dragon)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

# ---[ Dependencies
set(3RDPARTY_DIR C:/3rdparty)
set(PYTHON_DIR C:/Anaconda)
set(NUMPY_DIR C:/Anaconda/Lib/site-packages/numpy/core)
FIND_PACKAGE(CUDA REQUIRED)

# ---[ Config types
set(CMAKE_BUILD_TYPE Release CACHE STRING "set build type to release")
set(CMAKE_CONFIGURATION_TYPES  Release CACHE STRING "set build type to release" FORCE)

# ---[ Includes
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${INCLUDE_DIR})
include_directories(${3RDPARTY_DIR}/include)
include_directories(${PYTHON_DIR}/include)
include_directories(${NUMPY_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)

# ---[ libs
set(3RDPARTY_LIBS ${3RDPARTY_DIR}/lib)
link_directories(${3RDPARTY_LIBS})

# ---[ Install
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR} CACHE STRING "set install prefix" FORCE)

# ---[ Subdirectories
add_subdirectory(src)

# ---[ Flags

# ---[ Warnings




