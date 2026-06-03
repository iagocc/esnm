cmake_minimum_required(VERSION 3.15...3.26)

project(esnm LANGUAGES CXX)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(Python_FIND_VIRTUALENV FIRST)
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")

find_package(Python 3.13
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

find_package(OpenMP REQUIRED)

nanobind_add_module(
  percentile
  STABLE_ABI
  NB_STATIC
  src/percentile/percentile.cpp
)

nanobind_add_module(
  mechanism
  STABLE_ABI
  NB_STATIC
  src/mechanism.cpp
  src/distributions/prng.cpp
)

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
link_directories(${GSL_LIBRARY_DIRS})
target_link_libraries(mechanism PRIVATE GSL::gsl GSL::gslcblas OpenMP::OpenMP_CXX)

install(TARGETS percentile mechanism LIBRARY DESTINATION ${PROJECT_NAME})
