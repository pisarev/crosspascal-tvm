cmake_minimum_required(VERSION 3.20)
project(crosspascal_sentencepiece LANGUAGES CXX)

include(FetchContent)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(SPM_PROTOBUF_PROVIDER internal CACHE STRING "" FORCE)
FetchContent_Declare(sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG v0.2.1
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(sentencepiece)

add_library(crosspascal_sentencepiece SHARED sentencepiece_c.cpp)
target_compile_features(crosspascal_sentencepiece PRIVATE cxx_std_17)
target_include_directories(crosspascal_sentencepiece PRIVATE
  "${sentencepiece_SOURCE_DIR}/src"
  "${sentencepiece_SOURCE_DIR}/third_party/protobuf-lite")
target_link_libraries(crosspascal_sentencepiece PRIVATE sentencepiece-static)
