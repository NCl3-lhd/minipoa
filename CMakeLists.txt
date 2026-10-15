cmake_minimum_required(VERSION 3.9)

project(minipoa VERSION 1.5.3
    LANGUAGES C CXX
    DESCRIPTION "Minipoa: A minimizer-based method for fast and memory-efficient partial order alignment.")

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

# 要求必须支持指定的C++标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 禁用编译器特定的扩展，只使用标准C++
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置默认构建类型为Release（自动包含-O3）
# 放在project()之后，任何目标定义之前
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================
# SIMD options (user selectable; auto-detect if none specified)
# ============================================================
option(ENABLE_AVX512 "Enable AVX-512 backend" OFF)
option(ENABLE_AVX2    "Enable AVX2 backend"     OFF)
option(ENABLE_SSE2    "Enable SSE2 backend"     OFF)

if(ENABLE_AVX512 OR ENABLE_AVX2 OR ENABLE_SSE2)
  # user explicitly chose a backend
else()
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx2"   COMPILER_SUPPORTS_AVX2)
  check_cxx_compiler_flag("-msse4.1" COMPILER_SUPPORTS_SSE41)

  if(COMPILER_SUPPORTS_AVX2)
    set(ENABLE_AVX2 ON)
  elseif(COMPILER_SUPPORTS_SSE41)
    set(ENABLE_SSE2 ON)
  else()
    message(FATAL_ERROR "Compiler does not support AVX2 or SSE4.1")
  endif()
endif()

if(ENABLE_AVX512)
    message(STATUS "SIMD backend: AVX-512")
    add_compile_options(-mavx512f -mavx512bw)
    add_compile_definitions(ENABLE_AVX512)

elseif(ENABLE_AVX2)
    message(STATUS "SIMD backend: AVX2")
    add_compile_options(-mavx2)
    add_compile_definitions(ENABLE_AVX2)

elseif(ENABLE_SSE2)
    message(STATUS "SIMD backend: SSE4.1")
    add_compile_options(-msse4.1)
    add_compile_definitions(ENABLE_SSE2)

else()
    message(FATAL_ERROR "No SIMD backend selected")
endif()

include(GNUInstallDirs)

# 设置静态库输出目录为项目二进制目录下的lib文件夹
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# 设置动态库输出目录为项目二进制目录下的lib文件夹
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# 设置可执行文件输出目录为项目二进制目录下的bin文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# # SIMD
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# 设置C++编译标志：启用所有警告和额外警告，遵循严格的ISO标准
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build all libraries as shared")
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

# 1. 查找 zlib 包
find_package(ZLIB REQUIRED)

find_package(OpenMP)

# OpenMP处理
if(OpenMP_CXX_FOUND)
    add_compile_options(-Wunknown-pragmas)
else()
    message(WARNING "OpenMP not found, compilation will continue without OpenMP support")
endif()

# 查找 src 目录下的所有 .cpp 文件
file(GLOB SOURCES "src/*.cpp" "src/*.c")

add_library(${PROJECT_NAME}
    src/align.cpp
    src/graph.cpp
    src/file_io.cpp
    src/kalloc.c
    src/kstring.c
    src/lchain.cpp
    src/mem_alloc_utils.cpp
    src/minimizer.cpp
    src/sequence.cpp
    src/utils.c)

target_link_libraries(${PROJECT_NAME} PRIVATE ZLIB::ZLIB)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

# 添加可执行文件
add_executable(${PROJECT_NAME}_bin src/main.cpp)
target_link_libraries(${PROJECT_NAME}_bin ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME}_bin PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

# 设置头文件路径
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# 安装 minipoa 库文件到库目录
# 安装库目标，并指定要导出
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# 安装 minipoa_bin 可执行文件到二进制目录
install(TARGETS ${PROJECT_NAME}_bin DESTINATION ${CMAKE_INSTALL_BINDIR})

# 安装头文件到包含目录（使用项目名作为子目录避免冲突）
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# 导出配置（可选，便于其他CMake项目使用）
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
