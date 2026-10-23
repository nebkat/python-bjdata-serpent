cmake_minimum_required(VERSION 3.21)

project(bjdata_serpent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# Both libraries are submodules, so a build from an sdist fetches nothing. serpent takes the
# cpp-unaligned target that is already there rather than cloning its own.
add_subdirectory(extern/cpp-unaligned EXCLUDE_FROM_ALL)
set(SERPENT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
# Żmij, serpent's real-number writer (only to_json uses it here), does not compile with Apple
# Clang for x86_64; std::to_chars writes the same text there.
if(APPLE AND (CMAKE_OSX_ARCHITECTURES MATCHES "x86_64" OR (NOT CMAKE_OSX_ARCHITECTURES AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")))
    set(SERPENT_USE_ZMIJ OFF CACHE BOOL "" FORCE)
endif()
add_subdirectory(extern/cpp-serpent EXCLUDE_FROM_ALL)

nanobind_add_module(_core NB_STATIC
    src/ext/common.cpp
    src/ext/decode.cpp
    src/ext/encode.cpp
    src/ext/module.cpp)
target_link_libraries(_core PRIVATE serpent::serpent)

# nanobind links a macOS extension against a list of the interpreter's symbols, and the one
# that grows a bytes object in place - documented, and stable since Python 2 - is not on it.
if(APPLE)
    target_link_options(_core PRIVATE "LINKER:-U,__PyBytes_Resize")
endif()

install(TARGETS _core LIBRARY DESTINATION bjdata_serpent)
