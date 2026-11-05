cmake_minimum_required(VERSION 3.20)
project(cpp_nn LANGUAGES CXX)

include(FetchContent)

find_package(nlohmann_json 3.11.3 QUIET)
if(NOT nlohmann_json_FOUND)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

add_executable(cpp-nn
    src/main.cpp
    src/storage.cpp
    src/tensor.cpp
    src/util.cpp
)

target_compile_features(cpp-nn PRIVATE cxx_std_23)
target_include_directories(cpp-nn PRIVATE src)
target_link_libraries(cpp-nn PRIVATE nlohmann_json::nlohmann_json)
