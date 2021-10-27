cmake_minimum_required(VERSION 3.0)

project("viy" LANGUAGES CXX)

# find all source files
file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# add include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# build
add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)
target_include_directories(${PROJECT_NAME} PRIVATE include)

# opencv setup
find_package(OpenCV REQUIRED) # PATHS /usr/local/share/OpenCV NO_DEFAULT_PATH
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})





