cmake_minimum_required(VERSION 3.30)
project(zeuron VERSION 0.3.1)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)
include_directories(vendor/fenster)
include_directories(vendor/ByteStream/include)

add_library(zeuron
        STATIC
        src/Neuron.cpp
        src/Layer.cpp
        src/NeuralNetwork.cpp
        src/Logger.cpp
        src/Random.cpp
        src/Visualizer.cpp
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (WIN32)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
elseif (APPLE OR UNIX)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
endif()

if(UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
    target_link_libraries(zeuron ${X11_LIBRARIES})
    include_directories(${X11_INCLUDE_DIR})
endif()

add_subdirectory(vendor/ByteStream)

target_link_libraries(zeuron ByteStream)

function(create_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} zeuron)
    if(UNIX AND NOT APPLE)
        target_link_libraries(${TEST_NAME} ${X11_LIBRARIES})
    endif()
    add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_BINARY_DIR}/${TEST_NAME})
endfunction()

include(CTest)
enable_testing()
create_test(XOR tests/XOR.cpp)
create_test(AND tests/AND.cpp)
create_test(OR tests/OR.cpp)
create_test(CircleClassification tests/CircleClassification.cpp)
create_test(MultiClassClassification tests/MultiClassClassification.cpp)
create_test(Sinusoidal tests/Sinusoidal.cpp)
