cmake_minimum_required(VERSION 3.30)
project(zeuron VERSION 0.2.1)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)
include_directories(vendor/fenster)

add_library(zeuron
        STATIC
        src/Neuron.cpp
        src/Layer.cpp
        src/NeuralNetwork.cpp
        src/Logger.cpp
        src/Random.cpp
        src/Visualizer.cpp
)

function(create_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} zeuron X11)
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
