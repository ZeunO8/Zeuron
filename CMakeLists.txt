cmake_minimum_required(VERSION 3.30)
project(nnpp)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_library(nnpp
        STATIC
        src/Neuron.cpp
        src/Layer.cpp
        src/NeuralNetwork.cpp
        src/Logger.cpp
        src/Random.cpp
)

function(create_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} nnpp)
    add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_BINARY_DIR}/${TEST_NAME})
endfunction()

include(CTest)
enable_testing()
create_test(XOR tests/XOR.cpp)
create_test(AND tests/AND.cpp)
create_test(OR tests/OR.cpp)
create_test(CircleClassification tests/CircleClassification.cpp)
create_test(MultiClassClassification tests/MultiClassClassification.cpp)