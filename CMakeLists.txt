cmake_minimum_required(VERSION 3.30)
project(zeuron VERSION 0.3.3)

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
        src/Timer.cpp
)

if(UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
    target_link_libraries(zeuron ${X11_LIBRARIES})
    include_directories(${X11_INCLUDE_DIR})
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(zeuron OpenMP::OpenMP_CXX)
endif()

add_subdirectory(vendor/ByteStream)

target_link_libraries(zeuron ByteStream)

if(WIN32)
    set(TEST_EXT ".exe")
endif()

function(create_test TEST_NAME TEST_SOURCE TEST_ARGS)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} zeuron)
    if(UNIX AND NOT APPLE)
        target_link_libraries(${TEST_NAME} ${X11_LIBRARIES})
    endif()
    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE_DIR:${TEST_NAME}>/${TEST_NAME}${TEST_EXT} ${TEST_ARGS})
endfunction()

include(CTest)
enable_testing()
create_test(XOR tests/XOR.cpp "")
create_test(AND tests/AND.cpp "")
create_test(OR tests/OR.cpp "")
create_test(CircleClassification tests/CircleClassification.cpp "")
create_test(MultiClassClassification tests/MultiClassClassification.cpp "")
create_test(Sinusoidal tests/Sinusoidal.cpp force-train)
