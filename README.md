# nn++

A simple NeuralNetwork library written in C++

Uses CMake for it's build system and comes with some included tests

### Building

Either use your preferred IDE of choice, or run the following commands to build

```bash
cmake -B build .
cmake --build build
```

### Testing

```bash
ctest --test-dir build --rerun-failed --output-on-failure -C Debug
```