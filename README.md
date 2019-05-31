# cs133_final_proj
This repo contains a header-only library to load and test a pre-trained neural network.

## File List
File/Folder name  | Usage
------------- | -------------
cmake/         | Contains cmake helper files
data/          | Contains data file (e.g. Pre-trained network, dataset)
include/       | Contains all header files
src/           | Contains file for pybind
test/          | Contains unit tests
CMakeLists.txt | CMakeLists file
syntax.txt     | The syntax of pre-trained network definition file


## Usage
This library can load pre-trained neural network, load dataset, and run test on dataset (tested on MNIST dataset).

## Dependencies
Eigen, Pybind11

## How to build
Under project root directory

```
git clone https://github.com/pybind/pybind11.git
mkdir build
cd build
cmake ..
make
```

Executable should be generated in `build` folder.
