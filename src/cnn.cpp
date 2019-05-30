#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include "network.hpp"
namespace py = pybind11;

PYBIND11_MODULE(example, m) {
    py::class_<Network>(m, "Network")
        .def(py::init<>())
        .def("read_from_board", &Network::read_from_board)
        .def("load_network", &Network::load_network)
        .def("test", &Network::test);
}