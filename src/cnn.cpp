#include <pybind11/pybind11.h>
#include "network.hpp"
namespace py = pybind11;
struct Pet {
    Pet(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }

    std::string name;
};
PYBIND11_MODULE(example, m) {
    py::class_<Network>(m, "Network")
        .def(py::init<>())
        .def("read_from_board", &Network::read_from_board)
        .def("run", &Network::run);
}