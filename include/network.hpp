#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <vector>
#include "Layer.hpp"

class Network {
private:
    std::vector<Layer> m_layers;
    ErrorType m_error_type;

public:
    Network();
    ~Network();

    void parse(const std::string& filename);
};