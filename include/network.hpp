#include <Eigen/Eigen>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iostream>
#include <vector>
#include "Layer.hpp"
#include "Factory.hpp"

const int height = 28;
const int width = 28;

class Network
{
private:
    std::vector<Layer *> m_layers;
    ErrorType m_error_type;
    Eigen::MatrixXd m_data;
    int label;
    std::ifstream images;
    std::ifstream labels;

public:
    Network();
    ~Network();

    void load_network(const std::string &filename);
    void open_file(std::string image_filename, std::string label_filename);
    void read_one_data();
    void read_from_board();
    Eigen::MatrixXd go_through_layers();
    Eigen::VectorXd soft_max(Eigen::MatrixXd data);
    int predict_label(Eigen::VectorXd soft_max_result);
    double error(Eigen::VectorXd output);
    bool run();
};