#include "network.hpp"

void Network::parse(const std::string& filename) {
    std::ifstream definition_file(filename);

    int layer_num;

    definition_file >> layer_num;

    int layer_type_num, error_type_num;
    int layer_row, layer_column;

    for (int i = 0; i < layer_num; i++) {
        definition_file >> layer_type_num >> layer_column >> layer_row;
        Eigen::MatrixXd layer_matrix(layer_row, layer_column);

        // read matrix
        for (int r = 0; r < layer_row; r++) {
            for (int c = 0; c < layer_column ; c++) {
                definition_file >> layer_matrix(r, c);
            }
        }

        // read response function
        double mid_point, leftside_value, rightside_value;
        definition_file >> mid_point >> leftside_value >> rightside_value;

        auto layer_response_funtion = [](auto mid_point, auto leftside_value, auto rightside_value) {
            return [=](double x) { return x < mid_point ? leftside_value : rightside_value; };
        };

        // m_layers.push_back(
        //     LayerFactory((LayerType)layer_type_num, layer_matrix, layer_response_funtion(mid_point, leftside_value, rightside_value)));
    }

    definition_file >> error_type_num;
    m_error_type = (ErrorType)error_type_num;
}