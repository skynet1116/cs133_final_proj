#include "network.hpp"

Network::Network() {}

Network::~Network() {}

void Network::load_network(const std::string& filename) {
    std::ifstream definition_file(filename);

    int layer_num;

    definition_file >> layer_num;

    int layer_type_num, error_type_num;
    int layer_row, layer_column;

    for (int i = 0; i < layer_num; i++) {
        definition_file >> layer_type_num >> layer_row >> layer_column;
        Eigen::MatrixXd layer_matrix(layer_row, layer_column);

        // read matrix
        for (int r = 0; r < layer_row; r++) {
            for (int c = 0; c < layer_column; c++) {
                definition_file >> layer_matrix(r, c);
            }
        }

        // read response function
        double mid_point, leftside_value, rightside_value;
        definition_file >> mid_point >> leftside_value >> rightside_value;

        auto layer_response_funtion = [](auto mid_point, auto leftside_value, auto rightside_value) {
            return [=](double x) { return x < mid_point ? leftside_value * x : rightside_value * x; };
        };
        LayerFactory factory;
        m_layers.push_back(factory.CreateLayer((LayerType)layer_type_num, layer_matrix,
                                               layer_response_funtion(mid_point, leftside_value, rightside_value)));
    }

    definition_file >> error_type_num;
    m_error_type = (ErrorType)error_type_num;
}

void Network::open_file(std::string image_filename, std::string label_filename) {
    images.open(image_filename, std::ios::in | std::ios::binary);
    labels.open(label_filename, std::ios::in | std::ios::binary);

    char number;
    for (int i = 1; i <= 16; ++i) {
        images.read(&number, sizeof(char));
    }
    for (int i = 1; i <= 8; ++i) {
        labels.read(&number, sizeof(char));
    }
}

void Network::read_one_data() {
    Eigen::MatrixXd d(width, height);

    char number;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            images.read(&number, sizeof(char));
            if (number == 0) {
                d(i, j) = 0;
            } else {
                d(i, j) = 1;
            }
        }
    }

    m_data = d;

    // Reading label
    labels.read(&number, sizeof(char));

    label = number;
}

Eigen::MatrixXd Network::go_through_layers() {
    std::vector<Eigen::MatrixXd> matrices(m_layers.size() + 1);
    matrices[0] = m_data;
    for (int i = 0; i < m_layers.size(); i++) {
        Layer* cur_layer = m_layers[i];

        matrices[i + 1] = cur_layer->calculate(matrices[i]);
    }

    return matrices[matrices.size() - 1];
}

Eigen::VectorXd Network::soft_max(Eigen::MatrixXd data) {
    Eigen::VectorXd result(data.size());
    double sum = 0.0;
    for (int i = 0; i < data.size(); i++) {
        sum = sum + exp(data(i));
    }
    for (int i = 0; i < data.size(); i++) {
        result(i) = exp(data(i)) / sum;
    }
    return result;
}

int Network::predict_label(Eigen::VectorXd soft_max_result) {
    int label = 0;
    int max = 0;
    for (int i = 0; i < soft_max_result.size(); i++) {
        if (soft_max_result(i) > max) {
            max = soft_max_result(i);
            label = i;
        }
    }

    return label;
}

double Network::error(Eigen::VectorXd output) {
    double result = 0.0;
    output.normalize();
    if (m_error_type == error_type_abs) {
        for (int i = 0; i < output.size(); i++) {
            result += abs(output(i) - label == i ? 1 : 0);
        }
    } else {
        for (int i = 0; i < output.size(); i++) {
            result += pow((output(i) - label == i ? 1 : 0), 2);
        }
        result = pow(result, 0.5);
    }

    return result;
}

bool Network::run() {
    Eigen::MatrixXd result = go_through_layers();
    std::cout << "Expect label is: " << label << std::endl;

    Eigen::VectorXd soft_max_result = soft_max(result);
    int predicted_label = predict_label(soft_max_result);
    std::cout << "Predict label is: " << predicted_label << std::endl;
    double error_result = error(soft_max_result);
    std::cout << "Error is: " << error_result << std::endl;

    if (predicted_label != label) return false;
    return true;
}