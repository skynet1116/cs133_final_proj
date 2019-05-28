#include "LinearTransformationLayer.hpp"
#include <iostream>
LinearTransformationLayer::LinearTransformationLayer( Eigen::MatrixXd param,
                                                     std::function<double(double)> rf)
    : layer_type(layer_type), layer_matrix(param), layer_response_function(rf) {}

LinearTransformationLayer::~LinearTransformationLayer() {}

Eigen::MatrixXd LinearTransformationLayer::calculate(Eigen::MatrixXd input_data) {
    Eigen::Map<Eigen::RowVectorXd> reshaped_data(input_data.data(), input_data.size());

    Eigen::MatrixXd result = reshaped_data * layer_matrix;

    for (int i = 0; i < result.rows(); i++)
    {
        for (int j = 0; j < result.cols(); j++)
        {
            result(i, j) = layer_response_function(result(i, j));
        }
    }

    return result;
}

