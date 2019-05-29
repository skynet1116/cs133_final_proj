#ifndef CS133_FINAL_LINEAR_TRANSFORMATION_LAYER_HPP
#define CS133_FINAL_LINEAR_TRANSFORMATION_LAYER_HPP

#include "Layer.hpp"

class LinearTransformationLayer : public Layer {
public:
    LinearTransformationLayer(Eigen::MatrixXd param,
                                                     std::function<double(double)> rf)
    : layer_type(layer_type), layer_matrix(param), layer_response_function(rf) {}

    Eigen::Tensor<double, 3> calculate(Eigen::Tensor<double, 3> input_data)
    {
        double* input_data_array = input_data.data();
        int input_size = input_data.dimension(0) * input_data.dimension(1) * input_data.dimension(2);

        Eigen::RowVectorXd reshaped_data(input_size);

        for (int i = 0; i < input_size; i++) {
            reshaped_data(i) = input_data_array[i];
        }

        Eigen::MatrixXd result = reshaped_data * layer_matrix;

        for (int i = 0; i < result.rows(); i++)
        {
            for (int j = 0; j < result.cols(); j++)
            {
                result(i, j) = layer_response_function(result(i, j));
            }
        }

        Eigen::Tensor<double, 3> ret(1, result.rows(), result.cols());
        for (int i = 0; i < result.rows(); i++) {
            for (int j = 0; j < result.cols(); j++) {
                ret(0, i, j) = result(i, j);
            }
        }

        return ret;
    }
    ~LinearTransformationLayer() {}

private:
    LayerType layer_type;
    Eigen::MatrixXd layer_matrix;
    std::function<double(double)> layer_response_function;
};

#endif