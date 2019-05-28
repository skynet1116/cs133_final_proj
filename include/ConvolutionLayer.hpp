//
// Created by 15752 on 2019/5/28.
//

#ifndef CS133FINAL_CONVOLUTIONLAYER_HPP
#define CS133FINAL_CONVOLUTIONLAYER_HPP
#include <Layer.hpp>
class ConvolutionLayer : public Layer
{
private:
    LayerType layer_type;
    Eigen::MatrixXd layer_matrix;
    std::function<double(double)> layer_response_function;

public:
    ConvolutionLayer(LayerType type, Eigen::MatrixXd param, std::function<double(double)> rf) : layer_type(type),
                                                                                                layer_matrix(param),
                                                                                                layer_response_function(rf){};

    Eigen::MatrixXd calculate(Eigen::MatrixXd input_data)
    {
        int space_x = input_data.cols() - layer_matrix.cols() + 1;
        int space_y = input_data.rows() - layer_matrix.rows() + 1;
        Eigen::MatrixXd result(space_x, space_y);
        for (int i = 0; i < space_x; i++)
        {
            for (int j = 0; i < space_y; j++)
            {
                auto convolution_window = input_data.block(i, j, layer_matrix.cols(), layer_matrix.rows());
                double sum = 0;
                for (int row = 0; i < convolution_window.cols(); i++)
                {
                    for (int col = 0; i < convolution_window.rows(); i++)
                    {
                        sum += convolution_window(row, col) * layer_matrix(row, col);
                    }
                }
                result << sum;
            }
        }
        return result;
    }
};

class ConvolutionLayerCreator : public LayerFactory
{
public:
    Layer *FactoryMethod(LayerType type, Eigen::MatrixXd param, std::function<double(double)> rf)
    {
        return ConvolutionLayer(type, param, rf);
    }
};

#endif //CS133FINAL_CONVOLUTIONLAYER_HPP
