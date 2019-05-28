//
// Created by 15752 on 2019/5/28.
//

#ifndef CS133FINAL_CONVOLUTIONLAYER_HPP
#define CS133FINAL_CONVOLUTIONLAYER_HPP
#include "Layer.hpp"
#include <iostream>

class ConvolutionLayer : public Layer
{
private:
    Eigen::MatrixXd layer_matrix;
    std::function<double(double)> layer_response_function;

public:
    ConvolutionLayer(Eigen::MatrixXd param, std::function<double(double)> rf) : layer_matrix(param),
                                                                                layer_response_function(rf){};
    ~ConvolutionLayer() = default;
    Eigen::MatrixXd calculate(Eigen::MatrixXd input_data)
    {
        std::cout << "conv" << std::endl;
        int space_x = input_data.cols() - layer_matrix.cols() + 1;
        int space_y = input_data.rows() - layer_matrix.rows() + 1;
        Eigen::MatrixXd result(space_x, space_y);
        for (int i = 0; i < space_x; i++)
        {
            for (int j = 0; j < space_y; j++)
            {
                auto convolution_window = input_data.block(i, j, layer_matrix.cols(), layer_matrix.rows());
                double sum = 0;
                for (int row = 0; row < convolution_window.cols(); row++)
                {
                    for (int col = 0; col < convolution_window.rows(); col++)
                    {
                        sum += convolution_window(row, col) * layer_matrix(row, col);
                    }
                }
                std::cout << sum << std::endl;
                result(i, j) = sum;
            }
        }
        response_function(result);
        return result;
    }
    void response_function(Eigen::MatrixXd &data)
    {
        for (int i = 0; i < data.rows(); i++)
        {
            for (int j = 0; j < data.cols(); j++)
            {
                data(i, j) = layer_response_function(data(i, j));
            }
        }
    }
};

//  class ConvolutionLayerCreator : public LayerFactory

//  {
//  public:
//      ConvolutionLayer FactoryMethod(LayerType type, Eigen::MatrixXd param, std::function<double(double)> rf)
//      {
//          return ConvolutionLayer(type, param, rf);
//      }
//  };
#endif //CS133FINAL_CONVOLUTIONLAYER_HPP
