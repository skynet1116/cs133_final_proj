/// \ConvolutionLayer.,hpp
/// \brief implement  convolution layer of neural network
#ifndef CS133FINAL_CONVOLUTIONLAYER_HPP
#define CS133FINAL_CONVOLUTIONLAYER_HPP
#include "Layer.hpp"
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
/// \brief child class of Layer class
/// convolution layer takes a 3*3*3 tensor as input and output a 3*3*3 tensor,
/// convolution layer takes a list of  3*3*3 tensor as parameter, which are converlution kernels
class ConvolutionLayer : public Layer
{
    using tensor = Eigen::Tensor<double, 3>;

private:
    std::vector<tensor> kernels; ///<
    std::function<double(double)> response_function;

public:
    ConvolutionLayer(std::vector<tensor> param, std::function<double(double)> rf) : kernels(param), response_function(rf){};
    ~ConvolutionLayer() = default;
    Eigen::MatrixXd conv_2d(Eigen::MatrixXd input_data, Eigen::MatrixXd kernel)
    {
        int space_x = input_data.cols() - kernel.cols() + 1;
        int space_y = input_data.rows() - kernel.rows() + 1;
        Eigen::MatrixXd result(space_x, space_y);
        for (int i = 0; i < space_x; i++)
        {
            for (int j = 0; j < space_y; j++)
            {
                auto convolution_window = input_data.block(i, j, kernel.cols(), kernel.rows());
                double sum = 0;
                for (int row = 0; row < convolution_window.cols(); row++)
                {
                    for (int col = 0; col < convolution_window.rows(); col++)
                    {
                        sum += convolution_window(row, col) * kernel(row, col);
                    }
                }
                sum = response_function(sum);
                result(i, j) = sum;
            }
        }
        return result;
    }
    tensor calculate(tensor input)
    {
        int conv_window_x = input.dimension(1) - kernels[0].dimension(1) + 1;
        int conv_window_y = input.dimension(2) - kernels[0].dimension(2) + 1;
        tensor result(kernels.size(), conv_window_x, conv_window_y);
        auto tensor2matrix = [](tensor t) -> std::vector<Eigen::MatrixXd> {
            std::vector<Eigen::MatrixXd> result;
            for (int depth = 0; depth < t.dimension(0); depth++)
            {
                Eigen::MatrixXd temp(t.dimension(1), t.dimension(2));
                for (int x = 0; x < t.dimension(1); x++)
                {
                    for (int y = 0; y < t.dimension(2); y++)
                    {
                        temp(x, y) = t(depth, x, y);
                    }
                }
                result.push_back(temp);
            }
            // std::cout<<result.size()<<std::endl;
            return result;
        };
        std::vector<Eigen::MatrixXd> input_matrice = tensor2matrix(input);
        for (int i = 0; i < kernels.size(); i++)
        {
            tensor kernel = kernels[i];
            std::vector<Eigen::MatrixXd> kernel_matrice = tensor2matrix(kernel);
            Eigen::MatrixXd frame(conv_window_x, conv_window_y);
            frame.setConstant(0.0);
            for (int index = 0; index < kernel_matrice.size(); index++)
            {
                frame += conv_2d(input_matrice[index], kernel_matrice[index]);
            }
            for (int x = 0; x < conv_window_x; x++)
            {
                for (int y = 0; y < conv_window_y; y++)
                {
                    result(i, x, y) = frame(x, y);
                }
            }
        }

        return result;
    }
};
#endif //CS133FINAL_CONVOLUTIONLAYER_HPP
