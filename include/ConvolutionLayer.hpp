/// \ConvolutionLayer.,hpp
/// \brief implement  convolution layer of neural network
#ifndef CS133FINAL_CONVOLUTIONLAYER_HPP
#define CS133FINAL_CONVOLUTIONLAYER_HPP
#include "Layer.hpp"
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
/// \brief child class of Layer class
/// convolution layer takes a 3d tensor as input and output a 3d tensor,
/// convolution layer takes a list of  3d tensor as parameter, which are convolution kernels
class ConvolutionLayer : public Layer
{
    using tensor = Eigen::Tensor<double, 3>;

private:
    std::vector<tensor> kernels;                     ///<convolution kernels passed in as parameter
    std::function<double(double)> response_function; ///<lambda functions

public:
    /// \brief constructor of convolution layer
    ///\param[in] param convolution kernels
    ///\param[in] param rf several most commonly used response functions defined in Network.hpp
    ConvolutionLayer(std::vector<tensor> param, std::function<double(double)> rf) : kernels(param), response_function(rf){};
    /// \brief nothing is alloc on heap in constructor, so nothing need to manually delete in destructor
    ~ConvolutionLayer() = default;
    /// \brief compute convolution on 2d matrix
    ///
    /// this function is invoked in 3d convolution by divilding 3d tensor to vector of 2d matrix
    ///\param[in] input_data data matrix which convolution will be applyed
    ///\param[in] kernel a 2d convolution kernel
    Eigen::MatrixXd conv_2d(Eigen::MatrixXd input_data, Eigen::MatrixXd kernel)
    {
        int space_x = input_data.cols() - kernel.cols() + 1; ///<output matrix size.x after convolution
        int space_y = input_data.rows() - kernel.rows() + 1; ///<output matrix size.y after convolution
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
                sum = response_function(sum); ///< apply response function for every value on the output matrix
                result(i, j) = sum;
            }
        }
        return result;
    }
    ///\brief concrete implementation of abstruct definition in base class: Layer
    ///\param[in] input 3d tensor from origin data or last layer of the neural network
    tensor calculate(tensor input)
    {
        int conv_window_x = input.dimension(1) - kernels[0].dimension(1) + 1;
        int conv_window_y = input.dimension(2) - kernels[0].dimension(2) + 1;
        tensor result(kernels.size(), conv_window_x, conv_window_y);
        ///\brief a lambda function that turn a 3d tensor into a vector of 2d matrices
        ///
        /// both input data and convolution kernels are needed to be turned,
        /// using a lambda function can reuse the code  for the entire convolutiion layer.
        /// a 3d tensor has 3 dimentions which we let their size to be d0,d1,d2
        /// in our implimentation we take d0 as the depth of tensor
        /// so there will be d0 matrices and their width and height are all d1 and d2
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
                frame += conv_2d(input_matrice[index], kernel_matrice[index]); ///< for a multi-channel input data and multi-channel convolution kernel, add up all result value from each channel
            }
            for (int x = 0; x < conv_window_x; x++)
            {
                for (int y = 0; y < conv_window_y; y++)
                {
                    result(i, x, y) = frame(x, y); ///< add the result matrix of one kernel convolution as one page of the final output tensor
                }
            }
        }

        return result;
    }
};
#endif //CS133FINAL_CONVOLUTIONLAYER_HPP
