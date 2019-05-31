/// \file Layer.,hpp
/// \brief Implement the base class for neural network layers.

#ifndef CS133_FINAL_LAYER_HPP
#define CS133_FINAL_LAYER_HPP

#include <Eigen/Eigen>
#include <functional>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
///\enum LayerType Indicating which kind the layer belongs to.
enum LayerType
{
    linear_transformation,
    convolution
};
///\enum ErrorType Indicating which type of error to use.
enum ErrorType
{
    error_type_abs,
    error_type_l2
};

///\brief  The base class for neural network layers.
class Layer
{
public:
    ///\brief Abstract	definition	of	common	layer's calculation.
    ///
    ///Input a 3d tensor and output a 3d tensor.
    virtual Eigen::Tensor<double, 3> calculate(Eigen::Tensor<double, 3> input_data) { return Eigen::Tensor<double, 3>(1, 2, 3); }
};

#endif //CS133_FINAL_LAYER_HPP
