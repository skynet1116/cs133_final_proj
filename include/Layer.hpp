//
// Created by 15752 on 2019/5/27.
//

#ifndef CS133_FINAL_LAYER_HPP
#define CS133_FINAL_LAYER_HPP

#include <Eigen/Eigen>
#include <functional>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

enum LayerType
{
    linear_transformation,
    convolution
};

enum ErrorType
{
    error_type_abs,
    error_type_l2
};


class Layer
{
public:
	virtual Eigen::Tensor<double, 3> calculate(Eigen::Tensor<double, 3> input_data) { return Eigen::Tensor<double,3>(1, 2, 3); }
};

#endif //CS133_FINAL_LAYER_HPP
