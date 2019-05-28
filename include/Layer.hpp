//
// Created by 15752 on 2019/5/27.
//

#ifndef CS133_FINAL_LAYER_HPP
#define CS133_FINAL_LAYER_HPP

#include <Eigen/Eigen>
#include <functional>
#include <iostream>

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
    virtual Eigen::MatrixXd calculate(Eigen::MatrixXd input_data) {}
};

#endif //CS133_FINAL_LAYER_HPP
