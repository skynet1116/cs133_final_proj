//
// Created by 15752 on 2019/5/27.
//

#ifndef CS133_FINAL_LAYER_HPP
#define CS133_FINAL_LAYER_HPP

#include <Eigen/Eigen>
#include <functional>

enum LayerType
{
    linear_transformation,
    convolution
};

// TODO
enum ErrorType
{
    error_type1,
    error_type2
};

class Layer
{
public:
    virtual Eigen::MatrixXd calculate(Eigen::MatrixXd input_data) {}
};

#endif //CS133_FINAL_LAYER_HPP
