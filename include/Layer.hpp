//
// Created by 15752 on 2019/5/27.
//

#ifndef UNTITLED_LAYER_HPP
#define UNTITLED_LAYER_HPP

#include <Eigen/Eigen>
#include <functional>

enum LayerType {
    linear_transformation,
    convolution
};

// TODO
enum ErrorType {
    error_type1,
    error_type2
};

class Layer {
public:
    virtual ~Layer();

    virtual Eigen::MatrixXd calculate(Eigen::MatrixXd input_data) = 0;
};

class LayerFactory {
public:
    virtual Layer *FactoryMethod(LayerType, Eigen::MatrixXd, std::function<double(double)>) = 0;

    virtual ~LayerFactory() {}
};

#endif //UNTITLED_LAYER_HPP
