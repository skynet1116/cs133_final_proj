/// \file Factory.hpp
/// \brief Initializes	concrete layers	of	the	network.
#ifndef CS133_FINAL_FACTORY_HPP
#define CS133_FINAL_FACTORY_HPP
#include "ConvolutionLayer.hpp"
#include "LinearTransformationLayer.hpp"
#include "MaxPoolLayer.hpp"
/// \brief using factory method to initializes	concrete layers	of	the	network.
class LayerFactory
{
public:
    ///\param[in] param linear transform parameter
    ///\param[in] rf response function
    ///\return a new linear layer
    Layer *CreateLinearLayer(Eigen::MatrixXd param, std::function<double(double)> rf)
    {
        return new LinearTransformationLayer(param, rf);
    }
    ///\param[in] kernels convolution kernels
    ///\param[in] rf response function
    ///\return a new convolution layer
    Layer *CreateConvolutionLayer(std::vector<Eigen::Tensor<double, 3>> kernels, std::function<double(double)> rf)
    {
        return new ConvolutionLayer(kernels, rf);
    }
    ///\param[in] param sample size
    ///\param[in] rf response function
    ///\return a new max-pooling layer
    Layer *CreateMaxPoolLayer(std::vector<int> param, std::function<double(double)> rf)
    {
        return new MaxPoolLayer(param, rf);
    }
};
#endif //CS133_FINAL_FACTORY_HPP