#ifndef CS133_FINAL_FACTORY_HPP
#define CS133_FINAL_FACTORY_HPP
#include "ConvolutionLayer.hpp"
#include "LinearTransformationLayer.hpp"
#include "MaxPoolLayer.hpp"
class LayerFactory
{
public:
    Layer *CreateLinearLayer(Eigen::MatrixXd param, std::function<double(double)> rf)
    {
        return new LinearTransformationLayer(param, rf);
    }
    Layer *CreateConvolutionLayer(std::vector<Eigen::Tensor<double,3>> kernels)
    {
        return new ConvolutionLayer(kernels);
    }
    Layer *CreateMaxPoolLayer(std::vector<int> param, std::function<double(double)> rf)
    {
        return new MaxPoolLayer(param, rf);
    }
};
#endif //CS133_FINAL_FACTORY_HPP