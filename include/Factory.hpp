#ifndef CS133_FINAL_FACTORY_HPP
#define CS133_FINAL_FACTORY_HPP
#include "ConvolutionLayer.hpp"
#include "LinearTransformationLayer.hpp"
class LayerFactory
{
public:
    Layer *CreateLayer(LayerType type, Eigen::MatrixXd param, std::function<double(double)> rf)
    {
        switch (type)
        {
        case linear_transformation:
            return new LinearTransformationLayer(param, rf);
            break;
        case convolution:
            //return new ConvolutionLayer(param, rf);
            break;
        default:
            break;
        }
    }
};
#endif //CS133_FINAL_FACTORY_HPP