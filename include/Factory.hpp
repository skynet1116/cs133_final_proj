#ifndef CS133_FINAL_FACTORY_HPP
#define CS133_FINAL_FACTORY_HPP
#include "ConvolutionLayer.hpp"
#include "LinearTransformation.hpp"
class LayerFactory
{
public:
    LayerType CreateLayer(LayerType type, Eigen::MatrixXd param, std::function<double(double)> rf)
    {
        switch (type)
        {
        case linear_transformation:
            return LinearTransformationLayer(param, rf);
            break;
        case convolution:
            return ConvolutionLayer(param, rf);
            break;
        default:
            break;
        }
    }
};
#endif //CS133_FINAL_FACTORY_HPP