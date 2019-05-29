#ifndef CS133_FINAL_FACTORY_HPP
#define CS133_FINAL_FACTORY_HPP
#include "ConvolutionLayer.hpp"
#include "LinearTransformationLayer.hpp"
class LayerFactory
{
public:
    Layer *CreateLinearLayer(Eigen::MatrixXd param, std::function<double(double)> rf)
    {
        return new LinearTransformationLayer(param, rf);
    }
};
#endif //CS133_FINAL_FACTORY_HPP