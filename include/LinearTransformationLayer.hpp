#include "Layer.hpp"

class LinearTransformationLayer : public Layer {
public:
    LinearTransformationLayer(LayerType layer_type, Eigen::MatrixXd param, std::function<double(double)> rf);
    Eigen::MatrixXd calculate(Eigen::MatrixXd input_data);
    ~LinearTransformationLayer();

private:
    LayerType layer_type;
    Eigen::MatrixXd layer_matrix;
    std::function<double(double)> layer_response_function;
};