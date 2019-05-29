#include "ConvolutionLayer.hpp"
int main()
{
    Eigen::Tensor<double,3> data(5,5,5);
    data.setConstant(0.25);
    std::vector<Eigen::Tensor<double,3>> kernels;
    for (int i=0;i<6;i++){
        Eigen::Tensor<double,3> temp(5,3,3);
        temp.setConstant(0.1*i);
        kernels.push_back(temp);
    }
    ConvolutionLayer conv(5,6,0,1,kernels);

    std::cout<<conv.calculate(data)<<std::endl;
}