#include "ConvolutionLayer.hpp"
std::vector<Eigen::Tensor<double, 3>> kernels;
Eigen::Tensor<double, 3> data(3, 3, 3);
int main()
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                data(i, j, k) = i + j + k*2;
            }
        }
    }
    for (int kernel = 0; kernel < 2; kernel++)
    {
        Eigen::Tensor<double, 3> temp(3, 2, 2);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    temp(i, j, k) = i + j + k;
                }
            }
        }
        kernels.push_back(temp);
    }
    std::cout << "data" << std::endl;
    std::cout << data << std::endl;
    std::cout << "kernels" << std::endl;
    for (auto i : kernels)
    {
        std::cout << i << std::endl;
        std::cout<<"_______"<<std::endl;
    }
    std::cout<<"cnn!"<<std::endl;
    ConvolutionLayer object(kernels,[](double x){return x;});
    std::cout<<object.calculate(data)<<std::endl;
    return 0;
}
