/// \file unit_test_max_pool.cpp
/// \brief Unit test whether the max_pool layer is right.
#include "MaxPoolLayer.hpp"
#include <functional>
#include <iostream>
#include <Eigen/Eigen>

int main()
{   
    ///\code
    ///std::vector<int> param = {2, 2};
    ///\endcode
    ///Set the pooling block size is 2*2 
    ///\code
    ///std::function<double(double)> rf;
    ///\endcode
    ///Set the response function of this layer which is f(x)=x+1
    ///\code
    ///Eigen::Tensor<double, 3> input(2, 4, 4);
    ///\endcode
    ///The input data which will generate later
    ///\code
    ///Eigen::Tensor<double, 3> output(2, 2, 2);
    ///\endcode
    ///The output data of this layer
    std::vector<int> param = {2, 2};
    std::cout << "The pooling block size : " << param[0] << " " << param[1] << std::endl;
    std::function<double(double)> rf;
    rf = [=](double x) { return x + 1; };
    MaxPoolLayer m(param, rf);
    Eigen::Tensor<double, 3> input(2, 4, 4);
    std::cout << "The input data : " << std::endl;
    for (int i = 0; i < input.dimension(0); i++)
    {
        std::cout << " [ " ; 
        for (int j = 0; j < input.dimension(1); j++)
        {
            std::cout << " [ " ; 
            for (int k = 0; k < input.dimension(2); k++)
            {
                input(i, j, k) = (i + 1) * (j + 1) * (k + 1);
                std::cout << input(i,j,k) << " " ;
            }
            std::cout << " ] " ; 
        }
        std::cout << " ] " ; 
    }
    Eigen::Tensor<double, 3> output(2, 2, 2);
    output = m.calculate(input);
    std::cout<<std::endl;
    std::cout << "The output data : " << std::endl;
    for (int i = 0; i < output.dimension(0); i++)
    {
        std::cout << " [ " ; 
        for (int j = 0; j < output.dimension(1); j++)
        {
            std::cout << " [ " ; 
            for (int k = 0; k < output.dimension(2); k++)
            {
                output(i, j, k) = (i + 1) * (j + 1) * (k + 1);
                std::cout << output(i,j,k) << " " ;
            }
            std::cout << " ] " ; 
        }
        std::cout << " ] " ; 
    }
    return 0;
} 
