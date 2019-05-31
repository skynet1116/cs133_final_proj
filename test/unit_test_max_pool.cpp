#include "MaxPoolLayer.hpp"
#include <functional>
#include <iostream>
#include <Eigen/Eigen>

int main()
{
    std::vector<int> param = {2, 2};
    std::function<double(double)> rf;
    rf = [=](double x) { return x + 1; };
    Eigen::Tensor<double, 3> input(2, 4, 4);
    for (int i = 0; i < input.dimension(0); i++)
    {
        for (int j = 0; j < input.dimension(1); j++)
        {
            for (int k = 0; k < input.dimension(2); k++) //may wrong!!
            {
                input(i, j, k) = (i + 1) * (j + 1) * (k + 1);
            }
        }
    }
    std::cout << param[0] << " " << param[1] << std::endl;
    std::cout << input << std::endl;
    MaxPoolLayer m(param, rf);

    std::cout << m.calculate(input) << std::endl;
    return 0;
}