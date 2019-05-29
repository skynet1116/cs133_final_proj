//
// Created by 15752 on 2019/5/28.
//

#ifndef CS133FINAL_CONVOLUTIONLAYER_HPP
#define CS133FINAL_CONVOLUTIONLAYER_HPP
#include "Layer.hpp"
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

class ConvolutionLayer : public Layer
{
    using tensor = Eigen::Tensor<double, 3>;

private:
    int input_channel;
    int output_channel;
    int padding;
    int stride;
    std::vector<tensor> kernels;

public:
    ConvolutionLayer(int in, int out, int padding, int stride, std::vector<tensor> param) : input_channel(in),
                                                                                            output_channel(out),
                                                                                            padding(padding),
                                                                                            stride(stride),
                                                                                            kernels(param){};
    ~ConvolutionLayer() = default;
    tensor calculate(tensor input)
    {
        int output_x = kernels[0].dimension(1) - input.dimension(1);
        int output_y = kernels[0].dimension(2) - input.dimension(2);
        tensor result(output_channel, output_x, output_y);

        for (int channel = 0; channel < output_channel; channel++)
        {

            tensor kernel = kernels[channel];
            int conv_window_x = input.dimension(1) - kernel.dimension(1) + 1;
            int conv_window_y = input.dimension(2) - kernel.dimension(2) + 1;
            for (int i = 0; i < conv_window_x; i++)
            {
                for (int j = 0; j < conv_window_y; j++)
                {
                    int sum=0;
                    for (int x = 0; x < kernel.dimension(1); x++)
                    {
                        for (int y = 0; x < kernel.dimension(2); y++)
                        {
                            for (int frame = 0; frame < input_channel; frame++)
                            {
                                sum += kernel(frame, x, y) * input(frame, x + i, y + j);
                            }
                        }
                        result(channel, i, j) = sum;
                    }
                }
            }
            return result;
        }
    }
};
#endif //CS133FINAL_CONVOLUTIONLAYER_HPP
