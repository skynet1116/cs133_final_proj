#ifndef CS133FINAL_MAXPOOLLAYER_HPP
#define CS133FINAL_MAXPOOLLAYER_HPP

#include "Layer.hpp"

#include <vector>



class MaxPoolLayer : public Layer {
public:
    MaxPoolLayer(std::vector<int> param, std::function<double(double)> rf)
    : layer_param(param), layer_response_function(rf) {}

    double window_max(Eigen::Tensor<double,3> input_data, int i, int begin_row, int end_row, int begin_col, int end_col)
    {
        double max=0.0;
        for(int j=begin_row;j<end_row;j++){
            for(int k=begin_col;k<end_col;k++){
                if(input_data(i,j,k)>max){
                    max=input_data(i,j,k);
                }
            }
        }
        return max;
    }

    Eigen::Tensor<double,3> calculate(Eigen::Tensor<double,3> input_data)
    {
        int result_row=input_data.dimension(1)/layer_param[0];
        int result_col=input_data.dimension(2)/layer_param[1];
        Eigen::Tensor<double,3> result(input_data.dimension(0),result_row,result_col) ;

        for (int i = 0; i < result.dimension(0); i++)
        {
            for (int j = 0; j < result.dimension(1); j++)
            {
                for (int k = 0; k < result.dimension(2); k++)//may wrong!!
                {
                    result(i, j, k) = window_max(input_data,i,(j*layer_param[0]),((j+1)*layer_param[0]),(k*layer_param[1]),((k+1)*layer_param[1]));
                    result(i, j, k) = layer_response_function(result(i, j, k));
                }
            }
        }

        return result;
    }

    ~MaxPoolLayer() {}

private:
    std::vector<int> layer_param;
    std::function<double(double)> layer_response_function;
};


#endif