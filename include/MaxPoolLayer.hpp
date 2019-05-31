/// \file MaxPoolLayer.hpp
/// \brief implement Max-Pool layer of neural network.
#ifndef CS133FINAL_MAXPOOLLAYER_HPP
#define CS133FINAL_MAXPOOLLAYER_HPP
#include "Layer.hpp"
#include <vector>
/// \brief Child class of Layer class.
///
/// Max-Pool layer takes a 3d tensor as input and output a 3d tensor,
/// it takes a vector of two int as parameter, which is the size of the pooling block.
class MaxPoolLayer : public Layer {
public:
    /// \brief Constructor of Max-Pool layer.
    ///\param[in] param a vector of two int representing the size of pooling block
    ///\param[in] rf several most commonly used response functions defined in Network.hpp
    MaxPoolLayer(std::vector<int> param, std::function<double(double)> rf)
    : layer_param(param), layer_response_function(rf) {}
    /// \brief Nothing is alloc on heap in constructor, so nothing need to be manually deleted in destructor.
    ~MaxPoolLayer() {}
    /// \brief Do max-pool on one layer of the input data.
    ///
    /// This function is compute the max in the block of one layer of the input data.
    ///\param[in] input_data data matrix which max-pooling will be applyed
    ///\param[in] the layer i of input data which max-pooling will be applyed
    ///\param[in] the begin_row of the pooling block
    ///\param[in] the end_row of the pooling block
    ///\param[in] the begin_col of the pooling block
    ///\param[in] the end_col of the pooling block
    ///\return the max in the pooling block of the i-th layer of input data.
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
    /// \brief Do max-pool on the input data.
    ///
    /// This function is to do max-pool to every layers of the input data.
    ///\param[in] input_data data matrix which max-pooling will be applyed
    ///\return the max-pooling result 3d tensor
    ///
    ///\code
    ///int result_row=input_data.dimension(1)/layer_param[0];
    ///\endcode
    ///output matrix size after convolution
    ///\code
    ///int result_col=input_data.dimension(2)/layer_param[1];
    ///\endcode
    ///output matrix size after convolution
    ///\code
    ///result(i, j, k) = layer_response_function(result(i, j, k));
    ///\endcode
    /// apply response function for every value on the output matrix
    Eigen::Tensor<double,3> calculate(Eigen::Tensor<double,3> input_data)
    {
        int result_row=input_data.dimension(1)/layer_param[0];
        int result_col=input_data.dimension(2)/layer_param[1];
        Eigen::Tensor<double,3> result(input_data.dimension(0),result_row,result_col) ;

        for (int i = 0; i < result.dimension(0); i++)
        {
            for (int j = 0; j < result.dimension(1); j++)
            {
                for (int k = 0; k < result.dimension(2); k++)
                {
                    result(i, j, k) = window_max(input_data,i,(j*layer_param[0]),((j+1)*layer_param[0]),(k*layer_param[1]),((k+1)*layer_param[1]));
                    result(i, j, k) = layer_response_function(result(i, j, k));
                }
            }
        }

        return result;
    }


private:
    std::vector<int> layer_param;                         ///<a vector of two int representing the size of pooling block passed in as parameter.
    std::function<double(double)> layer_response_function;///<lambda functions.
};


#endif //CS133FINAL_MAXPOOLLAYER_HPP