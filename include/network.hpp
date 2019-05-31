/// \file Network.hpp
/// \brief Implementation of the whole network structure
#ifndef CS133_FINAL_NETWORK_HPP
#define CS133_FINAL_NETWORK_HPP

#include <Eigen/Eigen>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iostream>
#include <vector>
#include "Layer.hpp"
#include "Factory.hpp"

const int height = 28;
const int width = 28;

/// \brief The interface of the library
class Network
{
private:
    std::vector<Layer *> m_layers; ///< All layers
    ErrorType m_error_type; ///< Choosen error type
    Eigen::MatrixXd m_data; ///< Input data
    int label; ///< The read in correct label
    std::ifstream images; ///< Image dataset file stream
    std::ifstream labels; ///< Label dataset file stream

public:
    /// \brief Constructor
    Network();
    /// \brief Destructor
    ~Network();

    /// \brief Function to load network from given file
    /// Network definition file have the same syntax as give syntax.txt
    void load_network(const std::string &filename);
    /// \brief Function to open file stream of images and labels from given file
    /// Now only works for MNIST dataset or other dataset with the same format
    void open_file(std::string image_filename, std::string label_filename);
    /// \brief Function to load one pair of image-label from file stream
    /// Loaded data are stored in m_data and label
    void read_one_data();
    /// \brief Function to load data from 'writing board'
    /// This is used in Pybind
    void read_from_board(std::vector<int>);
    /// \brief Function to go through all stored layers
    /// Simply pass data and go through all layers in m_layers
    Eigen::MatrixXd go_through_layers();
    /// \brief Function do soft max
    Eigen::VectorXd soft_max(Eigen::MatrixXd data);
    /// \brief Function to get predicted label from output data
    int predict_label(Eigen::VectorXd soft_max_result);
    /// \brief Function to calculate error
    double error(Eigen::VectorXd output);
    /// \brief Function to run the network after all setting is done
    bool run();
    /// \brief Function to test network in Pybind
    int test();
};
Network::Network() {}

Network::~Network() {}

void Network::load_network(const std::string &filename)
{
    std::ifstream definition_file(filename);

    int layer_num;

    definition_file >> layer_num;

    int layer_type_num, error_type_num;
    int out_dim, in_dim, layer_row, layer_column;
    int response_function_num;

    std::vector<std::function<double(double)>> response_functions = {
        [=](double x) { return x; },                     //NO need for RF
        [=](double x) { return 1.0 / (1.0 + exp(-x)); }, //sigmoid
        [=](double x) { if(x<0){return 0.0;} else{return x;} },                               //RELU
        [=](double x) { if(x<0){return (0.1*x);} else{return x;} },                               //Leaky RELU
        [=](double x) { return tanh(x); }                //tanh
    };

    for (int i = 0; i < layer_num; i++)
    {
        definition_file >> layer_type_num;
        // std::cout << layer_type_num << std::endl;

        if (layer_type_num == 0)
        {
            definition_file >> layer_row >> layer_column;
            Eigen::MatrixXd layer_matrix(layer_row, layer_column);
            for (int r = 0; r < layer_row; r++)
            {
                for (int c = 0; c < layer_column; c++)
                {
                    definition_file >> layer_matrix(r, c);
                }
            }

            // read response function
            definition_file >> response_function_num;

            LayerFactory factory;
            m_layers.push_back(factory.CreateLinearLayer(layer_matrix, response_functions[response_function_num]));
        }
        else
        {
            definition_file >> out_dim >> in_dim >> layer_row >> layer_column;

            std::vector<Eigen::Tensor<double, 3>> tensors;
            for (int d = 0; d < out_dim; d++)
            {
                Eigen::Tensor<double, 3> t(in_dim, layer_row, layer_column);
                for (int i_d = 0; i_d < in_dim; i_d++)
                {
                    for (int r = 0; r < layer_row; r++)
                    {
                        for (int c = 0; c < layer_column; c++)
                        {
                            definition_file >> t(i_d, r, c);
                        }
                    }
                }
                tensors.push_back(t);
            }

            // read response function
            definition_file >> response_function_num;

            LayerFactory factory;
            std::vector<int> mp_param = {2, 2};
            m_layers.push_back(factory.CreateConvolutionLayer(tensors, response_functions[response_function_num]));
            m_layers.push_back(factory.CreateMaxPoolLayer(mp_param, [](double x) { return x; }));
        }
    }

    definition_file >> error_type_num;
    m_error_type = (ErrorType)error_type_num;
}

void Network::open_file(std::string image_filename, std::string label_filename)
{
    images.open(image_filename, std::ios::in | std::ios::binary);
    labels.open(label_filename, std::ios::in | std::ios::binary);

    char number;
    for (int i = 1; i <= 16; ++i)
    {
        images.read(&number, sizeof(char));
    }
    for (int i = 1; i <= 8; ++i)
    {
        labels.read(&number, sizeof(char));
    }
}

void Network::read_one_data()
{
    Eigen::MatrixXd d(width, height);

    char number;
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            images.read(&number, sizeof(char));
            if (number == 0)
            {
                d(i, j) = 0;
            }
            else
            {
                d(i, j) = 1;
            }
        }
    }

    m_data = d;

    // Reading label
    labels.read(&number, sizeof(char));

    label = number;
}
void Network::read_from_board(std::vector<int> input)
{
    Eigen::MatrixXd d(width, height);
    int count = 0;
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            d(i, j) = input[count];
            count++;
        }
    }
    m_data = d.transpose();
    label = -1;
}

Eigen::Tensor<double, 3> mToT(Eigen::MatrixXd input)
{
    Eigen::Tensor<double, 3> result(1, input.rows(), input.cols());

    for (int i = 0; i < input.rows(); i++)
    {
        for (int j = 0; j < input.cols(); j++)
        {
            result(0, i, j) = input(i, j);
        }
    }

    return result;
}

Eigen::MatrixXd tToM(Eigen::Tensor<double, 3> input)
{
    Eigen::MatrixXd result(input.dimension(1), input.dimension(2));

    for (int i = 0; i < input.dimension(1); i++)
    {
        for (int j = 0; j < input.dimension(2); j++)
        {
            result(i, j) = input(0, i, j);
        }
    }

    return result;
}

Eigen::MatrixXd Network::go_through_layers()
{
    std::vector<Eigen::Tensor<double, 3>> tensors(m_layers.size() + 1);
    tensors[0] = mToT(m_data);
    for (int i = 0; i < m_layers.size(); i++)
    {
        // std::cout << "Layer: " << i << std::endl;
        // std::cout << tensors[i] << std::endl;
        Layer *cur_layer = m_layers[i];

        tensors[i + 1] = cur_layer->calculate(tensors[i]);
    }

    // std::cout << tensors[tensors.size() - 1] << std::endl;
    return tToM(tensors[tensors.size() - 1]);
}

Eigen::VectorXd Network::soft_max(Eigen::MatrixXd data)
{
    Eigen::VectorXd result(data.size());
    double sum = 0.0;
    for (int i = 0; i < data.size(); i++)
    {
        sum = sum + exp(data(i));
    }
    for (int i = 0; i < data.size(); i++)
    {
        result(i) = exp(data(i)) / sum;
    }

    return result / result.sum();
}

int Network::predict_label(Eigen::VectorXd soft_max_result)
{
    int label = 0;
    double max = 0;
    for (int i = 0; i < soft_max_result.size(); i++)
    {
        if (soft_max_result(i) > max)
        {
            max = soft_max_result(i);
            label = i;
        }
    }
    return label;
}

double Network::error(Eigen::VectorXd output)
{
    double result = 0.0;
    //output.normalize();
    if (m_error_type == error_type_abs)
    {
        for (int i = 0; i < output.size(); i++)
        {
            result += std::abs(output(i) - (label == i ? 1 : 0));
        }
    }
    else
    {
        for (int i = 0; i < output.size(); i++)
        {
            result += pow((output(i) - (label == i ? 1 : 0)), 2);
        }
        result = pow(result, 0.5);
    }

    return result;
}

bool Network::run()
{
    Eigen::MatrixXd result = go_through_layers();
    std::cout << "Expect label is: " << label << std::endl;

    Eigen::VectorXd soft_max_result = soft_max(result);
    int predicted_label = predict_label(soft_max_result);

    double error_result = error(soft_max_result);
    for (int i = 0; i < soft_max_result.size(); i++)
    {
        std::cout << "Probability of label " << i << " is: " << soft_max_result(i) << std::endl;
    }

    std::cout << "Predict label is: " << predicted_label << std::endl;

    if (predicted_label != label)
    {
        std::cout << "Prediction wrong!" << std::endl;
        return false;
    }
    std::cout << "Prediction right!" << std::endl;
    return true;
}
int Network::test()
{
    Eigen::MatrixXd result = go_through_layers();
    Eigen::VectorXd soft_max_result = soft_max(result);
    int predicted_label = predict_label(soft_max_result);
    double error_result = error(soft_max_result);
    for (int i = 0; i < soft_max_result.size(); i++)
    {
        std::cout << "Probability of label " << i << " is: " << soft_max_result(i) << std::endl;
    }
    std::cout << "Predict label is: " << predicted_label << std::endl;
    return predicted_label;
}
#endif