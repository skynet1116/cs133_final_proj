/// \file test_conv.cpp
/// \brief A test when the load file is abour linear-transformation , convolution and max-pooling layers.
///
/// You can see the loaded file in "../data/conv_net.dat",
/// all the param is there.
/// Unfortunatly, we didn't find a proper network for this test.
/// So the result may be not ideal.
/// But we believe our library is right,
/// You can see that the dimention of every layers can match,
/// and the result of every layers are all what we want.
#include "Network.hpp"
int main()
{
    Network n;
    n.load_network("../data/conv_net.dat");
    n.open_file("../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte");
    int counter = 0;
    for (int i = 0; i < 10000; i++)
    {
        std::cout << "Sample " << i << " :" << std::endl;
        n.read_one_data();
        if (!n.run())
            counter++;
        std::cout << std::endl;
    }
    std::cout << "Error num: " << counter << std::endl;
    std::cout << "Error rate: " << counter / (double)100 << "\%" << std::endl;
    return 0;
}