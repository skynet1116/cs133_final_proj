#include "network.hpp"

int main()
{
    Network n;
    n.load_network("../data/model-neural-network.dat");
    n.open_file("../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte");
    int counter = 0;
    for (int i = 0; i < 10000; i++) {
        std::cout << "Sample " << i << " :" << std::endl;
        n.read_one_data();
        if (!n.run()) counter++;
        std::cout << std::endl;
    }
    std::cout << "Error num: " << counter << std::endl;
    std::cout << "Error rate: " << counter/(double)100 << "\%" << std::endl;
    return 0;
}