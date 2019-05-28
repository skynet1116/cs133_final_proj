#include "network.hpp"

int main() {
    Network n;
    n.load_network("../data/model-neural-network.dat");
    n.open_file("../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte");
    int counter = 0;
    for (int i = 0; i < 10000; i++) {
        n.read_one_data();
        if (!n.run()) counter++;
    }
    std::cout << "Error num: " << counter << std::endl;
    return 0;
}