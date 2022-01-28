#include <cstdio>
#include "trainer.hpp"

int main() {
    using namespace std;
    using namespace trainer;

    TrainerConfig config;
    config.hiddenLayerSize = 512;
    config.sampleFactor = 1;
    config.trainDataPath = "dataset/train-images-idx3-ubyte";
    config.trainLabelPath = "dataset/train-labels-idx1-ubyte";
    config.testDataPath = "dataset/t10k-images-idx3-ubyte";
    config.testLabelPath = "dataset/t10k-labels-idx1-ubyte";

    Trainer trainer;
    trainer.init(config);
    // trainer.loadNetFrom("net-h512.txt");
    trainer.train(true, 128, 16384, 0.25);
    trainer.test(true);
    puts("Testing on training data ...");
    float trainAcc = trainer.testOnTrainData(false);
    printf("Training Data Accuracy: %.4f\n", trainAcc);

    trainer.saveNetTo("network.txt");
    puts("The network has been saved to 'network.txt'");

    return 0;
}
