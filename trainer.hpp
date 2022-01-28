#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <cstdio>
#include <random>
#include "net.hpp"
#include "loader.hpp"

namespace trainer {

using namespace std;
using namespace loader;
using namespace net;

struct TrainerConfig {
    const char* trainDataPath;
    const char* trainLabelPath;
    const char* testDataPath;
    const char* testLabelPath;
    int hiddenLayerSize;
    int sampleFactor;
};

class Trainer {
public:
    void init(TrainerConfig config) {
        // Initialize random generator
        random_device rd;
        randGen.seed(rd());

        // Load dataset
        trainLoader.load(config.trainDataPath, config.trainLabelPath, config.sampleFactor);
        testLoader.load(config.testDataPath, config.testLabelPath, config.sampleFactor);

        // Get dataset info
        auto& rawTrainData = trainLoader.getDataSet();
        auto& rawTrainLabels = trainLoader.getLabels();

        uint8_t minLabel = rawTrainLabels[0];
        uint8_t maxLabel = rawTrainLabels[0];
        for (auto l: rawTrainLabels) {
            if (l < minLabel)
                minLabel = l;
            else if (l > maxLabel)
                maxLabel = l;
        }

        labelOffset = minLabel;

        // Initialize network
        nConf.nInput = rawTrainData[0].size();
        nConf.nHidden = config.hiddenLayerSize;
        nConf.nOutput = maxLabel - minLabel + 1;
        net.init(nConf);
        trainBuf.init(nConf);

        printf("Dataset\n");
        printf("  Number of vectors: %lu\n", rawTrainLabels.size());
        printf("  Input size: %d\n", nConf.nInput);
        printf("  Label range: %d-%d\n", minLabel, maxLabel);
        printf("Configured:\n");
        printf("  Hidden size: %d\n", nConf.nHidden);
    }

    void train(bool verbose, int batchSize, int batchNum, float startRate) {
        auto& rawData = trainLoader.getDataSet();      
        auto& rawLabels = trainLoader.getLabels();
        int dataSize = rawLabels.size();

        Vector input;
        input.resize(nConf.nInput);
        int sample = 0;

        for (int t = 0; t < batchNum; ++t) {
            float_t rate = startRate / (1 + log1p(t));
            int accTime = 0;
            uint8_t accLabel, predLabel;
            trainBuf.reset();

            for (int i = 0; i < batchSize; ++i) {
                // randomly pick a sample
                sample = randGen() % dataSize;

                rawToInput(rawData[sample], input);
                accLabel = rawLabels[sample] - labelOffset;

                net.updateTrainBuffer(input, accLabel, trainBuf);

                auto output = trainBuf.getOutput();
                output.maxCoeff(&predLabel);
                if (predLabel == accLabel) ++accTime;
            }

            // Output message
            if (verbose) {
                printf("Batch: %05d, Acc: %.3f\n", t, static_cast<float>(accTime) / batchSize);
            }

            // Update network
            net.applyTrainBuffer(trainBuf, rate / batchSize);
        }
        if (verbose) putchar('\n');
    }

    float test(bool verbose) const {
        return testOn(verbose, testLoader.getDataSet(), testLoader.getLabels());
    }

    float testOnTrainData(bool verbose) const {
        return testOn(verbose, trainLoader.getDataSet(), trainLoader.getLabels());
    }

    void saveNetTo(const char* path) const { net.saveTo(path); }
    void loadNetFrom(const char *path) {
        NetConfig config = net.loadFrom(path);
        nConf = config;
    }

private:
    void rawToInput(const vector<uint8_t>& raw, Vector& in) const {
        for (int i = 0; i < nConf.nInput; ++i) {
            in[i] = static_cast<float_t>(raw[i]) / (1 << 8);
        }
    }

    float testOn(bool verbose, const vector<vector<uint8_t>>& rawData, const vector<uint8_t>& rawLabels) const {
        int dataSize = rawLabels.size();

        Vector input;
        input.resize(nConf.nInput);
        uint8_t accLabel, predLabel;
        int accTime = 0;

        for (int i = 0; i < dataSize; ++i) {
            rawToInput(rawData[i], input);
            accLabel = rawLabels[i] - labelOffset;

            Vector predProbs = net.predict(input);
            predProbs.maxCoeff(&predLabel);

            if (accLabel == predLabel) ++accTime;

            if (verbose) {
                printf("Test %d : expect %d, ", i, accLabel + labelOffset);
                if (accLabel == predLabel) {
                    printf("correct with %.2f\n", predProbs[accLabel]);
                } else {
                    printf("get %d with %.2f, %d with %.2f\n",
                            predLabel + labelOffset, predProbs[predLabel],
                            accLabel + labelOffset, predProbs[accLabel]);
                }
            }
        }

        float acc = static_cast<float>(accTime) / dataSize;

        if (verbose) {
            printf("Test Accuracy: %.4f\n", acc);
        }

        return acc;
    }

private:
    Loader trainLoader;
    Loader testLoader;

    NetConfig   nConf;
    Network     net;
    TrainBuffer trainBuf;

    uint8_t labelOffset;
    mt19937 randGen;
};

}

#endif