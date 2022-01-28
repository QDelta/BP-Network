#ifndef LOADER_HPP
#define LOADER_HPP

#include <cstdint>
#include <cassert>
#include <cstdio>
#include <vector>

inline uint32_t endianMap(uint32_t v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    v = ((v << 8) & 0xff00ff00) | ((v >> 8) & 0x00ff00ff);
    return (v << 16) | (v >> 16); 
#else
    return v;
#endif
}

namespace loader {

using namespace std;

class Loader {
    vector<vector<uint8_t>> dataSet;
    vector<uint8_t>         labels;
public:
    const vector<vector<uint8_t>>& getDataSet() const { return dataSet; }
    const vector<uint8_t>&         getLabels()  const { return labels;  }

    void load(const char *dataFilePath, const char *labelFilePath, uint32_t sampleFactor) {
        FILE* dataFile = fopen(dataFilePath, "rb");
        FILE* labelFile = fopen(labelFilePath, "rb");
        assert(dataFile && labelFile);

        uint32_t magic, nImages, nLabels;
        uint32_t rawSize, rawNRows, rawNCols;
        uint32_t size, nRows, nCols;

        fread(&magic, sizeof(magic), 1, dataFile);
        magic = endianMap(magic);
        assert(magic == 2051);

        fread(&magic, sizeof(magic), 1, labelFile);
        magic = endianMap(magic);
        assert(magic == 2049);

        fread(&nImages, sizeof(nImages), 1, dataFile);
        fread(&nLabels, sizeof(nLabels), 1, labelFile);
        nImages = endianMap(nImages);
        nLabels = endianMap(nLabels);
        assert(nImages == nLabels);

        dataSet.resize(nImages);
        labels.resize(nImages);

        fread(&rawNRows, sizeof(rawNRows), 1, dataFile);
        fread(&rawNCols, sizeof(rawNCols), 1, dataFile);
        rawNRows = endianMap(rawNRows);
        rawNCols = endianMap(rawNCols);
        rawSize = rawNRows * rawNCols;

        // sampling: k x k -> 1
        nRows = rawNRows / sampleFactor;
        nCols = rawNCols / sampleFactor;
        size = nRows * nCols;
        vector<uint8_t> rawImage(rawSize);

        uint8_t label;
        for (int t = 0; t < nImages; ++t) {
            fread(&label, sizeof(label), 1, labelFile);
            labels[t] = label;

            fread(rawImage.data(), sizeof(rawImage[0]), rawSize, dataFile);
            dataSet[t].resize(size);
            for (uint32_t i = 0; i < nRows; ++i) {
                for (uint32_t j = 0; j < nCols; ++j) {
                    uint16_t sum = 0;
                    uint32_t baseI = i * sampleFactor;
                    uint32_t baseJ = j * sampleFactor;
                    for (uint32_t ii = 0; ii < sampleFactor; ++ii) {
                        for (uint32_t jj = 0; jj < sampleFactor; ++jj) {
                            sum += rawImage[(baseI + ii) * rawNCols + (baseJ + jj)];
                        }
                    }
                    dataSet[t][i * nCols + j] = sum / (sampleFactor * sampleFactor);
                }
            }
        }

        fclose(dataFile);
        fclose(labelFile);
    }
};

}

#endif