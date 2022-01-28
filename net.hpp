#ifndef NET_HPP
#define NET_HPP

#include <cmath>
#include <cstdio>
#include "predef.hpp"

namespace net {

using namespace std;

// output layer uses softmax + cross-entropy
inline Vector softmax(const Vector& x) {
    float_t max = x.maxCoeff();
    Vector expv = x.unaryExpr([max](auto x) { return exp(x - max); });
    return expv / expv.sum();
}
inline Vector errDerive(const Vector& out, uint8_t label) {
    Vector x = out;
    x[label] -= static_cast<float>(1);
    return x;
}

// hidden layer uses sigmoid
inline Vector activ(const Vector& x) { 
    return x.unaryExpr([](auto x) { return static_cast<float_t>(1.0 / (1.0 + exp(-x))); });
}
inline Vector activDerive(const Vector& y) { 
    return y.unaryExpr([](auto y) { return static_cast<float_t>(y * (1.0 - y)); });
}

struct NetConfig {
    int nInput;
    int nHidden;
    int nOutput;
};

class TrainBuffer {
    Matrix iWGrad;
    Matrix oWGrad;
    Vector iBGrad;
    Vector oBGrad;

    Vector oLayer;
    Vector hLayer;

    friend class Network;
public:
    void init(NetConfig config) {
        iWGrad = Matrix::Zero(config.nHidden, config.nInput);
        oWGrad = Matrix::Zero(config.nOutput, config.nHidden);
        iBGrad = Vector::Zero(config.nHidden);
        oBGrad = Vector::Zero(config.nOutput);
        hLayer = Vector::Zero(config.nHidden);
        oLayer = Vector::Zero(config.nOutput);
    }

    void reset() {
        iWGrad.setZero();
        oWGrad.setZero();
        iBGrad.setZero();
        oBGrad.setZero();
        hLayer.setZero();
        oLayer.setZero();
    }

    const Vector& getOutput() const { return oLayer; }
};

class Network {
    Matrix iWeight;
    Matrix oWeight;
    Vector iBias;
    Vector oBias;
    
    int nInput;
    int nHidden;
    int nOutput;

public:
    void init(NetConfig config) {
        nInput = config.nInput;
        nHidden = config.nHidden;
        nOutput = config.nOutput;
        iWeight = Matrix::Random(nHidden, nInput);
        oWeight = Matrix::Random(nOutput, nHidden);
        iBias = Vector::Random(nHidden);
        oBias = Vector::Random(nOutput);
    }

    Vector predict(const Vector& in) const {
        Vector hLayer = activ(iWeight * in + iBias);
        Vector probs = softmax(oWeight * hLayer + oBias);
        return probs;
    }

    void updateTrainBuffer(const Vector& in, uint8_t label, TrainBuffer& tbuf) const {
        // forward propagation
        tbuf.hLayer = activ(iWeight * in + iBias);
        tbuf.oLayer = softmax(oWeight * tbuf.hLayer + oBias);

        // back propagation
        Vector d1 = errDerive(tbuf.oLayer, label);   
        Vector d2 = (oWeight.transpose() * d1).cwiseProduct(activDerive(tbuf.hLayer));

        tbuf.oBGrad += d1;
        tbuf.iBGrad += d2;
        tbuf.oWGrad += d1 * tbuf.hLayer.transpose();
        tbuf.iWGrad += d2 * in.transpose();
    }

    void applyTrainBuffer(const TrainBuffer& tbuf, float_t rate) {
        iWeight -= rate * tbuf.iWGrad;
        oWeight -= rate * tbuf.oWGrad;
        iBias -= rate * tbuf.iBGrad;
        oBias -= rate * tbuf.oBGrad;
    }

    void saveTo(const char* path) const {
        FILE* save_file = fopen(path, "w");
        assert(save_file);

        fprintf(save_file, "%d %d %d\n", nInput, nHidden, nOutput);

        for (int i = 0; i < nHidden; ++i) {
            for (int j = 0; j < nInput; ++j) {
                fprintf(save_file, "%lf ", iWeight(i, j));
            }
        }
        fputc('\n', save_file);

        for (int i = 0; i < nOutput; ++i) {
            for (int j = 0; j < nHidden; ++j) {
                fprintf(save_file, "%lf ", oWeight(i, j));
            }
        }
        fputc('\n', save_file);

        for (int i = 0; i < nHidden; ++i) {
            fprintf(save_file, "%lf ", iBias[i]);
        }
        fputc('\n', save_file);

        for (int i = 0; i < nOutput; ++i) {
            fprintf(save_file, "%lf ", oBias[i]);
        }

        fclose(save_file);
    }

    NetConfig loadFrom(const char* path) {
        FILE* load_file = fopen(path, "r");
        assert(load_file);

        NetConfig config;
        fscanf(load_file, "%d %d %d", &config.nInput, &config.nHidden, &config.nOutput);
        init(config);

        double tmp;

        for (int i = 0; i < nHidden; ++i) {
            for (int j = 0; j < nInput; ++j) {
                fscanf(load_file, "%lf", &tmp);
                iWeight(i, j) = static_cast<float_t>(tmp);
            }
        }

        for (int i = 0; i < nOutput; ++i) {
            for (int j = 0; j < nHidden; ++j) {
                fscanf(load_file, "%lf", &tmp);
                oWeight(i, j) = tmp;
            }
        }

        for (int i = 0; i < nHidden; ++i) {
            fscanf(load_file, "%lf", &tmp);
            iBias[i] = tmp;
        }

        for (int i = 0; i < nOutput; ++i) {
            fscanf(load_file, "%lf", &tmp);
            oBias[i] = tmp;
        }

        fclose(load_file);
        return config;
    }
};

}

#endif