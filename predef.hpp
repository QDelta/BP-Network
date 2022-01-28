#ifndef PREDEF_HPP
#define PREDEF_HPP

#include <Eigen/Core>

using float_t = float;

using Vector = Eigen::Vector<float_t, Eigen::Dynamic>;
using Matrix = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;

#endif