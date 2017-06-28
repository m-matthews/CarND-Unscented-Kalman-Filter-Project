#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // code derived from previous project "CarND-Extended-Kalman-Filter-Project".
  VectorXd rmse = VectorXd::Zero(4);

  int n = estimations.size();

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if(n != ground_truth.size() || n == 0) {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  // accumulate squared residuals
  VectorXd residual(4);
  for(unsigned int i=0; i < n; ++i) {
    residual = estimations[i] - ground_truth[i];
    // coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  // calculate the mean
  rmse = rmse/n;

  // calculate the squared root
  rmse = rmse.array().sqrt();

  return rmse;
}
