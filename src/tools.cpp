#include <iostream>
#include <cmath>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

/**
* A helper method to calculate RMSE.
*/
VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse = VectorXd::Zero(4);

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  //accumulate squared residuals
  for (unsigned int i = 0; i < estimations.size(); ++i) {

    VectorXd residual = estimations[i] - ground_truth[i];

    //coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse / estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  return rmse;
}

/**
* A helper method to convert a polar range, bearing, and range rate vector
* into a cartesian position and velocity vector.
*/
VectorXd Tools::PolarToCartesianMeasurement(const VectorXd& x_state) {
  float rho = x_state(0);
  float phi = x_state(1);
  float rho_dot = x_state(2);

  float tan_phi = tan(phi);
  float px = sqrt(rho * rho / (1 + tan_phi * tan_phi));
  float py = tan_phi * px;
  float vx = rho_dot * cos(phi);
  float vy = rho_dot * sin(phi);
  float v  = sqrt(vx * vx + vy * vy);

  VectorXd cartesian_vec(3);
  cartesian_vec << px, py, v;
  return cartesian_vec;
}

/**
* A helper method to convert a cartesian position and speed vector into a Polar
* range, bearing, and range rate vector.
*/
VectorXd Tools::CartesianToPolarMeasurement(const VectorXd& x_state) {
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float position_vec_magnitude = sqrt(px * px + py * py);

  if (px == 0) {
    px = 1e-10;
    position_vec_magnitude = sqrt(px * px + py * py);
  }

  if (position_vec_magnitude == 0) position_vec_magnitude = 1e-10;

  VectorXd polar_vec(3);
  polar_vec << position_vec_magnitude,
               atan2(py, px),
               (px * vx + py * vy) / position_vec_magnitude;

  return polar_vec;
}

/**
* A helper method to produce the UKF sigma points weights.
*/
VectorXd Tools::ProducePredictionWeights(unsigned int num_sigma_points,
                                       unsigned int augmented_state_size,
                                       double lambda) {
  VectorXd weights(num_sigma_points);
  // set weights
  weights.fill(0.5 / (lambda + augmented_state_size));
  weights(0) = lambda / (lambda + augmented_state_size);

  return weights;
}

/**
* A helper method to normalise an angle between -pi and pi.
*/
double Tools::NormaliseAngle(double theta) {
  return fmod(theta, (M_PI + 1e-10));
}

/**
* A helper method to normalise each angle in the vector between -pi and pi.
*/
VectorXd Tools::NormaliseAngles(VectorXd thetas) {
  VectorXd normalised_thetas(thetas.size());
  for (unsigned int i = 0; i < thetas.size(); i++) {
    normalised_thetas(i) = NormaliseAngle(thetas(i));
  }
  return normalised_thetas;
}

/**
* A helper method to calculate the Normalised Innovation Squared (NIS) statistic.
*/
double Tools::CalculateNIS(VectorXd z_diff, MatrixXd S) {
  return z_diff.transpose() * S.inverse() * z_diff;
}
