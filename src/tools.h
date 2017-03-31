#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations,
                                const std::vector<Eigen::VectorXd> &ground_truth);

  /**
  * A helper method to convert a polar range, bearing, and range rate vector
  * into a cartesian position and velocity vector.
  */
  Eigen::VectorXd PolarToCartesianMeasurement(const Eigen::VectorXd& x_state);

  /**
  * A helper method to convert a cartesian position and speed vector into a Polar
  * range, bearing, and range rate vector.
  */
  Eigen::VectorXd CartesianToPolarMeasurement(const Eigen::VectorXd& x_state);

  /**
  * A helper method to produce the UKF sigma points weights.
  */
  Eigen::VectorXd ProducePredictionWeights(unsigned int num_sigma_points,
                                           unsigned int augmented_state_size,
                                           double lambda);

  /**
  * A helper method to normalise an angle between -pi and pi.
  */
  double NormaliseAngle(double theta);

  /**
  * A helper method to normalise each angle in the vector between -pi and pi.
  */
  Eigen::VectorXd NormaliseAngles(Eigen::VectorXd theta_vec);

  /**
  * A helper method to calculate the Normalised Innovation Squared (NIS) statistic.
  */
  double CalculateNIS(Eigen::VectorXd z_diff, Eigen::MatrixXd S);
};

#endif /* TOOLS_H_ */
