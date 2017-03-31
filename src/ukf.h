#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* Initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* Previous timestamp
  long previous_timestamp_;

  ///* If this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* If this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* State vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* State covariance matrix
  MatrixXd P_;

  ///* Augmented sigma points matrix
  MatrixXd Xsig_aug_;

  ///* Predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Number of sigma points
  int n_sig_;

  ///* Sigma point spreading parameter
  double lambda_;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* Lidar measurement noise covariance matrix
  MatrixXd R_lidar_;

  ///* Radar measurement noise covariance matrix
  MatrixXd R_radar_;

  ///* The current NIS for radar
  double NIS_radar_;

  ///* The current NIS for laser
  double NIS_laser_;

  ///* The Tool object encapsulates a handful of helper methods
  Tools tools;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /*****************************************************************************
   *  Prediction Logic
   ****************************************************************************/

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Produces n_sig_ (2 * n_aug_ + 1) sigma points (augmented by process noise)
   * and assigns them to the Xsig_aug_ matrix.
   */
  void ProduceAugmentedChaiSigmaMatrix();

  /**
   * Applies the CTRV (Constant Turning Rate and Velocity) model to each column
   * (sigma point) in Xsig_aug_ and assigns the results to the Xsig_pred_ matrix.
   * @param delta_t Time between k and k+1 in s
   */
  void ApplyCTRVTranformation(double delta_t);

  /**
   * Sets the state vector, x_, to be the weighted mean of the predicted sigma
   * points and the state covariance matrix, P_, as the weighted self similar
   * product (w * A * A^T) of the difference between each predicted sigma point
   * and new state vector x_s. This process completes the prediction step
   * of the UKF.
   */
  void PredictStateMeanAndCovarinaceMatrix();

  /**
   * Produces the column-wise difference between the predicted sigma points,
   * Xsig_pred_, and the state vector, x_. This process includes the
   * normalisation of angle values between pi and negative pi.
   * @return MatrixXd The X_difference matrix
   */
  MatrixXd X_difference();

  /**
   * Produces a weighted version of the X_difference matrix. Each column of the
   * X_difference matrix is multiplied by the corresponding sigma point weight.
   * @return MatrixXd The weighted X_difference matrix
   */
  MatrixXd X_difference_weighted();

  /*****************************************************************************
   *  Update Logic
   ****************************************************************************/

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param z_lidar_ The measurement at k+1
   */
  void UpdateLidar(VectorXd z_lidar);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param z_radar_ The measurement at k+1
   */
  void UpdateRadar(VectorXd z_radar);

  /**
   * Translates the predicted sigma point matrix into the radar measurement space.
   * @return MatrixXd The Zsig matrix in the radar measurement space
   */
  MatrixXd Zsig_radar();

  /**
   * Performs the UKF update step (updates the state vector, x_, and state
   * covariance matrix, P_, given a new measurement) and returns the NIS
   * (Normalised Innovation Statistic) score for this update. The logic is
   * independent of the measurement source or measurement space.
   * @param  z      The measurement at k+1
   * @param  Zsig   The predicted sigma point matrix in the measurement space
   * @param  R      The measurement noise covariance matrix
   * @return double The NIS score for the performed update.
   */
  double UpdateUKFAndReturnNIS(VectorXd z, MatrixXd Zsig, MatrixXd R);

  /**
   * Produces the column-wise difference between the predicted sigma points in
   * the measurement space, Zsig, and the weighted mean of that matrix.
   * @param  Zsig     The predicted sigma point matrix in the measurement space
   * @param  z_pred   The weighted mean of Zsig
   * @return MatrixXd ...
   */
  MatrixXd Z_difference(MatrixXd Zsig, VectorXd z_pred);

  /**
   * Produces the measurement covariance matrix, S, used in the UKF update step.
   * @param  Z_diff   The predicted sigma point matrix in the measurement space
   * @return MatrixXd The measurement covariance matrix, S
   */
  MatrixXd MeasurementCovarianceMatrix(MatrixXd Z_diff, MatrixXd R);

  /**
   * Produces the cross correlation matrix, Tc, used in the UKF update step.
   * @return MatrixXd The cross correlation matrix, Tc
   */
  MatrixXd CrossCorrelationMatrix(MatrixXd Z_diff);
};

#endif /* UKF_H */
