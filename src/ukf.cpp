#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = n_x_ + 2;

  // Number of sigma points
  n_sig_ = 2 * n_aug_ + 1;

  // initial state vector
  x_ = VectorXd(n_x_);
  x_.fill(0.0);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // augmented sigma points matrix
  Xsig_aug_ = MatrixXd(n_aug_, n_sig_);
  Xsig_aug_.fill(0.0);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  Xsig_pred_.fill(0.0);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.8;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Prediction weights
  weights_ = tools.ProducePredictionWeights(n_sig_, n_aug_, lambda_);

  // Lidar measurement noise covariance matrix
  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << pow(std_laspx_, 2), 0,
              0,                  pow(std_laspy_, 2);

  // Radar measurement noise covariance matrix
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << pow(std_radr_, 2), 0,                   0,
              0,                 pow(std_radphi_, 2), 0,
              0,                 0,                   pow(std_radrd_, 2);

  // the current NIS for radar
  NIS_radar_ = 0;

  // the current NIS for lidar
  NIS_laser_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "UKF: " << endl;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      x_ << tools.PolarToCartesianMeasurement(meas_package.raw_measurements_),
            0, 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_, // px and py
            0, 0, 0;
    }

    previous_timestamp_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // compute the time elapsed between the current and previous measurements
  //dt - expressed in seconds
  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = meas_package.timestamp_;

  Prediction(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    if (use_radar_) UpdateRadar(meas_package.raw_measurements_);
  } else if (use_laser_) {
    UpdateLidar(meas_package.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << endl << x_ << endl;
  cout << "P_ = " << endl << P_ << endl;
}

/*******************************************************************************
 *  Prediction Logic
 ******************************************************************************/

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  ProduceAugmentedChaiSigmaMatrix();
  ApplyCTRVTranformation(delta_t);
  PredictStateMeanAndCovarinaceMatrix();
}

/**
 * Produces n_sig_ (2 * n_aug_ + 1) sigma points (augmented by process noise)
 * and assigns them to the Xsig_aug_ matrix.
 */
void UKF::ProduceAugmentedChaiSigmaMatrix() {
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create augmented mean state
  x_aug << x_, 0, 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.block(0, 0, n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug((n_x_ + 1), (n_x_ + 1)) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug_.col(0) = x_aug;

  MatrixXd root_lambda_term = sqrt(lambda_ + n_aug_) * A;
  MatrixXd x_by_n(n_aug_, n_aug_);
  x_by_n.colwise() = x_aug;
  Xsig_aug_.block(0, 1, n_aug_, n_aug_) = x_by_n + root_lambda_term;
  Xsig_aug_.block(0, (n_aug_ + 1), n_aug_, n_aug_) = x_by_n - root_lambda_term;
}

/**
 * Applies the CTRV (Constant Turning Rate and Velocity) model to each column
 * (sigma point) in Xsig_aug_ and assigns the results to the Xsig_pred_ matrix.
 * @param delta_t Time between k and k+1 in s
 */
void UKF::ApplyCTRVTranformation(double delta_t) {
  float dt2_over_2 = delta_t * delta_t / 2;

  for (unsigned int i = 0; i < n_sig_; i++) {
    //predict sigma points
    VectorXd xk = Xsig_aug_.col(i);
    float px        = xk(0);
    float py        = xk(1);
    float v         = xk(2);
    float psi       = xk(3);
    float psi_d     = xk(4);
    float nu_a      = xk(5);
    float nu_psi_dd = xk(6);

    VectorXd nu_k_influence(n_x_);
    nu_k_influence << dt2_over_2 * cos(psi) * nu_a,
                      dt2_over_2 * sin(psi) * nu_a,
                      delta_t * nu_a,
                      dt2_over_2 * nu_psi_dd,
                      delta_t * nu_psi_dd;

    VectorXd x_prime(n_x_);
    x_prime.fill(0.0);
    if (psi_d == 0) {
      // avoid division by zero
      x_prime(0) = v * cos(psi) * delta_t;
      x_prime(1) = v * sin(psi) * delta_t;
    } else {
      x_prime(0) = (v / psi_d) * (sin(psi + psi_d * delta_t) - sin(psi));
      x_prime(1) = (v / psi_d) * (cos(psi) - cos(psi + psi_d * delta_t));
    }
    x_prime(3) = psi_d * delta_t;
    // write predicted sigma points into right column
    Xsig_pred_.col(i) = xk.head(n_x_) + x_prime + nu_k_influence;
  }
}

/**
 * Sets the state vector, x_, to be the weighted mean of the predicted sigma
 * points and the state covariance matrix, P_, as the weighted self similar
 * product (w * A * A^T) of the difference between each predicted sigma point
 * and new state vector x_s. This process completes the prediction step
 * of the UKF.
 */
void UKF::PredictStateMeanAndCovarinaceMatrix() {
  //predicted state mean
  x_ = Xsig_pred_ * weights_;

  //predicted state covariance matrix
  P_ = X_difference_weighted() * X_difference().transpose();
}

/**
 * Produces the column-wise difference between the predicted sigma points,
 * Xsig_pred_, and the state vector, x_. This process includes the
 * normalisation of angle values between pi and negative pi.
 * @return MatrixXd The X_difference matrix
 */
MatrixXd UKF::X_difference() {
  MatrixXd x_by_n(n_x_, n_sig_);
  x_by_n.colwise() = x_;

  MatrixXd X_diff = Xsig_pred_ - x_by_n;
  // Normalise every angle (psi) in the 4th row.
  X_diff.row(3) = tools.NormaliseAngles(X_diff.row(3));
  return X_diff;
}

/**
 * Produces a weighted version of the X_difference matrix. Each column of the
 * X_difference matrix is multiplied by the corresponding sigma point weight.
 * @return MatrixXd The weighted X_difference matrix
 */
MatrixXd UKF::X_difference_weighted() {
  // Apply the weights to X_diff as a row wise operation.
  return X_difference().array().rowwise() * weights_.transpose().array();
}

/*******************************************************************************
 *  Update Logic
 ******************************************************************************/

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {VectorXd} z_lidar
 */
void UKF::UpdateLidar(VectorXd z_lidar) {
  // create matrix for sigma points in measurement space
  // (no transformation required)
  MatrixXd Zsig = Xsig_pred_.block(0, 0, z_lidar.size(), n_sig_);

  NIS_laser_ = UpdateUKFAndReturnNIS(z_lidar, Zsig, R_lidar_);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {VectorXd} z_radar
 */
void UKF::UpdateRadar(VectorXd z_radar) {
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = Zsig_radar();

  NIS_radar_ = UpdateUKFAndReturnNIS(z_radar, Zsig, R_radar_);
}

/**
 * Translates the predicted sigma point matrix into the radar measurement space.
 * @return MatrixXd The Zsig matrix in the radar measurement space
 */
MatrixXd UKF::Zsig_radar() {
  MatrixXd Zsig = MatrixXd(3, n_sig_);
  //transform sigma points into measurement space
  for (int i = 0; i < n_sig_; i++) {

    // extract values for better readibility
    double px  = Xsig_pred_(0,i);
    double py  = Xsig_pred_(1,i);
    double v   = Xsig_pred_(2,i);
    double psi = Xsig_pred_(3,i);

    double vx = v * cos(psi);
    double vy = v * sin(psi);

    VectorXd cartesian_vec(4);
    cartesian_vec << px, py, vx, vy;

    Zsig.col(i) = tools.CartesianToPolarMeasurement(cartesian_vec);
  }

  return Zsig;
}

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
double UKF::UpdateUKFAndReturnNIS(VectorXd z, MatrixXd Zsig, MatrixXd R) {
  //mean predicted measurement
  VectorXd z_pred = Zsig * weights_;

  //difference between Zsig and z_pred
  MatrixXd Z_diff = Z_difference(Zsig, z_pred);

  //measurement covariance matrix S
  MatrixXd S = MeasurementCovarianceMatrix(Z_diff, R);

  //cross correlation matrix Tc
  MatrixXd Tc = CrossCorrelationMatrix(Z_diff);

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual (little z_diff as opposed to big Z_diff)
  VectorXd z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;

  P_ = P_ - K * S * K.transpose();

  return tools.CalculateNIS(z_diff, S);
}

/**
 * Produces the column-wise difference between the predicted sigma points in
 * the measurement space, Zsig, and the weighted mean of that matrix.
 * @param  Zsig     The predicted sigma point matrix in the measurement space
 * @param  z_pred   The weighted mean of Zsig
 * @return MatrixXd ...
 */
MatrixXd UKF::Z_difference(MatrixXd Zsig, VectorXd z_pred) {
  MatrixXd z_pred_by_n(z_pred.size(), n_sig_);
  z_pred_by_n.colwise() = z_pred;
  return Zsig - z_pred_by_n;
}

MatrixXd UKF::MeasurementCovarianceMatrix(MatrixXd Z_diff, MatrixXd R) {
  // Apply the weights to Z_diff.
  MatrixXd Z_diff_weighted = Z_diff.array().rowwise() * weights_.transpose().array();
  //measurement covariance matrix S
  return Z_diff_weighted * Z_diff.transpose() + R;
}

MatrixXd UKF::CrossCorrelationMatrix(MatrixXd Z_diff) {
  return X_difference_weighted() * Z_diff.transpose();
}
