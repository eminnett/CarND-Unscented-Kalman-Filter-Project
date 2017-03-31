#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "gtest/gtest.h"
#include "../src/Eigen/Dense"
#include "../src/ukf.h"
#include "../src/ground_truth_package.h"
#include "../src/measurement_package.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

TEST(UKF, ProduceAugmentedChaiSigmaMatrix) {
  UKF ukf;
  float comparison_tolerance = 0.00001;

  ukf.std_a_ = 0.2;
  ukf.std_yawdd_ = 0.2;

  VectorXd x1(ukf.n_x_);
  x1.fill(1.0);
  VectorXd x2(ukf.n_x_);
  x2 << 5.7441, 1.3800, 2.2049, 0.5015, 0.3528;
  vector<VectorXd> state_vectors {x1, x2};

  MatrixXd P1 = MatrixXd::Identity(ukf.n_x_, ukf.n_x_);
  MatrixXd P2(ukf.n_x_, ukf.n_x_);
  P2 << 0.0043, -0.0013,  0.0030, -0.0022, -0.0020,
       -0.0013,  0.0077,  0.0011,  0.0071,  0.0060,
        0.0030,  0.0011,  0.0054,  0.0007,  0.0008,
       -0.0022,  0.0071,  0.0007,  0.0098,  0.0100,
       -0.0020,  0.0060,  0.0008,  0.0100,  0.0123;
  vector<MatrixXd> covariance_matrices {P1, P2};

  MatrixXd Xsig_aug1(ukf.n_aug_, ukf.n_sig_);
  Xsig_aug1 << 1, 2.73205, 1,       1,       1,       1,       1,       1,      -0.732051,  1,         1,         1,         1,         1,        1,
               1, 1,       2.73205, 1,       1,       1,       1,       1,       1,        -0.732051,  1,         1,         1,         1,        1,
               1, 1,       1,       2.73205, 1,       1,       1,       1,       1,         1,        -0.732051,  1,         1,         1,        1,
               1, 1,       1,       1,       2.73205, 1,       1,       1,       1,         1,         1,        -0.732051,  1,         1,        1,
               1, 1,       1,       1,       1,       2.73205, 1,       1,       1,         1,         1,         1,        -0.732051,  1,        1,
               0, 0,       0,       0,       0,       0,       0.34641, 0,       0,         0,         0,         0,         0,        -0.34641,  0,
               0, 0,       0,       0,       0,       0,       0,       0.34641, 0,         0,         0,         0,         0,         0,       -0.34641;
  MatrixXd Xsig_aug2(ukf.n_aug_, ukf.n_sig_);
  Xsig_aug2 << 5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,  5.63052,  5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
               1.38,    1.34566,   1.52806,  1.38,     1.38,     1.38,     1.38,     1.38,    1.41434,  1.23194,  1.38,     1.38,     1.38,     1.38,     1.38,
               2.2049,  2.28414,   2.24557,  2.29582,  2.2049,   2.2049,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,  2.2049,   2.2049,   2.2049,   2.2049,
               0.5015,  0.44339,   0.631886, 0.516923, 0.595227, 0.5015,   0.5015,   0.5015,  0.55961,  0.371114, 0.486077, 0.407773, 0.5015,   0.5015,   0.5015,
               0.3528,  0.299973,  0.462123, 0.376339, 0.48417,  0.418721, 0.3528,   0.3528,  0.405627, 0.243477, 0.329261, 0.22143,  0.286879, 0.3528,   0.3528,
               0,       0,         0,        0,        0,        0,        0.34641,  0,       0,        0,        0,        0,        0,       -0.34641,  0,
               0,       0,         0,        0,        0,        0,        0,        0.34641, 0,        0,        0,        0,        0,        0,       -0.34641;
  vector<MatrixXd> Xsig_aug_expectations {Xsig_aug1, Xsig_aug2};

  for (unsigned int i = 0; i < state_vectors.size(); i++) {
    ukf.x_ = state_vectors[i];
    ukf.P_ = covariance_matrices[i];

    ukf.ProduceAugmentedChaiSigmaMatrix();

    EXPECT_EQ(ukf.n_aug_, ukf.Xsig_aug_.rows());
    EXPECT_EQ(ukf.n_sig_, ukf.Xsig_aug_.cols());

    VectorXd x_aug(ukf.n_aug_);
    x_aug << ukf.x_, 0, 0;
    EXPECT_EQ(x_aug, ukf.Xsig_aug_.col(0));

    MatrixXd P_aug = MatrixXd(ukf.n_aug_, ukf.n_aug_);
    P_aug.fill(0.0);
    P_aug.block(0, 0, ukf.n_x_, ukf.n_x_) = ukf.P_;
    P_aug(ukf.n_x_, ukf.n_x_) = ukf.std_a_ * ukf.std_a_;
    P_aug((ukf.n_x_ + 1), (ukf.n_x_ + 1)) = ukf.std_yawdd_ * ukf.std_yawdd_;
    MatrixXd A = P_aug.llt().matrixL();
    MatrixXd root_lambda_term = sqrt(ukf.lambda_ + ukf.n_aug_) * A;
    MatrixXd x_by_n(ukf.n_aug_, ukf.n_aug_);
    for (unsigned int k = 0; k < ukf.n_aug_; k++) {
      x_by_n.col(k) = x_aug;
    }

    EXPECT_EQ((x_by_n + root_lambda_term), ukf.Xsig_aug_.block(0, 1, ukf.n_aug_, ukf.n_aug_));
    EXPECT_EQ((x_by_n - root_lambda_term), ukf.Xsig_aug_.block(0, (ukf.n_aug_ + 1), ukf.n_aug_, ukf.n_aug_));

    for (unsigned int m = 0; m < ukf.n_aug_; m++) {
      for (unsigned int n = 0; n < ukf.n_sig_; n++) {
        EXPECT_NEAR(Xsig_aug_expectations[i](m, n), ukf.Xsig_aug_(m, n), comparison_tolerance);
      }
    }
  }
}

TEST(UKF, ApplyCTRVTranformation) {
  UKF ukf;
  float comparison_tolerance = 0.00001;

  MatrixXd Xsig_aug2(ukf.n_aug_, ukf.n_sig_);
  Xsig_aug2 << 5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,  5.63052,  5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
               1.38,    1.34566,   1.52806,  1.38,     1.38,     1.38,     1.38,     1.38,    1.41434,  1.23194,  1.38,     1.38,     1.38,     1.38,     1.38,
               2.2049,  2.28414,   2.24557,  2.29582,  2.2049,   2.2049,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,  2.2049,   2.2049,   2.2049,   2.2049,
               0.5015,  0.44339,   0.631886, 0.516923, 0.595227, 0.5015,   0.5015,   0.5015,  0.55961,  0.371114, 0.486077, 0.407773, 0.5015,   0.5015,   0.5015,
               0.3528,  0.299973,  0.462123, 0.376339, 0.48417,  0.418721, 0.3528,   0.3528,  0.405627, 0.243477, 0.329261, 0.22143,  0.286879, 0.3528,   0.3528,
               0,       0,         0,        0,        0,        0,        0.34641,  0,       0,        0,        0,        0,        0,       -0.34641,  0,
               0,       0,         0,        0,        0,        0,        0,        0.34641, 0,        0,        0,        0,        0,        0,       -0.34641;

  MatrixXd expectation(ukf.n_x_, ukf.n_sig_);
  expectation << 5.93553, 6.06251,  5.92217,  5.9415,   5.92361,  5.93516,  5.93705, 5.93553,  5.80832,  5.94481,  5.92935,  5.94553,  5.93589,  5.93401, 5.93553,
                 1.48939, 1.44673,  1.66484,  1.49719,  1.508,    1.49001,  1.49022, 1.48939,  1.5308,   1.31287,  1.48182,  1.46967,  1.48876,  1.48855, 1.48939,
                 2.2049,  2.28414,  2.24557,  2.29582,  2.2049,   2.2049,   2.23954, 2.2049,   2.12566,  2.16423,  2.11398,  2.2049,   2.2049,   2.17026, 2.2049,
                 0.53678, 0.473387, 0.678098, 0.554557, 0.643644, 0.543372, 0.53678, 0.538512, 0.600173, 0.395462, 0.519003, 0.429916, 0.530188, 0.53678, 0.535048,
                 0.3528,  0.299973, 0.462123, 0.376339, 0.48417,  0.418721, 0.3528,  0.387441, 0.405627, 0.243477, 0.329261, 0.22143,  0.286879, 0.3528,  0.318159;

  double delta_t = 0.1;
  ukf.Xsig_aug_ = Xsig_aug2;
  ukf.ApplyCTRVTranformation(delta_t);
  EXPECT_EQ(ukf.n_x_, ukf.Xsig_pred_.rows());
  EXPECT_EQ(ukf.n_sig_, ukf.Xsig_pred_.cols());

  for (unsigned int m = 0; m < ukf.n_x_; m++) {
    for (unsigned int n = 0; n < ukf.n_sig_; n++) {
      EXPECT_NEAR(expectation(m, n), ukf.Xsig_pred_(m, n), comparison_tolerance);
    }
  }
}

TEST(UKF, PredictStateMeanAndCovarinaceMatrix) {
  UKF ukf;
  float comparison_tolerance = 0.00001;

  ukf.Xsig_pred_ <<
     5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
       1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
      2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
     0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
      0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

  ukf.PredictStateMeanAndCovarinaceMatrix();

  VectorXd x_expectation(ukf.n_x_);
  x_expectation << 5.93637, 1.49035, 2.20528, 0.536853, 0.353577;

  MatrixXd P_expectation(ukf.n_x_, ukf.n_x_);
  P_expectation << 0.00543425, -0.0024053,  0.00341576, -0.00348196, -0.00299378,
                  -0.0024053,   0.010845,   0.0014923,   0.00980182,  0.00791091,
                   0.00341576,  0.0014923,  0.00580129,  0.000778632, 0.000792973,
                  -0.00348196,  0.00980182, 0.000778632, 0.0119238,   0.0112491,
                  -0.00299378,  0.00791091, 0.000792973, 0.0112491,   0.0126972;

  for (unsigned int i = 0; i < ukf.n_x_; i++) {
      EXPECT_NEAR(x_expectation(i), ukf.x_(i), comparison_tolerance);
  }

  for (unsigned int m = 0; m < ukf.n_x_; m++) {
    for (unsigned int n = 0; n < ukf.n_x_; n++) {
      EXPECT_NEAR(P_expectation(m, n), ukf.P_(m, n), comparison_tolerance);
    }
  }
}

TEST(UKF, PredictRadarMeasurement) {
  UKF ukf;
  float comparison_tolerance = 0.00001;

  int n_z = 3;

  //radar measurement noise standard deviation radius in m
  float std_radr = 0.3;

  //radar measurement noise standard deviation angle in rad
  float std_radphi = 0.0175;

  //radar measurement noise standard deviation radius change in m/s
  float std_radrd = 0.1;

  MatrixXd R(n_z, n_z);
  R << pow(std_radr, 2), 0,                  0,
       0,                pow(std_radphi, 2), 0,
       0,                0,                  pow(std_radrd, 2);

  //create example matrix with predicted sigma points
  ukf.Xsig_pred_ <<
    5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
      1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
     2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
    0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
     0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = ukf.Zsig_radar();

  //mean predicted measurement
  VectorXd z_pred = Zsig * ukf.weights_;

  //measurement covariance matrix S
  MatrixXd S = ukf.MeasurementCovarianceMatrix(ukf.Z_difference(Zsig, z_pred), R);

  VectorXd z_pred_expectation(n_z);
  z_pred_expectation << 6.12155, 0.245993, 2.10313;

  MatrixXd S_expecation(n_z, n_z);
  S_expecation << 0.0946171,  -0.000139448,  0.00407016,
                 -0.000139448, 0.000617548, -0.000770652,
                  0.00407016, -0.000770652,  0.0180917;

  for (unsigned int i = 0; i < n_z; i++) {
    EXPECT_NEAR(z_pred_expectation(i), z_pred(i), comparison_tolerance);
  }

  for (unsigned int m = 0; m < n_z; m++) {
    for (unsigned int n = 0; n < n_z; n++) {
      EXPECT_NEAR(S_expecation(m, n), S(m, n), comparison_tolerance);
    }
  }
}

TEST(UKF, UpdateUKFAndReturnNIS) {
  UKF ukf;
  float comparison_tolerance = 0.001;

  int n_z = 3;

  //radar measurement noise standard deviation radius in m
  float std_radr = 0.3;

  //radar measurement noise standard deviation angle in rad
  float std_radphi = 0.0175;

  //radar measurement noise standard deviation radius change in m/s
  float std_radrd = 0.1;

  MatrixXd R(n_z, n_z);
  R << pow(std_radr, 2), 0,                  0,
       0,                pow(std_radphi, 2), 0,
       0,                0,                  pow(std_radrd, 2);

  //create example vector for predicted state mean
  ukf.x_ << 5.93637, 1.49035, 2.20528, 0.536853, 0.353577;

  //create example matrix for predicted state covariance
  ukf.P_ << 0.0054342, -0.002405,  0.0034157, -0.0034819, -0.00299378,
           -0.002405,   0.01084,   0.001492,   0.0098018,  0.00791091,
            0.0034157,  0.001492,  0.0058012,  0.00077863, 0.000792973,
           -0.0034819,  0.0098018, 0.00077863, 0.011923,   0.0112491,
           -0.0029937,  0.0079109, 0.00079297, 0.011249,   0.0126972;

  //create example matrix with predicted sigma points
  ukf.Xsig_pred_ <<
     5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
       1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
      2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
     0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
      0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

  MatrixXd Zsig = ukf.Zsig_radar();

  //create matrix for sigma points in measurement space
  MatrixXd Zsig_expected(n_z, ukf.n_sig_);
  Zsig_expected <<
    6.1190,  6.2334,  6.1531,  6.1283,  6.1143,  6.1190,  6.1221,  6.1190,  6.0079,  6.0883,  6.1125,  6.1248,  6.1190,  6.1188,  6.12057,
    0.24428,  0.2337, 0.27316, 0.24616, 0.24846, 0.24428, 0.24530, 0.24428, 0.25700, 0.21692, 0.24433, 0.24193, 0.24428, 0.24515, 0.245239,
    2.1104,  2.2188,  2.0639,   2.187,  2.0341,  2.1061,  2.1450,  2.1092,  2.0016,   2.129,  2.0346,  2.1651,  2.1145,  2.0786,  2.11295;

  for (unsigned int m = 0; m < n_z; m++) {
    for (unsigned int n = 0; n < ukf.n_sig_; n++) {
      EXPECT_NEAR(Zsig_expected(m, n), Zsig(m, n), comparison_tolerance);
    }
  }

  //mean predicted measurement
  VectorXd z_pred = Zsig * ukf.weights_;

  //measurement covariance matrix S
  MatrixXd S = ukf.MeasurementCovarianceMatrix(ukf.Z_difference(Zsig, z_pred), R);

  VectorXd z_pred_expectation(n_z);
  z_pred_expectation << 6.12155, 0.245993, 2.10313;

  MatrixXd S_expecation(n_z, n_z);
  S_expecation << 0.0946171,  -0.000139448,  0.00407016,
                 -0.000139448, 0.000617548, -0.000770652,
                  0.00407016, -0.000770652,  0.0180917;

  for (unsigned int i = 0; i < n_z; i++) {
    EXPECT_NEAR(z_pred_expectation(i), z_pred(i), comparison_tolerance);
  }

  for (unsigned int m = 0; m < n_z; m++) {
    for (unsigned int n = 0; n < n_z; n++) {
      EXPECT_NEAR(S_expecation(m, n), S(m, n), comparison_tolerance);
    }
  }

  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z << 5.9214,   //rho in m
       0.2187,   //phi in rad
       2.0062;   //rho_dot in m/s

  float nis = ukf.UpdateUKFAndReturnNIS(z, Zsig, R);

  EXPECT_NEAR(2.54036, nis, comparison_tolerance);

  VectorXd x_expectation(ukf.n_x_);
  x_expectation << 5.92276, 1.41823, 2.15593, 0.489274, 0.321338;

  MatrixXd P_expectation(ukf.n_x_, ukf.n_x_);
  P_expectation << 0.00361579, -0.000357881, 0.00208316, -0.000937196, -0.00071727,
                  -0.000357881, 0.00539867,  0.00156846,  0.00455342,   0.00358885,
                   0.00208316,  0.00156846,  0.00410651,  0.00160333,   0.00171811,
                  -0.000937196, 0.00455342,  0.00160333,  0.00652634,   0.00669436,
                  -0.00071719,  0.00358884,  0.00171811,  0.00669426,   0.00881797;

  for (unsigned int i = 0; i < ukf.n_x_; i++) {
    EXPECT_NEAR(x_expectation(i), ukf.x_(i), comparison_tolerance);
  }

  for (unsigned int m = 0; m < ukf.n_x_; m++) {
    for (unsigned int n = 0; n < ukf.n_x_; n++) {
      EXPECT_NEAR(P_expectation(m, n), ukf.P_(m, n), comparison_tolerance);
    }
  }
}
