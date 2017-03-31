#include <vector>
#include "gtest/gtest.h"
#include "../src/Eigen/Dense"
#include "../src/tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

TEST(Tools, CalculateRMSE) {
  Tools tools;

  VectorXd zero_vec = VectorXd(4);
  VectorXd ones_vec = VectorXd(4);
  VectorXd twos_vec = VectorXd(4);
  VectorXd threes_vec = VectorXd(4);
  VectorXd negative_ones_vec = VectorXd(4);
  VectorXd one_through_four_vec = VectorXd(4);
  VectorXd four_through_one_vec = VectorXd(4);
  VectorXd two_through_five_vec = VectorXd(4);
  VectorXd five_through_two_vec = VectorXd(4);
  VectorXd negative_two_to_two = VectorXd(4);
  VectorXd two_to_negative_two = VectorXd(4);
  VectorXd two_one_one_two = VectorXd(4);
  VectorXd three_one_one_three = VectorXd(4);

  zero_vec.fill(0.0);
  ones_vec.fill(1.0);
  twos_vec.fill(2.0);
  threes_vec.fill(3.0);
  negative_ones_vec.fill(-1.0);
  one_through_four_vec << 1, 2, 3, 4;
  four_through_one_vec << 4, 3, 2, 1;
  two_through_five_vec << 2, 3, 4, 5;
  five_through_two_vec << 5, 4, 3, 2;
  negative_two_to_two << -2, -1, 1, 2;
  two_to_negative_two << 2, 1, -1, -2;
  two_one_one_two << 2, 1, 1, 2;
  three_one_one_three << 3, 1, 1, 3;

  vector<VectorXd> zero_vec_collection (3, zero_vec);
  vector<VectorXd> ones_vec_collection (3, ones_vec);
  vector<VectorXd> threes_vec_collection (3, threes_vec);
  vector<VectorXd> negative_ones_vec_collection (3, negative_ones_vec);
  vector<VectorXd> one_through_four_vec_collection (3, one_through_four_vec);
  vector<VectorXd> four_through_one_vec_collection (3, four_through_one_vec);
  vector<VectorXd> two_through_five_vec_collection (3, two_through_five_vec);
  vector<VectorXd> five_through_two_vec_collection (3, five_through_two_vec);
  vector<VectorXd> negative_two_to_two_collection (3, negative_two_to_two);
  vector<VectorXd> two_to_negative_two_collection (3, two_to_negative_two);

  EXPECT_EQ(zero_vec, tools.CalculateRMSE(zero_vec_collection, zero_vec_collection));
  EXPECT_EQ(zero_vec, tools.CalculateRMSE(ones_vec_collection, ones_vec_collection));
  EXPECT_EQ(ones_vec, tools.CalculateRMSE(ones_vec_collection, zero_vec_collection));
  EXPECT_EQ(ones_vec, tools.CalculateRMSE(zero_vec_collection, ones_vec_collection));
  EXPECT_EQ(ones_vec, tools.CalculateRMSE(negative_ones_vec_collection, zero_vec_collection));
  EXPECT_EQ(ones_vec, tools.CalculateRMSE(zero_vec_collection, negative_ones_vec_collection));
  EXPECT_EQ(threes_vec, tools.CalculateRMSE(threes_vec_collection, zero_vec_collection));
  EXPECT_EQ(twos_vec, tools.CalculateRMSE(threes_vec_collection, ones_vec_collection));

  EXPECT_EQ(one_through_four_vec, tools.CalculateRMSE(one_through_four_vec_collection, zero_vec_collection));
  EXPECT_EQ(four_through_one_vec, tools.CalculateRMSE(four_through_one_vec_collection, zero_vec_collection));
  EXPECT_EQ(one_through_four_vec, tools.CalculateRMSE(two_through_five_vec_collection, ones_vec_collection));
  EXPECT_EQ(four_through_one_vec, tools.CalculateRMSE(five_through_two_vec_collection, ones_vec_collection));

  EXPECT_EQ(one_through_four_vec, tools.CalculateRMSE(zero_vec_collection, one_through_four_vec_collection));
  EXPECT_EQ(four_through_one_vec, tools.CalculateRMSE(zero_vec_collection, four_through_one_vec_collection));
  EXPECT_EQ(one_through_four_vec, tools.CalculateRMSE(ones_vec_collection, two_through_five_vec_collection));
  EXPECT_EQ(four_through_one_vec, tools.CalculateRMSE(ones_vec_collection, five_through_two_vec_collection));

  EXPECT_EQ(three_one_one_three, tools.CalculateRMSE(one_through_four_vec_collection, four_through_one_vec_collection));
  EXPECT_EQ(three_one_one_three, tools.CalculateRMSE(five_through_two_vec_collection, two_through_five_vec_collection));

  EXPECT_EQ(two_one_one_two, tools.CalculateRMSE(negative_two_to_two_collection, zero_vec_collection));
  EXPECT_EQ(two_one_one_two, tools.CalculateRMSE(two_to_negative_two_collection, zero_vec_collection));
}

TEST(Tools, PolarToCartesianMeasurement) {
  Tools tools;
  float comparison_tolerance = 1e-6;


  VectorXd zero_expectation = VectorXd(3);
  VectorXd zero_vec_3 = VectorXd(3);
  zero_expectation.fill(0.0);
  zero_vec_3.fill(0.0);

  // When given a 3x1 vector with all 0s, a 4x1 vector with all 0s is
  // returned (the comparison should be exact as there is no potential
  // for division by 0).
  EXPECT_EQ(zero_expectation, tools.PolarToCartesianMeasurement(zero_vec_3));

  // -------------------------------------------------------------------------

  VectorXd ones_expectation = VectorXd(3);
  VectorXd ones_vec_3 = VectorXd(3);
  float a = sqrt(1/(1+tan(1)*tan(1)));
  ones_expectation << a, a*tan(1), 1;
  ones_vec_3.fill(1.0);

  // When given a 3x1 vector with all 1s, a 4x1 vector with the equivalent
  // cartesian position and velocity should be returned.
  VectorXd ones_vec_calculation = tools.PolarToCartesianMeasurement(ones_vec_3);
  for (unsigned int i = 0; i < 3; i++) {
    EXPECT_NEAR(ones_expectation(i), ones_vec_calculation(i), comparison_tolerance);
  }

  // -------------------------------------------------------------------------

  VectorXd ones_inverse_expectation = VectorXd(3);
  VectorXd ones_inverse_vec_3 = VectorXd(3);
  ones_inverse_expectation << 1, 1, sqrt(2);
  ones_inverse_vec_3 << sqrt(2), M_PI/4, 2/sqrt(2);

  // When given a 3x1 vector that solves the system of equations,
  // a 4x1 vector with all 1s should be returned.
  VectorXd ones_inverse_calculation = tools.PolarToCartesianMeasurement(ones_inverse_vec_3);
  for (unsigned int i = 0; i < 3; i++) {
    EXPECT_NEAR(ones_inverse_expectation(i), ones_inverse_calculation(i), comparison_tolerance);
  }
}

TEST(Tools, CartesianToPolarMeasurement) {
  Tools tools;
  float comparison_tolerance = 1e-6;


  VectorXd zero_expectation = VectorXd(3);
  VectorXd zero_vec_4 = VectorXd(4);
  zero_expectation.fill(0.0);
  zero_vec_4.fill(0.0);

  // When given a 4x1 vector with all 0s, a 3x1 vector with all 0s is
  // returned (allowing for approximation as dvision by 0 has to be avoided).
  VectorXd zero_vec_calculation = tools.CartesianToPolarMeasurement(zero_vec_4);
  for (unsigned int i = 0; i < 3; i++) {
    EXPECT_NEAR(zero_expectation(i), zero_vec_calculation(i), comparison_tolerance);
  }

  // -------------------------------------------------------------------------

  VectorXd ones_expectation = VectorXd(3);
  VectorXd ones_vec_4 = VectorXd(4);
  ones_expectation << sqrt(2), M_PI/4, 2/sqrt(2);
  ones_vec_4.fill(1.0);

  // When given a 4x1 vector with all 1s, a 3x1 vector with the equivalent
  // polar position and radial velocity should be returned.
  VectorXd ones_vec_calculation = tools.CartesianToPolarMeasurement(ones_vec_4);
  for (unsigned int i = 0; i < 3; i++) {
    EXPECT_NEAR(ones_expectation(i), ones_vec_calculation(i), comparison_tolerance);
  }

  // -------------------------------------------------------------------------

  VectorXd ones_inverse_expectation = VectorXd(3);
  VectorXd ones_inverse_vec_4 = VectorXd(4);
  ones_inverse_expectation.fill(1.0);
  float a = sqrt(1/(1+tan(1)*tan(1)));
  ones_inverse_vec_4 << a, a*tan(1), a, a*tan(1);

  // When given a 4x1 vector that solves the system of equations,
  // a 3x1 vector with all 1s should be returned.
  VectorXd ones_inverse_calculation = tools.CartesianToPolarMeasurement(ones_inverse_vec_4);
  for (unsigned int i = 0; i < 3; i++) {
    EXPECT_NEAR(ones_inverse_expectation(i), ones_inverse_calculation(i), comparison_tolerance);
  }
}

TEST(Tools, ProducePredictionWeights) {
  Tools tools;

  // num_weights, n_aug, lambda
  float param_sets[5][3] = {
    {5, 7, -4},
    {10, 7, -4},
    {5, 7, 4},
    {5, 7, 0},
    {5, 4, 7}
  };

  for (unsigned int i = 0; i < sizeof(param_sets)/sizeof(param_sets[0]); i++) {
    float num_weights = param_sets[i][0];
    float n_aug = param_sets[i][1]; // augmented state vector size;
    float lambda = param_sets[i][2];

    VectorXd weights = tools.ProducePredictionWeights(num_weights, n_aug, lambda);
    // There should be as many weights as specified.
    EXPECT_EQ(num_weights, weights.size());
    // The first weight should be different than the the second.
    EXPECT_NE(weights(0), weights(1));
    // The second through nth weights should be the same.
    for (unsigned int k = 2; k < num_weights; k++) {
      EXPECT_EQ(weights(1), weights(k));
    }
    // The first weight should be 2*lambda larger than the second.
    EXPECT_EQ(2*lambda*weights(1), weights(0));
    // The first weight should be lambda / (lambda + naug)
    EXPECT_EQ((1.0 * lambda / (lambda + n_aug)), weights(0));
    // The second weight should be 1 / (2 * (lambda + naug))
    EXPECT_EQ((1.0 / (2.0 * (lambda + n_aug))), weights(1));
  }
}

TEST(Tools, NormaliseAngle) {
  Tools tools;
  float comparison_tolerance = 1e-7;

  EXPECT_NEAR(M_PI/2,   tools.NormaliseAngle(M_PI/2),     comparison_tolerance);
  EXPECT_NEAR(M_PI,     tools.NormaliseAngle(M_PI),       comparison_tolerance);
  EXPECT_NEAR(M_PI,     tools.NormaliseAngle(3 * M_PI),   comparison_tolerance);
  EXPECT_NEAR(0.2*M_PI, tools.NormaliseAngle(3.2 * M_PI), comparison_tolerance);
  EXPECT_NEAR(M_PI,     tools.NormaliseAngle(15 * M_PI),  comparison_tolerance);
  EXPECT_NEAR(M_PI,     tools.NormaliseAngle(150 * M_PI), comparison_tolerance);
  EXPECT_NEAR(-M_PI,    tools.NormaliseAngle(-3 * M_PI),  comparison_tolerance);
  EXPECT_NEAR(-M_PI,    tools.NormaliseAngle(-M_PI),      comparison_tolerance);
  EXPECT_NEAR(-M_PI/2,  tools.NormaliseAngle(-M_PI/2),    comparison_tolerance);
  EXPECT_NEAR(-M_PI/2,  tools.NormaliseAngle(-M_PI*15/2), comparison_tolerance);
}

TEST(Tools, NormaliseAngles) {
  Tools tools;
  float comparison_tolerance = 1e-7;

  VectorXd inputs(10);
  inputs << M_PI/2, M_PI, 3 * M_PI, 3.2 * M_PI, 15 * M_PI, 150 * M_PI, -3 * M_PI, -M_PI, -M_PI/2, -M_PI*15/2;
  VectorXd outputs = tools.NormaliseAngles(inputs);
  VectorXd expectations(10);
  expectations << M_PI/2, M_PI, M_PI, 0.2*M_PI, M_PI, M_PI, -M_PI, -M_PI, -M_PI/2, -M_PI/2;

  for (unsigned int i = 0; i < outputs.size(); i++) {
    EXPECT_NEAR(expectations(i), outputs(i), comparison_tolerance);
  }
}

TEST(Tools, CalculateNIS) {
  Tools tools;

  VectorXd zero_vec = VectorXd(3);
  VectorXd ones_vec = VectorXd(3);
  VectorXd threes_vec = VectorXd(3);
  VectorXd negative_one_to_one = VectorXd(3);
  VectorXd one_to_negative_one = VectorXd(3);
  VectorXd one_to_three = VectorXd(3);
  VectorXd four_to_six = VectorXd(3);

  zero_vec.fill(0.0);
  ones_vec.fill(1.0);
  threes_vec.fill(3.0);
  negative_one_to_one << -1, 0, 1;
  one_to_negative_one << 1, 0, -1;
  one_to_three << 1, 2, 3;
  four_to_six << 4, 5, 6;

  MatrixXd S(3, 3);
  S << 1, 0, 0,
       0, 1, 0,
       0, 0, 1;

  EXPECT_EQ(0, tools.CalculateNIS(zero_vec, S));
  EXPECT_EQ(3, tools.CalculateNIS(ones_vec, S));
  EXPECT_EQ(27, tools.CalculateNIS(threes_vec, S));

  EXPECT_EQ(2, tools.CalculateNIS(negative_one_to_one, S));
  EXPECT_EQ(2, tools.CalculateNIS(one_to_negative_one, S));
  EXPECT_EQ(14, tools.CalculateNIS(one_to_three, S));
  EXPECT_EQ(77, tools.CalculateNIS(four_to_six, S));
}
