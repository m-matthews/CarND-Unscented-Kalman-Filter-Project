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
  // dimensions
  n_x_ = 5;
  n_aug_ = 7;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // Value of 0.50 selected after starting with 0.20 from lesson 7.17
  // and experimentation with 0.20, 0.50, 0.75, 1.00
  std_a_ = 0.50;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // Value of 0.50 selected after starting with 0.20 from lesson 7.17
  // and experimentation with 0.20, 0.50, 0.75, 1.00
  std_yawdd_ = 0.50;

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

  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

  // Lambda initialization
  lambda_ = 3 - n_aug_;

  // Create weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
    weights_(i) = 0.5/(n_aug_+lambda_);
  }

  // Create measurement noise covariance matrix
  R_radar_ = MatrixXd(3,3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0,std_radrd_*std_radrd_;

  R_lidar_ = MatrixXd(2,2);
  R_lidar_ << std_laspx_*std_laspx_, 0,
              0, std_laspy_*std_laspy_;

  H_lidar_ = MatrixXd(2,5);
  H_lidar_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;
  I_lidar_ = MatrixXd::Identity(5, 5);

  is_initialized_ = false;

  // Create NIS logging files
  NIS_radar_.open("../NIS_radar.txt", std::ofstream::out | std::ofstream::trunc);
  NIS_lidar_.open("../NIS_lidar.txt", std::ofstream::out | std::ofstream::trunc);
  if(!NIS_radar_.is_open() || !NIS_lidar_.is_open())
    cout << "ERROR: Cannot open NIS output file(s) in current folder." << endl;
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
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // convert radar from polar to cartesian coordinates and initialize state
      cout << "Note: Initialization performed with a RADAR measurement." << endl;
      float x = meas_package.raw_measurements_(0)*cos((float)meas_package.raw_measurements_(1));
      float y = meas_package.raw_measurements_(0)*sin((float)meas_package.raw_measurements_(1));
      x_ << x, y, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // initialize state
      cout << "Note: Initialization performed with a LASER measurement." << endl;
      x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0;
    }

    // set the initial timestamp
    previous_timestamp_ = meas_package.timestamp_;

    // initializing complete with no need to predict or update
    is_initialized_ = true;
    return;
  }

  // check if this sensor is being used
  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) ||
      (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_))
    return;

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  // compute the time elapsed between the current and previous measurements (expressed in seconds)
  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = meas_package.timestamp_;
  this->Prediction(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    this->UpdateRadar(meas_package);
  } else {
    // Laser updates
    this->UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  SigmaPointPrediction(delta_t);
  PredictMeanAndCovariance();
}

/**
 * Predicts sigma points.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::SigmaPointPrediction(double delta_t) {
  // Code developed based on lesson 7.17 Augmentation Assignment and
  // lesson 7.20 Sigma Point Prediction Assignment.

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_+1) = 0;

  // create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_*std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_*std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  double sqrt_lambda = sqrt(lambda_+n_aug_);
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i+1)        = x_aug + sqrt_lambda * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt_lambda * L.col(i);
  }

  // predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * (sin(yaw+yawd*delta_t) - sin(yaw));
      py_p = p_y + v/yawd * (cos(yaw) - cos(yaw+yawd*delta_t));
    }
    else {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

/**
 * Predict the Mean and Covariance.
 */
void UKF::PredictMeanAndCovariance() {
  // Code developed based on lesson 7.23 Predict Mean and Covariance Assignment.

  // predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    x_diff(3) = this->NormalizeAngle(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Using standard Kalman Filter as per https://discussions.udacity.com/t/lidar-update-in-ukfs/236278/16
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff = z - H_lidar_ * x_;

  MatrixXd PHt = P_ * H_lidar_.transpose();
  MatrixXd S = H_lidar_ * PHt + R_lidar_;
  MatrixXd K = PHt * S.inverse();

  // new estimate
  x_ = x_ + (K * z_diff);
  P_ = (I_lidar_ - K * H_lidar_) * P_;

  // output the NIS for validation in UKF_Analysis.ipynb.
  NIS_lidar_ << z_diff.transpose()*S.inverse()*z_diff << "\n";
  NIS_lidar_.flush();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Code developed based on lesson 7.26 Predict Radar Measurement Assignment and
  // lesson 7.29 UKF Update Assignment.

  VectorXd z = meas_package.raw_measurements_;

  int n_z = 3;

  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {

    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  // mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z,n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    z_diff(1) = this->NormalizeAngle(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  S = S + R_radar_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  // calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    z_diff(1) = this->NormalizeAngle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    x_diff(3) = this->NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;

  z_diff(1) = this->NormalizeAngle(z_diff(1));

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  // output the NIS for validation in UKF_Analysis.ipynb.
  NIS_radar_ << z_diff.transpose()*S.inverse()*z_diff << "\n";
  NIS_radar_.flush();
}

/**
 * Normalize an angle between -PI and PI.
 * @param {double} angle angle in radians to normalize.
 */
double UKF::NormalizeAngle(double angle)
{
  while (angle> M_PI) angle-=2.0*M_PI;
  while (angle<-M_PI) angle+=2.0*M_PI;
  return angle;
}
