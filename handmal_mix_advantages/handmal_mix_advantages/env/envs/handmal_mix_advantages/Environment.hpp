//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <set>
#include <cmath>
#include <deque>
#include <vector>
#include <string>
#include <random>
#include "../../RaisimGymEnv.hpp"
#include <stdexcept>
#include <algorithm>

inline void quatToEuler(const raisim::Vec<4> &quat, Eigen::Vector3d &eulerVec)
{
    double qw = quat[0], qx = quat[1], qy = quat[2], qz = quat[3];
    // roll (x-axis rotation)
    double sinr_cosp = 2 * (qw * qx + qy * qz);
    double cosr_cosp = 1 - 2 * (qx * qx + qy * qy);
    eulerVec[0] = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (qw * qy - qz * qx);
    if (std::abs(sinp) >= 1)
        eulerVec[1] = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        eulerVec[1] = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (qw * qz + qx * qy);
    double cosy_cosp = 1 - 2 * (qy * qy + qz * qz);
    eulerVec[2] = std::atan2(siny_cosp, cosy_cosp);
}
Eigen::Vector3d sphere2cart(const Eigen::Vector3d& sphereCoords) {
    double radius = sphereCoords[0];double theta = sphereCoords[1];
    double phi = sphereCoords[2];

    double x = radius * std::sin(theta) * std::cos(phi);double y = radius * std::sin(theta) * std::sin(phi);
    double z = radius * std::cos(theta);

    return Eigen::Vector3d(x, y, z);
}

Eigen::Vector3d cart2sphere(const Eigen::Vector3d& cartCoords) {
    double x = cartCoords[0];double y = cartCoords[1];
    double z = cartCoords[2];

    double radius = std::sqrt(x*x + y*y + z*z);double theta = std::acos(z / radius);
    double phi = std::atan2(y, x);

    return Eigen::Vector3d(radius, theta, phi);
}

Eigen::VectorXd lerp(const Eigen::VectorXd& start, const Eigen::VectorXd& end, double t) {
    return (1 - t) * start + t * end;
}

Eigen::Vector3d quatRotateInverse(const Eigen::Vector4d& q, const Eigen::Vector3d& v) {
    double q_w = q(3);
    Eigen::Vector3d q_vec = q.head<3>();

    Eigen::Vector3d a = v * (2.0 * q_w * q_w - 1.0);
    Eigen::Vector3d b = q_vec.cross(v) * (2.0 * q_w);
    Eigen::Vector3d c = q_vec * (q_vec.dot(v) * 2.0);

    return a - b + c;
}

Eigen::Vector3d quatRotate(const Eigen::Vector4d& q, const Eigen::Vector3d& v) {
    double q_w = q(3);
    Eigen::Vector3d q_vec = q.head<3>();

    Eigen::Vector3d a = v * (2.0 * q_w * q_w - 1.0);
    Eigen::Vector3d b = q_vec.cross(v) * (-2.0 * q_w);
    Eigen::Vector3d c = q_vec * (q_vec.dot(v) * 2.0);

    return a + b + c;
}

Eigen::Vector3d wrap_to_pi_miniuspi(const Eigen::Vector3d& angles) {
    Eigen::Vector3d wrapped_angles = angles.array() - (2 * M_PI) * (angles.array() > M_PI).cast<double>();
    wrapped_angles = wrapped_angles.array() + (2 * M_PI) * (wrapped_angles.array() < -M_PI).cast<double>();
    return wrapped_angles;
}

namespace raisim {

    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable, int env_id) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {
            setSeed(env_id);
            for (int i = 0; i < 10; i++)
                clean_randomizer = Eigen::VectorXd::Random(1)[0];
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            gen_.seed(seed);
            /// add objects
            world_ = std::make_unique<raisim::World>();
            handmal_ = world_->addArticulatedSystem(resourceDir_+"/handmal/urdf/handmal.urdf");
            handmal_->setName("handmal_");
            handmal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            world_->addGround();
            /// config reward coefficients
            rewards_.initializeFromConfigurationFile (cfg["reward"]);
            arm_rewards_.initializeFromConfigurationFile (cfg["arm_reward"]);

            READ_YAML(double, max_time_, cfg["max_time"]);
            READ_YAML(double, simulation_dt_, cfg["simulation_dt"]);
            READ_YAML(double, control_dt_, cfg["control_dt"]);
            READ_YAML(double, act_std_val_, cfg["action_std"]);
            READ_YAML(double, curriculumFactor_, cfg["curriculumFactor"]);
            READ_YAML(double, curriculumDecayFactor_, cfg["curriculumDecay"]);

            float traj_time_upper; float traj_time_lower;
            float hold_time_upper; float hold_time_lower;
            float lin_vel_x_schedule_upper; float lin_vel_x_schedule_lower;
            float ang_vel_yaw_schedule_upper; float ang_vel_yaw_schedule_lower;
            float goal_ee_l_schedule_upper; float goal_ee_l_schedule_lower;
            float goal_ee_p_schedule_upper; float goal_ee_p_schedule_lower;
            float goal_ee_y_schedule_upper; float goal_ee_y_schedule_lower;
            float init_lin_vel_x_range_upper; float init_lin_vel_x_range_lower;
            float final_lin_vel_x_range_upper; float final_lin_vel_x_range_lower;
            float init_ang_vel_yaw_range_upper; float init_ang_vel_yaw_range_lower;
            float final_ang_vel_yaw_range_upper; float final_ang_vel_yaw_range_lower;
            float init_goal_ee_l_range_upper; float init_goal_ee_l_range_lower;
            float final_goal_ee_l_range_upper; float final_goal_ee_l_range_lower;
            float init_goal_ee_p_range_upper; float init_goal_ee_p_range_lower;
            float final_goal_ee_p_range_upper; float final_goal_ee_p_range_lower;
            float init_goal_ee_y_range_upper; float init_goal_ee_y_range_lower;
            float final_goal_ee_y_range_upper; float final_goal_ee_y_range_lower;
            float init_ee_l_range_upper; float init_ee_l_range_lower;
            float init_ee_p_range_upper; float init_ee_p_range_lower;
            float init_ee_y_range_upper; float init_ee_y_range_lower;
            float goal_ee_delta_orn_l_range_upper; float goal_ee_delta_orn_l_range_lower;
            float goal_ee_delta_orn_p_range_upper; float goal_ee_delta_orn_p_range_lower;
            float goal_ee_delta_orn_y_range_upper; float goal_ee_delta_orn_y_range_lower;

            READ_YAML(float, traj_time_upper, cfg["traj_time_upper"]);
            READ_YAML(float, traj_time_lower, cfg["traj_time_lower"]);
            READ_YAML(float, hold_time_upper, cfg["hold_time_upper"]);
            READ_YAML(float, hold_time_lower, cfg["hold_time_lower"]);
            READ_YAML(float, lin_vel_x_schedule_upper, cfg["lin_vel_x_schedule_upper"]);
            READ_YAML(float, lin_vel_x_schedule_lower, cfg["lin_vel_x_schedule_lower"]);
            READ_YAML(float, ang_vel_yaw_schedule_upper, cfg["ang_vel_yaw_schedule_upper"]);
            READ_YAML(float, ang_vel_yaw_schedule_lower, cfg["ang_vel_yaw_schedule_lower"]);
            READ_YAML(float, goal_ee_l_schedule_upper, cfg["goal_ee_l_schedule_upper"]);
            READ_YAML(float, goal_ee_l_schedule_lower, cfg["goal_ee_l_schedule_lower"]);
            READ_YAML(float, goal_ee_p_schedule_upper, cfg["goal_ee_p_schedule_upper"]);
            READ_YAML(float, goal_ee_p_schedule_lower, cfg["goal_ee_p_schedule_lower"]);
            READ_YAML(float, goal_ee_y_schedule_upper, cfg["goal_ee_y_schedule_upper"]);
            READ_YAML(float, goal_ee_y_schedule_lower, cfg["goal_ee_y_schedule_lower"]);
            READ_YAML(float, init_lin_vel_x_range_upper, cfg["init_lin_vel_x_ranges_upper"]);
            READ_YAML(float, init_lin_vel_x_range_lower, cfg["init_lin_vel_x_ranges_lower"]);
            READ_YAML(float, final_lin_vel_x_range_upper, cfg["final_lin_vel_x_ranges_upper"]);
            READ_YAML(float, final_lin_vel_x_range_lower, cfg["final_lin_vel_x_ranges_lower"]);
            READ_YAML(float, init_ang_vel_yaw_range_upper, cfg["init_ang_vel_yaw_ranges_upper"]);
            READ_YAML(float, init_ang_vel_yaw_range_lower, cfg["init_ang_vel_yaw_ranges_lower"]);
            READ_YAML(float, final_ang_vel_yaw_range_upper, cfg["final_ang_vel_yaw_ranges_upper"]);
            READ_YAML(float, final_ang_vel_yaw_range_lower, cfg["final_ang_vel_yaw_ranges_lower"]);
            READ_YAML(float, init_goal_ee_l_range_upper, cfg["init_goal_ee_l_ranges_upper"]);
            READ_YAML(float, init_goal_ee_l_range_lower, cfg["init_goal_ee_l_ranges_lower"]);
            READ_YAML(float, final_goal_ee_l_range_upper, cfg["final_goal_ee_l_ranges_upper"]);
            READ_YAML(float, final_goal_ee_l_range_lower, cfg["final_goal_ee_l_ranges_lower"]);
            READ_YAML(float, init_goal_ee_p_range_upper, cfg["init_goal_ee_p_ranges_upper"]);
            READ_YAML(float, init_goal_ee_p_range_lower, cfg["init_goal_ee_p_ranges_lower"]);
            READ_YAML(float, final_goal_ee_p_range_upper, cfg["final_goal_ee_p_ranges_upper"]);
            READ_YAML(float, final_goal_ee_p_range_lower, cfg["final_goal_ee_p_ranges_lower"]);
            READ_YAML(float, init_goal_ee_y_range_upper, cfg["init_goal_ee_y_ranges_upper"]);
            READ_YAML(float, init_goal_ee_y_range_lower, cfg["init_goal_ee_y_ranges_lower"]);
            READ_YAML(float, final_goal_ee_y_range_upper, cfg["final_goal_ee_y_ranges_upper"]);
            READ_YAML(float, final_goal_ee_y_range_lower, cfg["final_goal_ee_y_ranges_lower"]);
            READ_YAML(float, init_ee_l_range_upper, cfg["init_ee_l_ranges_upper"]);
            READ_YAML(float, init_ee_l_range_lower, cfg["init_ee_l_ranges_lower"]);
            READ_YAML(float, init_ee_p_range_upper, cfg["init_ee_p_ranges_upper"]);
            READ_YAML(float, init_ee_p_range_lower, cfg["init_ee_p_ranges_lower"]);
            READ_YAML(float, init_ee_y_range_upper, cfg["init_ee_y_ranges_upper"]);
            READ_YAML(float, init_ee_y_range_lower, cfg["init_ee_y_ranges_lower"]);
            READ_YAML(float, goal_ee_delta_orn_l_range_upper, cfg["goal_ee_delta_orn_l_ranges_upper"]);
            READ_YAML(float, goal_ee_delta_orn_l_range_lower, cfg["goal_ee_delta_orn_l_ranges_lower"]);
            READ_YAML(float, goal_ee_delta_orn_p_range_upper, cfg["goal_ee_delta_orn_l_ranges_upper"]);
            READ_YAML(float, goal_ee_delta_orn_p_range_lower, cfg["goal_ee_delta_orn_l_ranges_lower"]);
            READ_YAML(float, goal_ee_delta_orn_y_range_upper, cfg["goal_ee_delta_orn_l_ranges_upper"]);
            READ_YAML(float, goal_ee_delta_orn_y_range_lower, cfg["goal_ee_delta_orn_l_ranges_lower"]);

            traj_time_ << traj_time_lower, traj_time_upper;
            hold_time_ << hold_time_lower, hold_time_upper;
            lin_vel_x_schedule_ << lin_vel_x_schedule_lower, lin_vel_x_schedule_upper;
            ang_vel_yaw_schedule_ << ang_vel_yaw_schedule_lower, ang_vel_yaw_schedule_upper;
            goal_ee_l_schedule_ << goal_ee_l_schedule_lower, goal_ee_l_schedule_upper;
            goal_ee_p_schedule_ << goal_ee_p_schedule_lower, goal_ee_p_schedule_upper;
            goal_ee_y_schedule_ << goal_ee_y_schedule_lower, goal_ee_y_schedule_upper;
            init_lin_vel_x_ranges_ << init_lin_vel_x_range_lower, init_lin_vel_x_range_upper;
            final_lin_vel_x_ranges_ << final_lin_vel_x_range_lower, final_lin_vel_x_range_upper;
            init_ang_vel_yaw_ranges_ << init_ang_vel_yaw_range_lower, init_ang_vel_yaw_range_upper;
            final_ang_vel_yaw_ranges_ << final_ang_vel_yaw_range_lower, final_ang_vel_yaw_range_upper;
            init_goal_ee_l_ranges_ << init_goal_ee_l_range_lower, init_goal_ee_l_range_upper;
            final_goal_ee_l_ranges_ << final_goal_ee_l_range_lower, final_goal_ee_l_range_upper;
            init_goal_ee_p_ranges_ << init_goal_ee_p_range_lower * M_PI, init_goal_ee_p_range_upper * M_PI;
            final_goal_ee_p_ranges_ << final_goal_ee_p_range_lower * M_PI, final_goal_ee_p_range_upper * M_PI;
            init_goal_ee_y_ranges_ << init_goal_ee_y_range_lower * M_PI, init_goal_ee_y_range_upper * M_PI;
            final_goal_ee_y_ranges_ << final_goal_ee_y_range_lower * M_PI, final_goal_ee_y_range_upper * M_PI;
            init_ee_l_ranges_ << init_ee_l_range_lower, init_ee_l_range_upper;
            init_ee_p_ranges_ << init_ee_p_range_lower * M_PI, init_ee_p_range_upper * M_PI;
            init_ee_y_ranges_ << init_ee_y_range_lower * M_PI, init_ee_y_range_upper * M_PI;
            goal_ee_delta_orn_l_ranges_ << goal_ee_delta_orn_l_range_lower, goal_ee_delta_orn_l_range_upper;
            goal_ee_delta_orn_p_ranges_ << goal_ee_delta_orn_p_range_lower, goal_ee_delta_orn_p_range_upper;
            goal_ee_delta_orn_y_ranges_ << goal_ee_delta_orn_y_range_lower, goal_ee_delta_orn_y_range_upper;

            cf_ = curriculumFactor_;
            /// get robot data
            gcDim_ = handmal_->getGeneralizedCoordinateDim();
            gvDim_ = handmal_->getDOF();
            nJoints_ = gvDim_ - 6; // 24-6=18
            /// indices of links that should not make contact with ground
            footIndices_.insert(handmal_->getBodyIdx("LF_SHANK"));
            footIndices_.insert(handmal_->getBodyIdx("RF_SHANK"));
            footIndices_.insert(handmal_->getBodyIdx("LH_SHANK"));
            footIndices_.insert(handmal_->getBodyIdx("RH_SHANK"));
            footVec_.push_back(handmal_->getBodyIdx("LF_SHANK"));
            footVec_.push_back(handmal_->getBodyIdx("RF_SHANK"));
            footVec_.push_back(handmal_->getBodyIdx("LH_SHANK"));
            footVec_.push_back(handmal_->getBodyIdx("RH_SHANK"));
            footFrame_.push_back(handmal_->getFrameIdxByName("LF_ADAPTER_TO_FOOT"));
            footFrame_.push_back(handmal_->getFrameIdxByName("RF_ADAPTER_TO_FOOT"));
            footFrame_.push_back(handmal_->getFrameIdxByName("LH_ADAPTER_TO_FOOT"));
            footFrame_.push_back(handmal_->getFrameIdxByName("RH_ADAPTER_TO_FOOT"));
            nFoot_ = 4;
//            footFrame_.push_back(anymal_->getFrameIdxByName("LF_shank_fixed_LF_FOOT"));footFrame_.push_back(anymal_->getFrameIdxByName("RF_shank_fixed_RF_FOOT"));
//            footFrame_.push_back(anymal_->getFrameIdxByName("LH_shank_fixed_LH_FOOT"));footFrame_.push_back(anymal_->getFrameIdxByName("RH_shank_fixed_RH_FOOT"));
            /// initialize containers
            gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
            gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
            pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);
            pTarget18_.setZero(nJoints_);
            motor_strength_.setZero(nJoints_);
//            gc_init_ << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8, 0.0, -2.6, 0;
            gc_init_ << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8, 0.0, -2.6, -1.57, 0.0, 2.0, 0.0;
            /// set pd gains for leg
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero(); jointPgain.tail(6).setConstant(100.0);
            jointPgain.segment(6,12).setConstant(5.0);

            jointDgain.setZero(); jointDgain.tail(6).setConstant(2.0);
            jointDgain.segment(6,12).setConstant(0.5);
            handmal_->setPdGains(jointPgain, jointDgain);
            handmal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
            /// policy input dimenstion
            obDim_ = 96;
//            obDim_ = 1;
            obDouble_.setZero(obDim_);
            /// action & observation scaling
            actionDim_ = nJoints_;
            actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
            actionMean_ = gc_init_.tail(nJoints_);

            actionStd_leg_.setZero(12); actionStd_arm_.setZero(6);
            actionStd_leg_.setConstant(act_std_val_);
            actionStd_arm_ << 2.0, 0.5, 0.5, 0.1, 0.1, 0.1;
            actionStd_ << actionStd_leg_, actionStd_arm_;

            last_foot_state_.setZero(nFoot_);
            current_foot_state_.setZero(nFoot_);
            previous_arm_joint_position_.setZero(6);

            lin_vel_x_ranges_.setZero(); ang_vel_yaw_ranges_.setZero();
            goal_ee_l_ranges_.setZero(); goal_ee_p_ranges_.setZero(); goal_ee_y_ranges_.setZero();

            n_traj_ = 10;
            ee_start_sphere_ << 0, 0, 0;

            /// if visualizable_, add visual sphere
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer(8080);
                for (int i = 0; i < n_traj_; i++) {
                    ee_traj_vis_.push_back(server_->addVisualBox("ee_traj" + std::to_string(i+1), 0.05, 0.05, 0.05, 0, 1, 0, 1));
                }
                ee_pos_vis_ = server_->addVisualSphere("ee_pos",0.03,0.1,0.1, 0.1, 1);
                ee_target_vis_ = server_->addVisualSphere("ee_target",0.03,0.1,0.1, 0.1, 1);
                std::cout<<"server on. port: 8080"<<std::endl;
            }
//            get_init_start_ee_sphere();
        }

        void init() final {
        }

        void kill_server(){
            server_->killServer();
        }

        void reset() final{
            step_counter_ = 0;
            gc_ = gc_init_; gv_.setZero();
            // set initail pose
//            gc_.segment(3,4) += 0.15*Eigen::VectorXd::Random(4); gc_.segment(3,4).normalize();
//            gc_.segment(7,12) += 0.15*Eigen::VectorXd::Random(12); gv_.segment(6,12) = 0.25*Eigen::VectorXd::Random(12);
//            gv_.head(1) = 0.2*Eigen::VectorXd::Random(1); gv_.segment(1,1) = 0.2*Eigen::VectorXd::Random(2);
//            gv_.segment(3,3) = 0.2*Eigen::VectorXd::Random(3);
//            gc_.tail(6) += 0.1*Eigen::VectorXd::Random(6); gv_.tail(6) = 1.0*Eigen::VectorXd::Random(6);
            handmal_->setState(gc_, gv_);
            previous_arm_joint_position_ = gc_init_.tail(6);

            for (int i = 0; i < 50; i++) {
                if (server_)    server_->lockVisualizationServerMutex();
                world_->integrate();
                if (server_)    server_->unlockVisualizationServerMutex();
            }

            pTarget_history_.clear();
            joint_position_history_.clear();
            joint_velocity_history_.clear();
            eef_position_history_.clear();
            eef_quat_history_.clear();

            for (int j = 0; j < 10; j++) {
                pTarget_history_.push_back(Eigen::VectorXd::Zero(18));
                joint_position_history_.push_back(Eigen::VectorXd::Zero(18));
                joint_velocity_history_.push_back(Eigen::VectorXd::Zero(18));
                eef_position_history_.push_back(Eigen::VectorXd::Zero((6)));
                eef_quat_history_.push_back(Eigen::VectorXd::Zero(4));
            }

            updateObservation();

            draw_debug_vis(); draw_ee_goal();

            t_a_.setZero(); t_s_.setZero();
            t_a_i_.setConstant(world_->getWorldTime()); t_s_i_.setConstant(world_->getWorldTime());
        }

        Eigen::Vector2f step(const Eigen::Ref<EigenVec>& action) final {
            pTargetleg_ = action.cast<double>().head(12).cwiseProduct(actionStd_leg_) + gc_init_.segment(7,12);
            pTargetarm_ = action.cast<double>().tail(6).cwiseProduct(actionStd_arm_) + previous_arm_joint_position_;
            pTarget18_ << pTargetleg_, pTargetarm_;
//            std::cout << pTarget18_ << "\n";
//            pTarget18_ = action.cast<double>().cwiseProduct(actionStd_) + gc_init_.tail(15);
            pTarget_.tail(nJoints_) = pTarget18_;

            handmal_->setPdTarget(pTarget_, vTarget_);

            auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);

            for(howManySteps_ = 0; howManySteps_ < loopCount; howManySteps_++){
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if(server_) server_->unlockVisualizationServerMutex();
            }

            updateObservation();

            draw_debug_vis(); draw_ee_goal();

            VecDyn current_torque = handmal_->getGeneralizedForce();
            Eigen::Vector3d z_axis; z_axis << 0, 0, 1;

            /// reward calculation
            rewards_.reset();
            arm_rewards_.reset();
            double clearance = 0.08;
            /// leg rewards
            rewards_.record("orientation", \
            pow(acos((baseRot_.e() * z_axis).cwiseProduct(z_axis).sum()),2));
            rewards_.record("torque",\
            handmal_->getGeneralizedForce().e().segment(7,12).squaredNorm());
            rewards_.record("dof_pos",\
            (gc_.segment(7,12)-gc_init_.segment(7,12)).squaredNorm()); // position(15)->tend not to bend legs
            rewards_.record("dof_vel", \
            (gv_.segment(6,12)).squaredNorm());
            rewards_.record("dof_acc", \
            (gv_.segment(6,12)-joint_velocity_history_[joint_velocity_history_.size()-1].segment(6,12)).squaredNorm());
            rewards_.record("tracking_lin_vel_xy_exp", \
            (exp(-(command_.head(2)-bodyLinearVel_.head(2)).squaredNorm())));
            rewards_.record("tracking_ang_vel_exp", \
            (exp(-(command_.tail(1)-bodyAngularVel_.tail(1)).squaredNorm())));
            for (int fi = 0; fi < nFoot_; fi++){
                rewards_.record("feet_air_time",std::clamp(t_s_[fi] - t_a_[fi],-0.3,0.3),true);
                rewards_.record("foot_slip", grf_bin_[fi] * foot_vel_[fi],true);
                rewards_.record("foot_clearance",pow((foot_pos_height_[fi]-clearance),2) * pow(foot_vel_[fi],0.25),true);
            }
            rewards_.record("act_smooth1", \
            (pTarget18_.head(12) - pTarget_history_[pTarget_history_.size()-1].head(12)).squaredNorm());
            rewards_.record("act_smooth2",\
             (pTarget18_.head(12) - 2 * pTarget_history_[pTarget_history_.size()-1].head(12)
                            + pTarget_history_[pTarget_history_.size()-2].head(12)).squaredNorm())  ;
            rewards_.record("base_motion", \
            (0.8* pow(bodyLinearVel_[2],2) + 0.2*abs(bodyAngularVel_[0]) + 0.2*abs(bodyAngularVel_[1])));
            /// arm rewards
            arm_rewards_.record("arm_energy_abs_sum", \
            (0));
            arm_rewards_.record("arm_tracking_ee_sphere", \
            (exp((-cart2sphere(quatRotateInverse(base_quat_, ee_pos_- base_pos_))-curr_ee_goal_sphere_).norm())));
            arm_rewards_.record("arm_tracking_ee_orn", \
            (0));
            arm_rewards_.record("arm_dof_pos", \
            (gc_.tail(6)-gc_init_.tail(6)).squaredNorm());
            arm_rewards_.record("arm_dof_vel", \
            (gv_.tail(6)).squaredNorm());
            arm_rewards_.record("arm_dof_acc", \
            (gv_.tail(6)-joint_velocity_history_[joint_velocity_history_.size()-1].tail(6)).squaredNorm());
            arm_rewards_.record("arm_act_smooth1", \
            (pTarget18_.tail(6) - pTarget_history_[pTarget_history_.size()-1].tail(6)).squaredNorm());
            arm_rewards_.record("arm_act_smooth2", \
            (pTarget18_.tail(6) - 2 * pTarget_history_[pTarget_history_.size()-1].tail(6)
                                + pTarget_history_[pTarget_history_.size()-2].tail(6)).squaredNorm());

            pTarget_history_.push_back(pTarget18_);
            joint_position_history_.push_back(pTarget_history_[pTarget_history_.size()-1] - gc_.tail(18));
            joint_velocity_history_.push_back(gv_.tail(18));
            eef_position_history_.push_back(ee_pos_);
            eef_quat_history_.push_back(ee_quat_);

            previous_arm_joint_position_ = gc_.tail(6);
            step_counter_++;

            Eigen::Vector2f rewards;
            rewards << float(rewards_.sum()), float(arm_rewards_.sum());
            return rewards;
        }


        void observe(Eigen::Ref<EigenVec> ob) final {
            Eigen::VectorXd gripper_mass = Eigen::VectorXd::Constant(1, 0.01);
            Eigen::VectorXd total_mass_ratio = Eigen::VectorXd::Constant(1, 1);
            Eigen::VectorXd friction_coeff = Eigen::VectorXd::Constant(1, 1);
            obDouble_ <<
                    bodyOrientation_.head(2), // dim 2
                    bodyAngularVel_, // dim 3
                    gc_.tail(18) - gc_init_.tail(18), // dim 18
                    gv_.tail(18), // dim 18
                    pTarget_history_[pTarget_history_.size()-1], // dim 18 59
                    current_foot_state_, // dim 4
                    command_, // dim 3
                    curr_ee_goal_sphere_, // dim 3
                    ee_goal_delta_orn_euler_, // dim 3 72
                    total_mass_ratio, // dim 1, total mass ratio
                    baseRot_.e().transpose() * (handmal_->getCOM().e() - handmal_->getBasePosition().e()), // dim 3, com position
                    gripper_mass, // dim 1, gripper mass
                    motor_strength_, // dim 18
                    friction_coeff; // dim 1, friction coefficient 96
            /// convert it to float
            ob = obDouble_.cast<float>();
        }

        void updateObservation() {
            handmal_->getState(gc_, gv_);
            raisim::Vec<4> quat;
            quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
            raisim::quatToRotMat(quat, baseRot_);
            bodyLinearVel_ = baseRot_.e().transpose() * gv_.segment(0, 3);
            bodyAngularVel_ = baseRot_.e().transpose() * gv_.segment(3, 3);
            base_quat_ = quat.e();

            quatToEuler(quat, bodyOrientation_);
            base_pos_ = handmal_->getBasePosition().e();

            // Update velocity & ee command
            resample_commands();
            resample_ee_goal();
            // Update ee position&orientation
            get_ee_pose();
            // Update ee goal
            update_curr_ee_goal();
            // Foot contact events
            updateFootContactStates();
        }

        bool isTerminalState(float &terminalReward, float &arm_terminalReward) final{
            terminalReward = float(terminalRewardCoeff_);
            arm_terminalReward = float(arm_terminalRewardCoeff_);
            for(auto& contact: handmal_->getContacts())
                if(footIndices_.find(contact.getlocalBodyIndex())==footIndices_.end())
                    return true;

            terminalReward = 0.f;
            arm_terminalReward = 0.f;
            return false;
        }

        void curriculumUpdate() final {
            update_counter_++;
            curriculumFactor_ = std::pow(curriculumFactor_, curriculumDecayFactor_);
            cf_ = curriculumFactor_;
            handmal_->setMass(handmal_->getBodyIdx("kinova_link_1"), handmal_->getMass(handmal_->getBodyIdx("kinova_link_1"))*cf_);
            handmal_->setMass(handmal_->getBodyIdx("kinova_link_2"), handmal_->getMass(handmal_->getBodyIdx("kinova_link_2"))*cf_);
            handmal_->setMass(handmal_->getBodyIdx("kinova_link_3"), handmal_->getMass(handmal_->getBodyIdx("kinova_link_3"))*cf_);
            lin_vel_x_ranges_ = get_curriculum_value(lin_vel_x_schedule_, init_lin_vel_x_ranges_, final_lin_vel_x_ranges_, update_counter_);
            ang_vel_yaw_ranges_ = get_curriculum_value(ang_vel_yaw_schedule_, init_ang_vel_yaw_ranges_, final_ang_vel_yaw_ranges_, update_counter_);

            goal_ee_l_ranges_ = get_curriculum_value(goal_ee_l_schedule_, init_goal_ee_l_ranges_, final_goal_ee_l_ranges_, update_counter_);
            goal_ee_p_ranges_ = get_curriculum_value(goal_ee_p_schedule_, init_goal_ee_p_ranges_, final_goal_ee_p_ranges_, update_counter_);
            goal_ee_y_ranges_ = get_curriculum_value(goal_ee_y_schedule_, init_goal_ee_y_ranges_, final_goal_ee_y_ranges_, update_counter_);
        }

        void setSeed(int seed){
            std::srand(seed);
            for (int i = 0; i < 10; i++)
                clean_randomizer = Eigen::VectorXd::Random(1)[0];
        }

    private:
        void updateFootContactStates(){
            Eigen::VectorXd swing_bin, foot_pos_bin;
            raisim::Vec<3> footVelocity;
            raisim::Vec<3> footPosition;

            swing_bin.setOnes(nFoot_);
            foot_pos_bin.setZero(nFoot_);
            grf_bin_.setZero(nFoot_);
            foot_vel_.setZero(nFoot_);
            foot_pos_height_.setZero(nFoot_);

            for (int footIdx_i = 0; footIdx_i < nFoot_; footIdx_i++){
                auto footIndex = footVec_[footIdx_i];
                // check for contact event
                for (auto &contact : handmal_->getContacts()){
                    if (contact.skip()) continue;
                    if (footIndex == contact.getlocalBodyIndex()){
                        auto impulse_i = (contact.getContactFrame().e() * contact.getImpulse().e()).norm();
                        if (impulse_i > 0){
                            grf_bin_[footIdx_i] = 1.0; swing_bin[footIdx_i] = 0.0;
                        }
                    }
                }
                // measure foot velocity
                handmal_->getFrameVelocity(footFrame_[footIdx_i], footVelocity);
                handmal_->getFramePosition(footFrame_[footIdx_i], footPosition);
                foot_pos_bin[footIdx_i] = (double)(footPosition[2] > 0.04);
                foot_vel_[footIdx_i] = footVelocity.squaredNorm();
                foot_pos_height_[footIdx_i] = footPosition[2];
            }

            for (int fi = 0; fi < nFoot_; fi++){
                if (foot_pos_bin[fi] == 1) current_foot_state_[fi] = 1.0; // 0이 땅에 닿았다, 1.0이 땅에서 떨어진 상태
                else if (swing_bin[fi] == 0) current_foot_state_[fi] = 0.0;
                // change contact
                if ((current_foot_state_[fi] + last_foot_state_[fi]) == 1.0){
                    if ((current_foot_state_[fi]) == 1.0 ){
                        t_a_i_[fi] = world_->getWorldTime(); t_s_[fi] = world_->getWorldTime() - t_s_i_[fi];
                        t_a_[fi] = 0;
                    }
                    else{
                        t_s_i_[fi] = world_->getWorldTime(); t_a_[fi] = world_->getWorldTime() - t_a_i_[fi];
                        t_s_[fi] = 0;
                    }
                }
            }
            last_foot_state_ = current_foot_state_;
        }

        void get_ee_pose(){
            handmal_->getFramePosition("kinova_joint_end_effector", eef_position_rai_);
            handmal_->getFrameOrientation("kinova_joint_end_effector", eef_orientation_rai_);

            ee_pos_ = eef_position_rai_.e();
            rotMatToQuat(eef_orientation_rai_, eef_quat_rai_);
            ee_quat_ = eef_quat_rai_.e();
        }

        Eigen::Vector2f get_curriculum_value(const Eigen::Vector2f& schedule, const Eigen::Vector2f& init_range, const Eigen::Vector2f& final_range, int counter) {
            float progress = std::min(1.0f, std::max(0.0f, ((static_cast<float>(counter) - schedule[0])/(schedule[1] - schedule[0]))));
            return (final_range - init_range) * progress + init_range;
        }

        void update_curr_ee_goal(){
            traj_timesteps_ = _random_pick(traj_time_[0], traj_time_[1]);
            traj_total_timesteps_ = traj_timesteps_ + _random_pick(hold_time_[0], hold_time_[1]);
            double t;
            t = std::min(1.0, std::max(0.0, goal_timer_/traj_timesteps_));
            curr_ee_goal_sphere_ = lerp(ee_start_sphere_,ee_goal_sphere_, t);
//            std::cout << "curr_ee_goal_sphere: " << curr_ee_goal_sphere_ << std::endl;
            goal_timer_++;
            if (goal_timer_ > traj_total_timesteps_) resample_ee_goal();
        }

        void resample_commands(){
            command_ << _random_pick(lin_vel_x_ranges_[0], lin_vel_x_ranges_[1]), 0, _random_pick(ang_vel_yaw_ranges_[0], ang_vel_yaw_ranges_[1]);
//            std::cout << "command: " << command_ << std::endl;
        }

        void draw_debug_vis() {
            if (visualizable_) {
                ee_target_vis_->setPosition(base_pos_ + quatRotate(base_quat_, curr_ee_goal_cart_));
                ee_pos_vis_->setPosition(ee_pos_);
            }
        }

        void draw_ee_goal(){
            if(visualizable_) {
                for (int i = 0; i < n_traj_; ++i) {
                    double t = static_cast<double>(i) / (n_traj_ - 1);
                    Eigen::Vector3d ee_target_all_sphere = ee_start_sphere_ + (ee_goal_sphere_ - ee_start_sphere_) * t;
                    Eigen::Vector3d ee_target_all_cart_world = sphere2cart(ee_start_sphere_ + (ee_goal_sphere_ - ee_start_sphere_) * t);

                    Eigen::Vector3d final_position = base_pos_ + quatRotate(base_quat_, ee_target_all_cart_world);
                    ee_traj_vis_[i]->setPosition(final_position);
                }
            }
        }

        void resample_ee_goal(){
            _resample_ee_goal_orn_once();
            ee_start_sphere_ = ee_goal_sphere_;
            for(int i = 0; i < 10; i++){
                _resample_ee_goal_sphere_once();
//                if (not _collision_check()) break;
            }
            ee_goal_cart_ = sphere2cart(ee_goal_sphere_);
            goal_timer_ = 0.;
        }
        void _resample_ee_goal_sphere_once(){
            ee_goal_sphere_[0] = _random_pick(goal_ee_l_ranges_[0], goal_ee_l_ranges_[1]);
            ee_goal_sphere_[1] = _random_pick(goal_ee_p_ranges_[0], goal_ee_p_ranges_[1]);
            ee_goal_sphere_[2] = _random_pick(goal_ee_y_ranges_[0], goal_ee_y_ranges_[1]);
        }
        void _resample_ee_goal_orn_once(){
            double ee_goal_delta_orn_r = _random_pick(goal_ee_delta_orn_l_ranges_[0], goal_ee_l_ranges_[1]);
            double ee_goal_delta_orn_p = _random_pick(goal_ee_delta_orn_p_ranges_[0], goal_ee_p_ranges_[1]);
            double ee_goal_delta_orn_y = _random_pick(goal_ee_delta_orn_y_ranges_[0], goal_ee_y_ranges_[1]);
            ee_goal_delta_orn_euler_ << ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y;
            ee_goal_orn_euler_ = wrap_to_pi_miniuspi(ee_goal_delta_orn_euler_ + bodyOrientation_);
//            std::cout << "ee_goal_delta_orn: " << ee_goal_delta_orn_euler_ << std::endl;
        }
        void get_init_start_ee_sphere(){
            Eigen::Vector3d init_start_ee_cart;
            init_start_ee_cart << 0.15, 0, 0.15;
            init_start_ee_sphere_ = cart2sphere(init_start_ee_cart);
        }

//        bool _collision_check(){
//            return false;
//        }

        double _random_pick(double low, double high) {
            std::uniform_real_distribution<double> dist(low, high);
            return dist(gen_);
        }
        /// for initialization
        int gcDim_, gvDim_, nJoints_;
        int step_counter_=0, howManySteps_=0, update_counter_=0, n_traj_=10;
        bool visualizable_ = false;
        raisim::ArticulatedSystem* handmal_;
        std::vector<raisim::Visuals *> ee_traj_vis_;
        raisim::Visuals* ee_pos_vis_;
        raisim::Visuals* ee_target_vis_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget18_, vTarget_, actionMean_, actionStd_, obDouble_;
        Eigen::Vector3d command_;
        Eigen::VectorXd actionStd_leg_, actionStd_arm_, pTargetleg_, pTargetarm_;
        Eigen::Vector2f traj_time_, hold_time_;
        double goal_timer_, traj_timesteps_, traj_total_timesteps_;
        Eigen::Vector2f lin_vel_x_schedule_, ang_vel_yaw_schedule_;
        Eigen::Vector2f goal_ee_l_schedule_, goal_ee_p_schedule_, goal_ee_y_schedule_;
        Eigen::Vector2f lin_vel_x_ranges_, ang_vel_yaw_ranges_;
        Eigen::Vector2f init_lin_vel_x_ranges_, final_lin_vel_x_ranges_;
        Eigen::Vector2f init_ang_vel_yaw_ranges_, final_ang_vel_yaw_ranges_;
        Eigen::Vector2f init_goal_ee_l_ranges_, final_goal_ee_l_ranges_, init_ee_l_ranges_, goal_ee_l_ranges_;
        Eigen::Vector2f init_goal_ee_p_ranges_, final_goal_ee_p_ranges_, init_ee_p_ranges_, goal_ee_p_ranges_;
        Eigen::Vector2f init_goal_ee_y_ranges_, final_goal_ee_y_ranges_, init_ee_y_ranges_, goal_ee_y_ranges_;
        Eigen::Vector2f goal_ee_delta_orn_l_ranges_, goal_ee_delta_orn_p_ranges_, goal_ee_delta_orn_y_ranges_;
/// Coefficients, constants and rewards
        double terminalRewardCoeff_ = -5., arm_terminalRewardCoeff_ = -5.;
        double max_time_ = 0.;
        double curriculumFactor_ = 0., curriculumDecayFactor_ = 0., cf_ = 0.;
        double act_std_val_ = 0.;
        Eigen::VectorXd motor_strength_;
/// for foots
        std::set<size_t> footIndices_;
        std::vector<size_t> footVec_, footFrame_;
        raisim::Vec<3> RF_footPosition, LF_footPosition, RH_footPosition, LH_footPosition;
        Eigen::VectorXd foot_pos_height_, foot_vel_, grf_bin_;
        int nFoot_ = 4;
        /// end-effector pose
        raisim::Vec<3> eef_position_rai_;
        raisim::Vec<4> eef_quat_rai_;
        raisim::Mat<3,3> eef_orientation_rai_;
        Eigen::Vector3d ee_pos_, base_pos_;
        Eigen::Vector4d ee_quat_, base_quat_;
        Eigen::Vector3d ee_goal_sphere_, ee_goal_cart_, ee_start_sphere_, ee_start_cart_, ee_orn_delta_, ee_goal_delta_orn_euler_, ee_goal_orn_euler_;
        Eigen::Vector3d curr_ee_goal_sphere_, curr_ee_goal_cart_;
        Eigen::Vector3d init_start_ee_sphere_;
        /// history deque
        raisim::Mat<3,3> baseRot_;
        Eigen::Vector3d bodyOrientation_, bodyLinearVel_, bodyAngularVel_;
        Eigen::VectorXd previous_arm_joint_position_;
        std::deque<Eigen::VectorXd> pTarget_history_, joint_position_history_, joint_velocity_history_;
        std::deque<Eigen::VectorXd> eef_position_history_, eef_quat_history_;
        Eigen::Vector4d t_a_, t_s_, t_a_i_, t_s_i_;
        Eigen::VectorXd current_foot_state_, last_foot_state_;
        /// etc.
        std::normal_distribution<double> normDist_;
        thread_local static std::mt19937 gen_;
//        Eigen::Matrix3d I_ = Eigen::Matrix3d::Identity();
        double clean_randomizer;
    };
    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
}
