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

namespace raisim {

    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable, int env_id) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {
            /// Set Seeds
            setSeed(env_id);
            for (int i = 0; i < 10; i++)
                clean_randomizer = Eigen::VectorXd::Random(1)[0];
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            gen_.seed(seed);
            /// Add objects
            world_ = std::make_unique<raisim::World>();
            handmal_ = world_->addArticulatedSystem(resourceDir_+"/handmal_test/urdf/handmal_test.urdf");
            handmal_->setName("handmal_");
            handmal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            world_->addGround();
            /// Set configuration
            rewards_.initializeFromConfigurationFile (cfg["reward"]);
            READ_YAML(double, max_time_, cfg["max_time"]); READ_YAML(double, simulation_dt_, cfg["simulation_dt"]);
            READ_YAML(double, control_dt_, cfg["control_dt"]); READ_YAML(double, act_std_val_, cfg["action_std"]);
            READ_YAML(double, curriculumFactor_, cfg["curriculumFactor"]); READ_YAML(double, curriculumDecayFactor_, cfg["curriculumDecay"]);
            READ_YAML(double, clearance_, cfg["clearance"]);
            cf_ = curriculumFactor_;
            /// Get robot data
            gcDim_ = handmal_->getGeneralizedCoordinateDim();
            gvDim_ = handmal_->getDOF();
            nJoints_ = gvDim_ - 6; // 21-6=15
            /// Indices of links that should not make contact with ground
            footIndices_.insert(handmal_->getBodyIdx("LF_SHANK"));footIndices_.insert(handmal_->getBodyIdx("RF_SHANK"));
            footIndices_.insert(handmal_->getBodyIdx("LH_SHANK"));footIndices_.insert(handmal_->getBodyIdx("RH_SHANK"));
            footVec_.push_back(handmal_->getBodyIdx("LF_SHANK"));footVec_.push_back(handmal_->getBodyIdx("RF_SHANK"));
            footVec_.push_back(handmal_->getBodyIdx("LH_SHANK"));footVec_.push_back(handmal_->getBodyIdx("RH_SHANK"));
            footFrame_.push_back(handmal_->getFrameIdxByName("LF_ADAPTER_TO_FOOT"));footFrame_.push_back(handmal_->getFrameIdxByName("RF_ADAPTER_TO_FOOT"));
            footFrame_.push_back(handmal_->getFrameIdxByName("LH_ADAPTER_TO_FOOT"));footFrame_.push_back(handmal_->getFrameIdxByName("RH_ADAPTER_TO_FOOT"));
            nFoot_ = 4;
            /// Initialize containers
            gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
            gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
            pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);
            pTarget15_.setZero(nJoints_);
            velocity_command_.setZero();
            eef_command_.setZero();
            gc_init_ << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8, 0.0, -2.6, 0;
            // in gc_: head(3):base position, segment(3,4):base orientation, segment(7,12):leg joint position, tail(3):arm joint position
            // in gv_: head(3):base linear_velocity, segment(3,3):base angular_velocity, segment(6,12):leg joint velocity, tail(3):arm joint velocity
            /// Set PD gains for legs and arm
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero(); jointPgain.tail(3).setConstant(400.0);
            jointPgain.segment(6,12).setConstant(50.0);
            jointDgain.setZero(); jointDgain.tail(3).setConstant(4.0);
            jointDgain.segment(6,12).setConstant(0.4);
            handmal_->setPdGains(jointPgain, jointDgain);
            handmal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
            /// Policy input dimenstion

            //            obDim_ = 207;
            obDim_ = 213;
            obDouble_.setZero(obDim_);
            /// Action & Observation scaling
            actionDim_ = nJoints_;
            actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
            actionMean_ = gc_init_.tail(nJoints_);
            actionStd_leg_.setZero(12); actionStd_arm_.setZero(3);
            actionStd_leg_.setConstant(act_std_val_);
            actionStd_arm_ << 0.1, 0.1, 0.1;
            actionStd_ << actionStd_leg_, actionStd_arm_;

            eef_position_.setZero(3);
            eef_position_init_.setZero(3);
            eef_orientation_.setZero(4);
            eef_orientation_init_.setZero(4);
            /// Set arm mass to zero when early training
            handmal_->setMass(handmal_->getBodyIdx("kinova_link_1"), handmal_->getMass(handmal_->getBodyIdx("kinova_link_1")) * 0.1);
            handmal_->setMass(handmal_->getBodyIdx("kinova_link_2"), handmal_->getMass(handmal_->getBodyIdx("kinova_link_2")) * 0.1);
            handmal_->setMass(handmal_->getBodyIdx("kinova_link_3"), handmal_->getMass(handmal_->getBodyIdx("kinova_link_3")) * 0.1);
            /// Init buffers about foot contact
            grf_bin_.setZero(nFoot_);
            foot_pos_height_.setZero(nFoot_);
            foot_vel_.setZero(nFoot_);
            v_x_max_ = 1.0 + 2.5/(1+exp(2));
            v_w_max_ = 0.5 + 1.0/(1+exp(2));

            /// Init visual objects
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer(8080);
                for (int i = 0; i < n_balls_; i++) {
                    desired_command_traj_.push_back(
                            server_->addVisualBox("desired_command_pos" + std::to_string(i+1), 0.05, 0.05, 0.05, 0, 1, 0, 1));
                }
                object_ = server_->addVisualSphere("manipulation_command_position",0.03, 0.1, 0.1,0.1, 1);
                object_eef_ = server_->addVisualSphere("eef_position",0.03,0.1,0.1, 0.1, 1);
                std::cout<<"server on. port: 8080"<<std::endl;
            }
        }


        void init() final {
        }


        void kill_server(){
            server_->killServer();
        }


        void reset() final{
            step_counter_ = 0;

            gc_ = gc_init_; gv_.setZero();
            /// If in training, add random noise.
            if(!setcmd_) {
                gc_.segment(3, 4) += 0.2 * Eigen::VectorXd::Random(4);
                gc_.segment(3, 4).normalize();
                gc_.segment(7, 12) += 0.2 * Eigen::VectorXd::Random(12);
                gv_.segment(6, 12) = 1.5 * Eigen::VectorXd::Random(12);
                gv_.head(1) = 0.5 * Eigen::VectorXd::Random(1);
                gv_.segment(1, 2) = 0.25 * Eigen::VectorXd::Random(2);
                gv_.segment(3, 3) = 0.25 * Eigen::VectorXd::Random(3);
                gc_.tail(3) += 0.2 * Eigen::VectorXd::Random(3);
                gv_.tail(3) = 0.25 * Eigen::VectorXd::Random(3);
            }

            handmal_->setState(gc_, gv_);
            pTarget_.tail(nJoints_) = Eigen::VectorXd::Zero(15) + actionMean_;
            handmal_->setPdTarget(pTarget_, vTarget_);

            for (int i = 0; i < 50; i++){
                if (server_)    server_->lockVisualizationServerMutex();
                world_->integrate();
                if (server_)    server_->unlockVisualizationServerMutex();
            }
            /// Clear buffers
            pTarget_history_.clear();
            joint_position_history_.clear();
            joint_velocity_history_.clear();
            eef_position_history_.clear();
            eef_orientation_history_.clear();
            last_foot_state_.setZero(nFoot_);
            current_foot_state_.setZero(nFoot_);
            t_a_.setZero(); t_s_.setZero(); t_a_i_.setConstant(world_->getWorldTime()); t_s_i_.setConstant(world_->getWorldTime());
            previous_arm_joint_position_ = gc_.tail(3);
            /// Set history buffers to zeros
            for (int j = 0; j < 10; j++) {
                pTarget_history_.push_back(Eigen::VectorXd::Zero(15));
                joint_position_history_.push_back(Eigen::VectorXd::Zero(15));
                joint_velocity_history_.push_back(Eigen::VectorXd::Zero(15));
                eef_position_history_.push_back(Eigen::VectorXd::Zero(3));
                eef_orientation_history_.push_back(Eigen::VectorXd::Zero(4));
            }
            /// Update observations, get end-effector initial position, sample velocity commands
            updateObservation();
            get_base_eef_init_position();
            sample_velocity_command();
            /// Visualization robot path
            if (visualizable_) {
                if(not stance_) {
                    object_->setColor(1,0,1,1);
                    object_->setPosition(0,0,-1);
                    object_eef_->setColor(0,1,1,1);
                    for (int i = 0; i < n_balls_; i++) {
                        if (velocity_command_[2] != 0) {
                            coordinatex_ = velocity_command_[0] / velocity_command_[2] * std::sin(velocity_command_[2] * i * 0.5)
                                           + velocity_command_[1] / velocity_command_[2] * (std::cos(velocity_command_[2] * i * 0.5) - 1);
                            coordinatey_ = -velocity_command_[0] / velocity_command_[2] * (std::cos(velocity_command_[2] * i * 0.5) - 1)
                                           + velocity_command_[1] / velocity_command_[2] * std::sin(velocity_command_[2] * i * 0.5);
                        } else {
                            coordinatex_ = velocity_command_[0] * i * 0.5;
                            coordinatey_ = velocity_command_[1] * i * 0.5;
                        }
                        desired_command_traj_[i]->setPosition({coordinatex_, coordinatey_, 0.2});
                    }
                }
                else {
                    object_->setColor(0,0,1,1);
                    object_eef_->setColor(0,0,1,1);
                    object_->setPosition(desired_eef_position_w_);
                    object_eef_->setPosition(eef_position_);
                }
            }
        }


        float step(const Eigen::Ref<EigenVec>& action) final {
            pTargetleg_ = action.cast<double>().head(12).cwiseProduct(actionStd_leg_) + gc_init_.segment(7,12);
            pTargetarm_ = action.cast<double>().tail(3).cwiseProduct(actionStd_arm_) + previous_arm_joint_position_; // only ee position tracking
            pTarget15_ << pTargetleg_, pTargetarm_;
            pTarget_.tail(nJoints_) = pTarget15_;

            handmal_->setPdTarget(pTarget_, vTarget_);

            auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);

            for(howManySteps_ = 0; howManySteps_ < loopCount; howManySteps_++){
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if(server_) server_->unlockVisualizationServerMutex();
            }

            updateObservation();
            if (visualizable_) {
                if(step_counter_==10){
                    object_->setColor(0,1,0,1);
                }
                object_eef_->setPosition((eef_position_rai_.e()));
            }
            step_counter_++;
            /// Compute rewards
            VecDyn current_torque = handmal_->getGeneralizedForce();
            rewards_.reset();
            for (int fi = 0; fi < nFoot_; fi++){
                if (stance_) {
                    rewards_.record("airtime",std::clamp(t_s_[fi] - t_a_[fi],-0.3,0.3),true);
                }
                else {
                    if (std::max(t_a_[fi], t_s_[fi]) < 0.25) {
                        rewards_.record("airtime", std::min(std::max(t_a_[fi], t_s_[fi]), 0.2), true);
                    }
                    else rewards_.record("airtime", 0, true);
                }
                rewards_.record("footslip",grf_bin_[fi] * foot_vel_[fi],true);
                rewards_.record("footclearance",pow((foot_pos_height_[fi] - clearance_),2) * pow(foot_vel_[fi],0.25),true);
            }

            Eigen::Vector3d z_axis; z_axis << 0, 0, 1;
            rewards_.record("linvel",
                            exp(-(!stance_)*(velocity_command_.head(2)-bodyLinearVel_.head(2)).squaredNorm()));
            rewards_.record("angvel",
                            exp(-(!stance_)*(velocity_command_.segment(2,1)-bodyAngularVel_.tail(1)).squaredNorm()));
            rewards_.record("manipulcmd",
                            exp(-1.5 * (stance_)*(desired_eef_position_ - eef_position_).squaredNorm()/pow(0.04/cf_,2)));
            // Calculate the cosine of the angle between the base rotation's z-axis and the given z-axis
            double cosine = (baseRot_.e().transpose() * z_axis).cwiseProduct(z_axis).sum();
            // Ensure the cosine value is within the valid range for acos
            if (cosine >= -1.0 && cosine <= 1.0) {
                // Calculate the angle and record it as the "ori" reward
                double angle = pow(acos(cosine), 2);
                rewards_.record("ori", angle);
            } else {
                // Handle the case where the cosine value is out of range
                // For example, you can set a default value or log a warning message
                std::cerr << "Warning: Cosine value out of range for acos calculation" << std::endl;
                rewards_.record("ori", 0); // Set a default value
            }
            rewards_.record("jtorque",
                            current_torque.e().segment(6,15).squaredNorm());
            rewards_.record("jposition",
                            (gc_.segment(7,15)-gc_init_.segment(7,15)).squaredNorm());
            rewards_.record("jspeed",
                            (gv_.tail(15)).squaredNorm());
            rewards_.record("jacc",
                            (gv_.segment(6,12)-joint_velocity_history_[joint_velocity_history_.size()-1].head(12)).squaredNorm());
            rewards_.record("actsmooth1",
                            float((pTarget15_ - pTarget_history_[pTarget_history_.size()-1]).squaredNorm()));
            rewards_.record("actsmooth2",
                            float((pTarget15_ - 2 * pTarget_history_[pTarget_history_.size()-1] + pTarget_history_[pTarget_history_.size()-2]).squaredNorm()));
            rewards_.record("base",
                            (0.8* pow(bodyLinearVel_[2],2) + 0.2*abs(bodyAngularVel_[0]) + 0.2*abs(bodyAngularVel_[1])));
            /// Update history buffers
            pTarget_history_.push_back(pTarget15_);
            joint_position_history_.push_back(pTarget_history_[pTarget_history_.size()-1] - gc_.tail(15));
            joint_velocity_history_.push_back(gv_.tail(15));
            eef_position_history_.push_back(eef_position_);
            eef_orientation_history_.push_back(eef_orientation_);

            float positive_reward, negative_reward;
            positive_reward = rewards_["linvel"] + rewards_["angvel"] + rewards_["manipulcmd"] + rewards_["airtime"];
            negative_reward = cf_ * (rewards_["footslip"] + rewards_["footclearance"] + rewards_["ori"] + rewards_["jtorque"] + rewards_["jposition"] \
            + rewards_["jspeed"] + rewards_["jacc"]  + rewards_["base"] + rewards_["actsmooth1"] + rewards_["actsmooth2"]);

            return float(positive_reward*exp(0.2*negative_reward)/howManySteps_);
        }


        void observe(Eigen::Ref<EigenVec> ob) final {
            obDouble_ <<
                        bodyOrientation_, // body orientation 3
                        bodyLinearVel_, /// body linear velocity 3
                        bodyAngularVel_, // body angular velocity 3
                        gc_.tail(15), // joint state1: joint angle 15
                        gv_.tail(15), // joint state2: joint velocity 15 // 45
                        pTarget_history_[pTarget_history_.size()-1], // desired joint position = action history 15*2=30 // 75
                        pTarget_history_[pTarget_history_.size()-3],
                        joint_position_history_[joint_position_history_.size()-1], // joint history1: joint angle 15*3=45 // 120
                        joint_position_history_[joint_position_history_.size()-3],
                        joint_position_history_[joint_position_history_.size()-5],
                        joint_velocity_history_[joint_velocity_history_.size()-1], // joint history2: joint velocity 15*3=45 // 165
                        joint_velocity_history_[joint_velocity_history_.size()-3],
                        joint_velocity_history_[joint_velocity_history_.size()-5],
                        robot_COM_-RF_footPosition_.e(), // relative foot position to robot COM 3*4=12 177
                        robot_COM_-LF_footPosition_.e(),
                        robot_COM_-RH_footPosition_.e(),
                        robot_COM_-LH_footPosition_.e(),
                        current_foot_state_, // contact foot binary state 4 181
                        foot_pos_height_, // foot height 4 185
                        eef_position_history_[eef_position_history_.size()-1], // end-effector position (in base frame) 3*3=9 194
                        eef_position_history_[eef_position_history_.size()-3],
                        eef_position_history_[eef_position_history_.size()-5],
                        eef_orientation_history_[eef_orientation_history_.size()-1], // end-effector quaternion (in base frame_, 4*3=12 206
                        eef_orientation_history_[eef_orientation_history_.size()-3],
                        eef_orientation_history_[eef_orientation_history_.size()-5],
                        eef_position_, // end-effector position (in base frame) 209
                        eef_orientation_, // end-effector quaternion (in base frame), 213
                        velocity_command_, // 3
                        eef_command_; //3
            /// Convert it to float
            ob = obDouble_.cast<float>();
        }


        void updateObservation() {
            handmal_->getState(gc_, gv_);

            raisim::Vec<4> quat;
            base_position_[0] = gc_[0]; base_position_[1] = gc_[1]; base_position_[2] = gc_[2];
            quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
            raisim::quatToRotMat(quat, baseRot_);
            bodyLinearVel_ = baseRot_.e().transpose() * gv_.segment(0, 3);
            bodyAngularVel_ = baseRot_.e().transpose() * gv_.segment(3, 3);

            robot_COM_ = handmal_->getCOM().e();
            handmal_->getFramePosition(footFrame_[0], LF_footPosition_);
            handmal_->getFramePosition(footFrame_[1], RF_footPosition_);
            handmal_->getFramePosition(footFrame_[2], LH_footPosition_);
            handmal_->getFramePosition(footFrame_[3], RH_footPosition_);
            quatToEuler(quat, bodyOrientation_);

            previous_arm_joint_position_ = gc_.tail(3);
            /// Foot contact events
            Eigen::VectorXd swing_bin;
            Eigen::VectorXd foot_pos_bin;
            raisim::Vec<3> footVelocity;
            raisim::Vec<3> footPosition;
            Eigen::Matrix3d eef_orientation_matrix;

            swing_bin.setOnes(nFoot_);
            foot_pos_bin.setZero(nFoot_);

            handmal_->getFramePosition("kinova_joint_end_effector", eef_position_rai_);
            handmal_->getFrameOrientation("kinova_joint_end_effector", eef_orientation_rai_);

            eef_position_ = baseRot_.e().transpose() * (eef_position_rai_.e() - base_position_);
            eef_orientation_matrix = baseRot_.e().transpose() * eef_orientation_rai_.e();

            Eigen::Quaterniond quat_from_matrix(eef_orientation_matrix);
            eef_orientation_[0] = quat_from_matrix.w();
            eef_orientation_[1] = quat_from_matrix.x();
            eef_orientation_[2] = quat_from_matrix.y();
            eef_orientation_[3] = quat_from_matrix.z();

            for (int footIdx_i = 0; footIdx_i < nFoot_; footIdx_i++) {
                auto footIndex = footVec_[footIdx_i];
                // check for contact event
                for (auto &contact : handmal_->getContacts()) {
                    if (contact.skip())
                        continue;
                    if (footIndex == contact.getlocalBodyIndex()) {
                        auto impulse_i = (contact.getContactFrame().e() * contact.getImpulse().e()).norm();
                        if (impulse_i > 0) {// Contact
                            grf_bin_[footIdx_i] = 1.0;
                            swing_bin[footIdx_i] = 0.0;
                        }
                    }
                }
                /// measure foot position and velocity
                handmal_->getFrameVelocity(footFrame_[footIdx_i], footVelocity);
                handmal_->getFramePosition(footFrame_[footIdx_i], footPosition);
                foot_vel_[footIdx_i] = footVelocity.squaredNorm();
                foot_pos_bin[footIdx_i] = (double)(footPosition[2] > clearance_);
                foot_pos_height_[footIdx_i] = footPosition[2];
            }
            /// Update t_s, t_a if contacts change
            for (int fi = 0; fi < nFoot_; fi++) {
                if (foot_pos_bin[fi] == 1) current_foot_state_[fi] = 1.0; // 0 = contact, 1 = ~contact
                else if (swing_bin[fi] == 0) current_foot_state_[fi] = 0.0;
                // change contact
                if ((current_foot_state_[fi] + last_foot_state_[fi]) == 1.0) {
                    if ((current_foot_state_[fi]) == 1.0) {
                        t_a_i_[fi] = world_->getWorldTime();
                        t_s_[fi] = world_->getWorldTime() - t_s_i_[fi];
                    }
                    else {
                        t_s_i_[fi] = world_->getWorldTime();
                        t_a_[fi] = world_->getWorldTime() - t_a_i_[fi];
                    }
                }
            }
            last_foot_state_ = current_foot_state_;
        }


        bool isTerminalState(float &terminalReward) final {
            terminalReward = float(terminalRewardCoeff_);
            for(auto& contact: handmal_->getContacts())
                if(footIndices_.find(contact.getlocalBodyIndex())==footIndices_.end())
                    return true;
            terminalReward = 0.f;
            return false;
        }

        void curriculumUpdate() final {
            itr_n_++;
            cf_ = std::pow(cf_, curriculumDecayFactor_);
            /// Velocity curriculum
            v_x_max_ = 1.0 + 2.5/(1+exp(-0.002*(itr_n_-1000)));
//            v_y_max_ = 0.3 + 0.6/(1+exp(-0.002*(itr_n_-1000)));
            v_w_max_ = 0.5 + 1.0/(1+exp(-0.002*(itr_n_-1000)));
            /// Arm mass curriculum
            if (itr_n_ > 1000) {
                handmal_->setMass(handmal_->getBodyIdx("kinova_link_1"),
                                  handmal_->getMass(handmal_->getBodyIdx("kinova_link_1")) * pow(cf_,1/(pow(curriculumFactor_,1000))));
                handmal_->setMass(handmal_->getBodyIdx("kinova_link_2"),
                                  handmal_->getMass(handmal_->getBodyIdx("kinova_link_2")) * pow(cf_,1/(pow(curriculumFactor_,1000))));
                handmal_->setMass(handmal_->getBodyIdx("kinova_link_3"),
                                  handmal_->getMass(handmal_->getBodyIdx("kinova_link_3")) * pow(cf_,1/(pow(curriculumFactor_,1000))));
            }
        }


        void setSeed(int seed) final{
            std::srand(seed);
            for (int i = 0; i < 10; i++)
                clean_randomizer = Eigen::VectorXd::Random(1)[0];
        }


        void setCommand(Eigen::Ref<EigenVec> command, bool stance) final{
            /// for test
            setcmd_ = true;
            velocity_command_ = command.head(3).cast<double>();
            eef_command_ = command.tail(3).cast<double>();
            stance_ = stance;
        }


        void get_base_eef_init_position(){
            raisim::Vec<3> eef_pos;
            handmal_->getFramePosition("kinova_joint_end_effector", eef_pos);
            eef_position_init_w_ = eef_pos.e();

            raisim::Vec<4> quat;
            quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
            raisim::quatToRotMat(quat, baseRot_);
            eef_position_init_ = baseRot_.e().transpose() * (eef_position_init_w_-base_position_);
        }


        void sample_velocity_command() {
            std::uniform_real_distribution<> distrib(0,1);
            t_stance_ = distrib(gen_);
            t_x_ = distrib(gen_); t_y_ = distrib(gen_); t_w_ = distrib(gen_);
            if(!setcmd_){
                velocity_command_.setZero();
                eef_command_.setZero();
                if (t_stance_ < 0.2) {
                    /// 20% stance command, do manipulation
                    stance_ = true;
                    r_ = cf_ * (distrib(gen_) * 0.2 + 0.15);
                    theta_ = distrib(gen_) * 2 * M_PI;
                    phi_ = distrib(gen_) * M_PI;
                    eef_command_.tail(3) <<  r_ * cos(theta_) * sin(phi_),
                            r_ * sin(theta_) * sin(phi_),
                            r_ * cos(phi_);
                } else {
                    /// 80% only locomotion
                    stance_ = false;
                    velocity_command_ << v_x_max_ * (-0.5 + 1.5 * t_x_), 0 , v_w_max_ * (-1 + 2 * t_w_);
//                velocity_command_[0] = v_x_max_*(-0.5+1.5*t_x_);
//                velocity_command_[1] = v_y_max_*(-1+2*t_y_);
//                velocity_command_[2] = v_w_max_*(-1+2*t_w_);
                }
            }
            Eigen::Vector3d base_position = base_position_;
            base_position[2] = 0.5; // z_invariant
            desired_eef_position_ = eef_command_.tail(3) + base_position; // base frame
            desired_eef_position_w_ = baseRot_.e() * desired_eef_position_ + base_position_; // world frame
        }


    private:
        int gcDim_, gvDim_, nJoints_, commandDim_;
        bool visualizable_ = false;
        raisim::ArticulatedSystem* handmal_;
        raisim::Mat<3,3> baseRot_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget15_, vTarget_;
        Eigen::VectorXd actionStd_leg_, actionStd_arm_, pTargetleg_, pTargetarm_;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_;
        Eigen::Vector3d previous_arm_joint_position_;
        int howManySteps_ = 0;
        int itr_n_ = 0;
        int step_counter_ = 0;
        /// variables about commands
        Eigen::Vector3d velocity_command_, eef_command_;
        bool setcmd_ = false;
        bool stance_ = false;
        double v_x_max_, v_y_max_, v_w_max_, t_x_, t_y_, t_w_, t_stance_;
        double r_=0., theta_=0., phi_=0., angle_=0., alpha_=0., beta_=0.;
        /// cfg
        double terminalRewardCoeff_ = -10.;
        double max_time_ = 0.;
        double curriculumFactor_ = 0., curriculumDecayFactor_ = 0., cf_ = 0.;
        double act_std_val_ = 0., clearance_ = 0.;
        /// foot
        int nFoot_;
        std::set<size_t> footIndices_;
        std::vector<size_t> footVec_;
        std::vector<size_t> footFrame_;
        Eigen::VectorXd last_foot_state_, current_foot_state_;
        Eigen::Vector4d t_a_, t_s_, t_a_i_, t_s_i_;
        raisim::Vec<3> RF_footPosition_, LF_footPosition_, RH_footPosition_, LH_footPosition_;
        Eigen::VectorXd foot_pos_height_, foot_vel_, grf_bin_;
        /// end-effector
        raisim::Vec<3> eef_position_rai_;
        raisim::Mat<3,3> eef_orientation_rai_;
        Eigen::Vector3d desired_eef_position_, desired_eef_position_w_, eef_position_init_w_;
        Eigen::VectorXd eef_position_, eef_position_init_, eef_orientation_, eef_orientation_init_;
        /// base and body
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, bodyOrientation_, robot_COM_;
        Eigen::Vector3d base_position_;
        /// history
        std::deque<Eigen::VectorXd> pTarget_history_, joint_position_history_, joint_velocity_history_, eef_position_history_, eef_orientation_history_;
        /// visual objects
        int n_balls_ = 9;
        double coordinatex_, coordinatey_;
        std::vector<raisim::Visuals *> desired_command_traj_;
        raisim::Visuals * object_;
        raisim::Visuals * object_eef_;

        std::normal_distribution<double> normDist_;
        thread_local static std::mt19937 gen_;
        double clean_randomizer;
    };
    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
}
