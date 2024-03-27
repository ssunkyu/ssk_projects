//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <set>
#include <cmath>
#include <deque>
#include <vector>
#include <string>
#include <random>
#include "../../RaisimGymEnv.hpp"
#include <fenv.h>
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
            /// Reward coefficients
            setSeed(env_id);
            for (int i = 0; i < 10; i++)
                clean_randomizer = Eigen::VectorXd::Random(1)[0];
            /// Reward coefficients
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            gen_.seed(seed);
            /// add objects
            world_ = std::make_unique<raisim::World>();
            handmal_ = world_->addArticulatedSystem(resourceDir_+"/handmal/urdf/handmal.urdf");
            handmal_->setName("handmal_");
            handmal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            world_->addGround();

            //config reward coefficients
            READ_YAML(double, linvelRewardCoeff_, cfg["linvelcoeff"]);
            READ_YAML(double, angvelRewardCoeff_, cfg["angvelcoeff"]);
            READ_YAML(double, manipulcmdRewardCoeff_, cfg["manipulcmdcoeff"]);
            READ_YAML(double, airtimeRewardCoeff_, cfg["airtimecoeff"]);
            READ_YAML(double, footslipRewardCoeff_, cfg["footslipcoeff"]);
            READ_YAML(double, footclRewardCoeff_, cfg["footclcoeff"]);
            READ_YAML(double, oriRewardCoeff_, cfg["oricoeff"]);
            READ_YAML(double, jtorqueRewardCoeff_, cfg["jtorquecoeff"]);
            READ_YAML(double, jpositionRewardCoeff_, cfg["jpositioncoeff"]);
            READ_YAML(double, jspeedRewardCoeff_, cfg["jspeedcoeff"]);
            READ_YAML(double, jaccRewardCoeff_, cfg["jacccoeff"]);
            READ_YAML(double, actsmooth1RewardCoeff_, cfg["actsmooth1coeff"]);
            READ_YAML(double, actsmooth2RewardCoeff_, cfg["actsmooth2coeff"]);
            READ_YAML(double, baseRewardCoeff_, cfg["basecoeff"]);
            READ_YAML(double, jpositionRewardCoeff2_, cfg["jpositioncoeff2"])
            READ_YAML(double, control_dt_, cfg["control_dt"]);
            READ_YAML(double, act_std_val, cfg["action_std"]);
            READ_YAML(double, curriculumFactor_, cfg["curriculumFactor"]);
            READ_YAML(double, curriculumDecayFactor_, cfg["curriculumDecay"]);
            cf = curriculumFactor_;

            /// get robot data
            gcDim_ = handmal_->getGeneralizedCoordinateDim();
            gvDim_ = handmal_->getDOF();
            nJoints_ = gvDim_ - 6; // 18-6=12

            /// indices of links that should not make contact with ground
            footIndices_.insert(handmal_->getBodyIdx("LF_SHANK"));footIndices_.insert(handmal_->getBodyIdx("RF_SHANK"));
            footIndices_.insert(handmal_->getBodyIdx("LH_SHANK"));footIndices_.insert(handmal_->getBodyIdx("RH_SHANK"));

            footVec_.push_back(handmal_->getBodyIdx("LF_SHANK"));footVec_.push_back(handmal_->getBodyIdx("RF_SHANK"));
            footVec_.push_back(handmal_->getBodyIdx("LH_SHANK"));footVec_.push_back(handmal_->getBodyIdx("RH_SHANK"));

            footFrame_.push_back(handmal_->getFrameIdxByName("LF_ADAPTER_TO_FOOT"));footFrame_.push_back(handmal_->getFrameIdxByName("RF_ADAPTER_TO_FOOT"));
            footFrame_.push_back(handmal_->getFrameIdxByName("LH_ADAPTER_TO_FOOT"));footFrame_.push_back(handmal_->getFrameIdxByName("RH_ADAPTER_TO_FOOT"));
            nFoot = 4;
//            footFrame_.push_back(anymal_->getFrameIdxByName("LF_shank_fixed_LF_FOOT"));footFrame_.push_back(anymal_->getFrameIdxByName("RF_shank_fixed_RF_FOOT"));
//            footFrame_.push_back(anymal_->getFrameIdxByName("LH_shank_fixed_LH_FOOT"));footFrame_.push_back(anymal_->getFrameIdxByName("RH_shank_fixed_RH_FOOT"));

            /// initialize containers
            gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
            gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
            pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);
            pTarget18_.setZero(nJoints_);
            qdes.setZero(gcDim_); commandDim_=6; command_.setZero(commandDim_);

            /// this is nominal configuration of anymal
            double hfe, kfe;
            hfe = 0.8; kfe = 1.2;
//            gc_init_ << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, hfe, -kfe, -0.03, hfe, -kfe, 0.03, -hfe, kfe, -0.03, -hfe, kfe, 0.0, -2.3, -1.57, 0.0, 2.0, 0.0;
            gc_init_ << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8, 0.0, -2.6, -1.57, 0.0, 2.0, 0.0;
//            gc_init_ << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8, 0.0, 2.76, -1.57, 0.0, 2.0, 0.0;
            // in gc_: head(3):base position, segment(3,4):base orientation, segment(7,12):leg, tail(6):arm
            // in gv_: head(3):base linearvelocity, segment(3,3):base angularvelocity, segment(6,12):leg, segment(18,3):large arm joint, tail(3):small arm joint
            /// set pd gains for leg
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero(); jointPgain.segment(18,3).setConstant(40.0); jointPgain.tail(3).setConstant(40.0);
            jointPgain.segment(6,12).setConstant(50.0); //default arm's pgain = 40.0, 15.0, default leg's pgain = 100.0
            jointDgain.setZero(); jointDgain.segment(18,3).setConstant(1.0); jointDgain.tail(3).setConstant(1.0);
            jointDgain.segment(6,12).setConstant(0.4); // defualt arm's dgain = 1.0, 0.5, default leg's dgain = 0.4
            handmal_->setPdGains(jointPgain, jointDgain);
            handmal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            // policy input dimenstion
            obDim_ = 243;

            // action
            actionDim_ = nJoints_;
            actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
            obDouble_.setZero(obDim_); invActionStd_.setZero(actionDim_);

            /// action & observation scaling
            actionMean_ = gc_init_.tail(nJoints_);
            // act_std_val = 0.3
//            actionStd_.setConstant(act_std_val);
            Eigen::VectorXd actionStd_leg;
            Eigen::VectorXd actionStd_arm;
            actionStd_leg.setZero(12); actionStd_arm.setZero(6);

            actionStd_leg.setConstant(act_std_val);
            actionStd_arm << 0.05, 0.05, 0.05, 0.05, 0.05, 0.05;

            actionStd_ << actionStd_leg,
                        actionStd_arm;

            Eigen::VectorXd invActionStd_leg;
            Eigen::VectorXd invActionStd_arm;
            invActionStd_leg.setZero(12); invActionStd_arm.setZero(6);

            invActionStd_leg.setConstant(act_std_val);
            invActionStd_arm << 20, 20, 20, 20, 20, 20;

            invActionStd_ << invActionStd_leg,
                            invActionStd_arm;

            last_contact.setZero(nFoot);
            grf_bin_obs.setZero(nFoot);
            last_swing.setZero(nFoot);
            last_foot_state.setZero(nFoot);
            eef_position_.setZero(3);
            eef_position_init_.setZero(3);
            eef_orientation_.setZero(4); // quaternion
            eef_orientation_init_.setZero(4);
            eef_position_init.setZero(3);

            step_counter = 0;
            v_x_max = 0.8 + 1.6/(1+exp(2));
            v_y_max = 0.4 + 0.8/(1+exp(2));
            v_w_max = 0.4 + 0.8/(1+exp(2));

            get_base_eef_position();
            sample_command();

            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer(8080);
                for (int i = 0; i < n_balls; i++) {
                    desired_command_traj.push_back(
                            server_->addVisualBox("desired_command_pos" + std::to_string(i), 0.1, 0.1, 0.1, 0, 1, 0));
                }
                object = server_->addVisualBox("manipulation_command_position",0.1,0.1,0.1, 0, 0,1);
                std::cout<<"server on. port: 8080"<<std::endl;
            }

            t_a.setZero(); t_s.setZero(); t_a_i.setConstant(world_->getWorldTime()); t_s_i.setConstant(world_->getWorldTime());
        }

        void get_base_eef_position(){
            raisim::Vec<3> eef_pos;
            handmal_->getFramePosition("kinova_joint_end_effector", eef_pos);
            eef_position_init_ = eef_pos.e();
            eef_position_init = eef_pos.e();

            raisim::Vec<4> quat;
            Eigen::Vector3d ppp;
            quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
            ppp[0] = gc_[0]; ppp[1] = gc_[1]; ppp[2] = gc_[2];
            raisim::quatToRotMat(quat, baseRot_);
            eef_position_init_ = baseRot_.e().transpose() * (eef_position_init_-ppp);
        }

        void sample_command() {
            std::uniform_real_distribution<> distrib(0,1);

            t_stance = distrib(gen_);
            t_x = distrib(gen_); t_y = distrib(gen_); t_w = distrib(gen_);

            if(t_stance < 0.15){
                command_.setZero(); stance = true;
                command_.tail(3) = cf * 0.2*Eigen::VectorXd::Random(3);
            }
            else{
                command_.setZero(); stance = false;
                command_[0] = -v_x_max/2 + 3*v_x_max*t_x/2;
                command_[1] = -v_y_max + 2*v_y_max*t_y;
                command_[2] = -v_w_max + 2*v_w_max*t_w;
            }
            desired_eef_position = command_.tail(3) + eef_position_init;
            desired_eef_position_ = command_.tail(3) + eef_position_init_;
        }


        void init() final {
        }

        void kill_server(){
            server_->killServer();
        }

        void reset() final{
            gc_ = gc_init_; gv_.setZero();
            gc_.segment(3,4) += 0.05*Eigen::VectorXd::Random(4); gc_.segment(3,4).normalize();
            gc_.segment(7,12) += 0.05*Eigen::VectorXd::Random(12); gv_.segment(6,12) = 0.5*Eigen::VectorXd::Random(12);
            gv_.head(1) = 0.2*Eigen::VectorXd::Random(1); gv_.segment(1,2) = 0.2*Eigen::VectorXd::Random(2); gv_.segment(3,3) = 0.2 *Eigen::VectorXd::Random(3);
//            gc_.tail(6) += 0.1*Eigen::VectorXd::Random(6); gv_.tail(6) = 1.0*Eigen::VectorXd::Random(6);
            handmal_->setState(gc_, gv_);

            qdes_history.clear();
            joint_position_history.clear();
            joint_velocity_history.clear();
            eef_position_history.clear();
            eef_orientation_history.clear();

            for (int i = 0; i < 50; i++){
                if (server_)    server_->lockVisualizationServerMutex();
                world_->integrate();
                if (server_)    server_->unlockVisualizationServerMutex();
            }

            for (int j = 0; j < 10; j++) {
            qdes_history.push_back(Eigen::VectorXd::Zero(18));
            joint_position_history.push_back(Eigen::VectorXd::Zero(18));
            joint_velocity_history.push_back(Eigen::VectorXd::Zero(18));
            eef_position_history.push_back(Eigen::VectorXd::Zero(3));
            eef_orientation_history.push_back(Eigen::VectorXd::Zero(4));
            }

            get_base_eef_position();
            sample_command();
            updateObservation();
            if (visualizable_) {
                if(not stance) {
                    object->setColor(1,0,0,1);
                    for (int i = 0; i < n_balls; i++) {
                        if (command_[2] != 0) {
                            coordinate_x = command_[0] / command_[2] * std::sin(command_[2] * i * 0.5)
                                           + command_[1] / command_[2] * (std::cos(command_[2] * i * 0.5) - 1);
                            coordinate_y = -command_[0] / command_[2] * (std::cos(command_[2] * i * 0.5) - 1)
                                           + command_[1] / command_[2] * std::sin(command_[2] * i * 0.5);
                        } else {
                            coordinate_x = command_[0] * i * 0.5;
                            coordinate_y = command_[1] * i * 0.5;
                        }
                        desired_command_traj[i]->setPosition({coordinate_x, coordinate_y, 0.2});
                    }
                }
                else{
                    object->setColor(0,0,1,1);
                    object->setPosition(eef_position_init);
                }
            }

            t_a.setZero(); t_s.setZero(); t_a_i.setConstant(world_->getWorldTime()); t_s_i.setConstant(world_->getWorldTime());
//            std::cout << command_[0] << "x" << std::endl;
//            std::cout << command_[1] << "y" << std::endl;
//            std::cout << command_[2] << "w" << std::endl;
        }

        float step(const Eigen::Ref<EigenVec>& action) final {
            pTarget18_ = action.cast<double>().cwiseProduct(actionStd_) + gc_init_.tail(18);
            pTarget_.tail(nJoints_) = pTarget18_;
            qdes = pTarget18_;

            handmal_->setPdTarget(pTarget_, vTarget_);

            auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);

            for(howManySteps_ = 0; howManySteps_ < loopCount; howManySteps_++){
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if(server_) server_->unlockVisualizationServerMutex();
            }

            updateObservation();

            // positive reward
            linvelReward_ = linvelRewardCoeff_ * exp(-((command_.head(2)-bodyLinearVel_.head(2)).squaredNorm()));
            angvelReward_ = angvelRewardCoeff_ * exp(-1.5*((command_.segment(2,1)-bodyAngularVel_.tail(1)).squaredNorm()));
            /// airtime
            manipulcmdReward_ = manipulcmdRewardCoeff_ * exp(-1.5*((desired_eef_position_-eef_position_).squaredNorm()/(desired_eef_position_-eef_orientation_init_).squaredNorm()));
//            manipulcmdReward_ = manipulcmdRewardCoeff_ * exp(-((command_.tail(3)-eff_position_)).squardNorm()/((command_.tail(3)-eff_position_init_).squardNorm()));
            // summation
            positive_reward = linvelReward_ + angvelReward_ + airtimeReward_ + manipulcmdReward_;

            VecDyn current_torque = handmal_->getGeneralizedForce();

            // negative reward
            /// foot clearance
            angle = pow(baseRot_[6],2) + pow(baseRot_[7],2) + pow(baseRot_[8]-1,2);
            oriReward_ = cf * oriRewardCoeff_ * angle;
            jtorqueReward_ = cf * jtorqueRewardCoeff_ * current_torque.squaredNorm();
            jpositionReward_ = cf * jpositionRewardCoeff_
                    * (gc_.segment(7,12)-gc_init_.segment(7,12)).squaredNorm();
//            if(stance) jpositionReward2_ = 0;
//            else jpositionReward2_ = cf * jpositionRewardCoeff2_
//                    * (gc_.tail(6)-gc_init_.tail(6)).squaredNorm();
            jspeedReward_ = cf * jspeedRewardCoeff_
                    * (gv_.tail(18)).squaredNorm();
            jaccReward_ = cf * jaccRewardCoeff_
                    * (gv_.tail(18)-joint_velocity_history[joint_velocity_history.size()-1]).squaredNorm();
            actsmooth1Reward_ = cf * actsmooth1RewardCoeff_
                    * (qdes - qdes_history[qdes_history.size()-1]).squaredNorm();
            actsmooth2Reward_ = cf * actsmooth2RewardCoeff_
                    * (qdes - 2 * qdes_history[qdes_history.size()-1] + qdes_history[qdes_history.size()-2]).squaredNorm();
            baseReward_ = cf * baseRewardCoeff_
                    * (0.8* pow(bodyLinearVel_[2],2) + 0.2*abs(bodyAngularVel_[0]) + 0.2*abs(bodyAngularVel_[1]));
            ///summation
            negative_reward = oriReward_ + jtorqueReward_ + jpositionReward_ + jspeedReward_ + footslipReward_
                    + jaccReward_ + baseReward_ + footclReward_ + actsmooth1Reward_ + actsmooth2Reward_
                    + jpositionReward2_;

            cumulative_reward = positive_reward * exp(0.2 * negative_reward);

            auto jterr = qdes_history[qdes_history.size()-1] - gc_.tail(18);
            auto vel_ = gv_.tail(18);

            qdes_history.push_back(qdes);
            joint_position_history.push_back(jterr);
            joint_velocity_history.push_back(vel_);
            eef_position_history.push_back(eef_position_);
            eef_orientation_history.push_back(eef_orientation_);

//    std::cout << linvelReward_ << " 1" << std::endl;
//    std::cout << angvelReward_ << " 2" << std::endl;
//    std::cout << airtimeReward_ << " 3" << std::endl;
//    std::cout << manipulcmdReward_ << " 4" << std::endl;
//    std::cout << footslipReward_ << " 5" << std::endl;
//    std::cout << oriReward_ << " 6" << std::endl;
//    std::cout << jtorqueReward_ << " 7" << std::endl;
//    std::cout << jpositionReward_ << " 8" << std::endl;
//    std::cout << jspeedReward_ << " 9" << std::endl;
//    std::cout << jaccReward_ << " 10" << std::endl;
//    std::cout << actsmooth1Reward_ << " 11" << std::endl;
//    std::cout << actsmooth2Reward_ << " 12" << std::endl;
//    std::cout << baseReward_ << " 13" << std::endl;
//    std::cout << positive_reward << " 14" << std::endl;
//    std::cout << negative_reward << " 15" << std::endl;
//    std::cout << cumulative_reward << " 16" << std::endl;
            return float(cumulative_reward/howManySteps_);
        }


        void observe(Eigen::Ref<EigenVec> ob) final {
            obDouble_ <<
                      command_, // command 6
                    bodyOrientation_, // body orientation 3
                    bodyAngularVel_, // body angular velocity 3
                    gc_.tail(18), // joint state1: joint angle 18
                    gv_.tail(18), // joint state2: joint velocity 18 // 48
                    qdes_history[qdes_history.size()-1], // desired joint position = action history 18*2=36 // 84
                    qdes_history[qdes_history.size()-2],
                    joint_position_history[joint_position_history.size()-1], // joint history1: joint angle 18*3=54 // 138
                    joint_position_history[joint_position_history.size()-2],
                    joint_position_history[joint_position_history.size()-3],
                    joint_velocity_history[joint_velocity_history.size()-1], // joint history2: joint velocity 18*3=54 // 192
                    joint_velocity_history[joint_velocity_history.size()-2],
                    joint_velocity_history[joint_velocity_history.size()-3],
                    robot_COM-RF_footPosition.e(), // relative foot position 3*4=12 /// 128 127 126 // 204
                    robot_COM-LF_footPosition.e(), /// 131 130 129
                    robot_COM-RH_footPosition.e(), /// 134 133 132
                    robot_COM-LH_footPosition.e(), /// 137 136 135
                    bodyLinearVel_, /// body linear velocity 3 /// 140 139 138 // 207
                    current_foot_state; /// contact foot binary state 4 /// 144 143 142 141 //211
                    foot_pos_height, // 215
                    eef_position_history[eef_position_history.size()-1], // 3*3=9 224
                    eef_position_history[eef_position_history.size()-2],
                    eef_position_history[eef_position_history.size()-3],
                    eef_orientation_history[eef_orientation_history.size()-1], // 4*3=12 236
                    eef_orientation_history[eef_orientation_history.size()-2],
                    eef_orientation_history[eef_orientation_history.size()-3],
                    eef_position_, // 239
                    eef_orientation_; // 243
            /// convert it to float
            ob = obDouble_.cast<float>();
//    std::cout << baseRot_.e().row(2).transpose() << "baserot_" << std::endl;
//    std::cout << bodyAngularVel_ << "bodyang" << std::endl; // body angular velocity 3
//    std::cout << gc_.tail(18) << "gc_" << std::endl;
//    std::cout << gv_.tail(18) << "gv_" << std::endl; // joint state2: joint velocity 18
//    std::cout << qdes_history[qdes_history.size()-1] << "qdes_his1" << std::endl;
//    std::cout << qdes_history[qdes_history.size()-2] << "qdes_his2" << std::endl;
//    std::cout << joint_position_history[joint_position_history.size()-1] << "jp_his1" << std::endl;
//    std::cout << joint_position_history[joint_position_history.size()-2] << "jp_his2" << std::endl;
//    std::cout << joint_position_history[joint_position_history.size()-3] << "jp_his3" << std::endl;
//    std::cout << joint_velocity_history[joint_velocity_history.size()-1] << "jv_his1" << std::endl;
//    std::cout << joint_velocity_history[joint_velocity_history.size()-2] << "jv_his2" << std::endl;
//    std::cout << joint_velocity_history[joint_velocity_history.size()-3] << "jv_his3" << std::endl;
//    std::cout << robot_COM-RF_footPosition.e() << "rfp" << std::endl;
//    std::cout << robot_COM-LF_footPosition.e() << "lfp" << std::endl;
//    std::cout << robot_COM-RH_footPosition.e() << "rhp" << std::endl;
//    std::cout << robot_COM-LH_footPosition.e() << "lhp" << std::endl;
//    std::cout << bodyLinearVel_ << "bodylin" <<std::endl;
//    std::cout << contact_state << "contact" << std::endl;
        }

        void updateObservation() {
            handmal_->getState(gc_, gv_);

            raisim::Vec<4> quat;
            quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
            raisim::quatToRotMat(quat, baseRot_);
            bodyLinearVel_ = baseRot_.e().transpose() * gv_.segment(0, 3);
            bodyAngularVel_ = baseRot_.e().transpose() * gv_.segment(3, 3);

            robot_COM = handmal_->getCOM().e();
            handmal_->getFramePosition(footFrame_[0], LF_footPosition);
            handmal_->getFramePosition(footFrame_[1], RF_footPosition);
            handmal_->getFramePosition(footFrame_[2], LH_footPosition);
            handmal_->getFramePosition(footFrame_[3], RH_footPosition);

            quatToEuler(quat, bodyOrientation_);

            // Foot contact events
            Eigen::VectorXd grf;
            Eigen::VectorXd grf_bin;
            Eigen::VectorXd swing_bin;
            Eigen::VectorXd foot_vel;
            Eigen::VectorXd foot_pos_bin;
            Eigen::VectorXd foot_pos_err;
            raisim::Vec<3> footVelocity;
            raisim::Vec<3> footPosition;
            raisim::Vec<3> eef_position;
            raisim::Mat<3,3> eef_orientation;
            Eigen::Matrix3d eef_orientation_matrix;
            Eigen::Quaternionf quat_eef_orientation;
            // raisim::Vec<3> net_impulse;
            double clearance = 0.02;
            grf.setZero(nFoot);
            grf_bin.setZero(nFoot);
            swing_bin.setOnes(nFoot);
            foot_vel.setZero(nFoot);
            foot_pos_height.setZero(nFoot);
            foot_pos_bin.setZero(nFoot);
            foot_pos_err.setZero(nFoot);
            current_foot_state = last_foot_state;

            handmal_->getFramePosition("kinova_joint_end_effector", eef_position);
            handmal_->getFrameOrientation("kinova_joint_end_effector", eef_orientation);

            eef_position_ = baseRot_.e().transpose() * eef_position.e();
            eef_orientation_matrix = baseRot_.e().transpose() * eef_orientation.e();

            Eigen::Quaterniond quat_from_matrix(eef_orientation_matrix);
            eef_orientation_(0) = quat_from_matrix.w();
            eef_orientation_(1) = quat_from_matrix.x();
            eef_orientation_(2) = quat_from_matrix.y();
            eef_orientation_(3) = quat_from_matrix.z();

            for (int footIdx_i = 0; footIdx_i < nFoot; footIdx_i++)
            {
                auto footIndex = footVec_[footIdx_i];
                // check for contact event
                for (auto &contact : handmal_->getContacts())
                {
                    if (contact.skip())
                        continue;
                    if (footIndex == contact.getlocalBodyIndex())
                    {
                        auto impulse_i = (contact.getContactFrame().e() * contact.getImpulse().e()).norm();
                        if (impulse_i > 0) // 땅에 닿았을 때
                        {
                            grf_bin[footIdx_i] = 1.0;
                            swing_bin[footIdx_i] = 0.0;
                        }
                    }
                }
                // measure foot velocity
                handmal_->getFrameVelocity(footFrame_[footIdx_i], footVelocity);
                handmal_->getFramePosition(footFrame_[footIdx_i], footPosition);
                foot_vel[footIdx_i] = footVelocity.squaredNorm();
                foot_pos_bin[footIdx_i] = (double)(footPosition[2] > clearance);
                foot_pos_height[footIdx_i] = footPosition[2];
            }

            airtimeReward_ = 0;
            footclReward_ = 0;
            footslipReward_ = 0;

            for (int fi = 0; fi < nFoot; fi++) {
                if (foot_pos_bin[fi] == 1)
                    current_foot_state[fi] = 1.0; // 0이 땅에 닿았다, 1.0이 땅에서 떨어진 상태
                else if (swing_bin[fi] == 0)
                    current_foot_state[fi] = 0.0;
                // change contact
                if ((current_foot_state[fi] + last_foot_state[fi]) == 1.0) {
                    if ((current_foot_state[fi]) == 1.0) {
                        t_a_i[fi] = world_->getWorldTime();
                        t_s[fi] = world_->getWorldTime() - t_s_i[fi];
                    } else {
                        t_s_i[fi] = world_->getWorldTime();
                        t_a[fi] = world_->getWorldTime() - t_a_i[fi];
                    }
                }
                if (stance) {
                    airtimeReward_ += cf * airtimeRewardCoeff_ * std::clamp(t_s[fi] - t_a[fi], -0.3, 0.3);
                }
                else {
                    if (std::max(t_a[fi], t_s[fi]) < 0.25) {
                        airtimeReward_ += airtimeRewardCoeff_ * std::min(std::max(t_a[fi], t_s[fi]), 0.2);
                    }
                }
                footclReward_ += cf * footclRewardCoeff_ * pow((foot_pos_height[fi]-0.05),2) * pow(foot_vel[fi],0.25);
                footslipReward_ += cf * footslipRewardCoeff_ * grf_bin[fi] * foot_vel[fi];
            }
            last_foot_state = current_foot_state;
        }
        bool isTerminalState(float &terminalReward) final
        {
            terminalReward = float(terminalRewardCoeff_);
            for(auto& contact: handmal_->getContacts())
                if(footIndices_.find(contact.getlocalBodyIndex())==footIndices_.end())
                    return true;

//            /// 0: roll 1: pitch 2: yaw
//            if (abs(bodyOrientation_[0]) > (0.8 /cf) || abs(bodyOrientation_[2]) > (0.8/cf))
//            {
//                return true;
//            }
            terminalReward = 0.f;
            return false;
        }

        void curriculumUpdate(double& itr_n) final {
            curriculumFactor_ = std::pow(curriculumFactor_, curriculumDecayFactor_);
            cf = curriculumFactor_;
            v_x_max = 0.8 + 1.6/(1+exp(-0.002*(itr_n-1000)));
            v_y_max = 0.4 + 0.8/(1+exp(-0.002*(itr_n-1000)));
            v_w_max = 0.4 + 0.8/(1+exp(-0.002*(itr_n-1000)));
        }

        void setSeed(int seed)
        {
            std::srand(seed);
            for (int i = 0; i < 10; i++)
                clean_randomizer = Eigen::VectorXd::Random(1)[0];
        }

    private:
        int gcDim_, gvDim_, nJoints_, commandDim_;
        int howManySteps_ = 0;
        bool visualizable_ = false;

        raisim::ArticulatedSystem* handmal_;
        raisim::Mat<3,3> baseRot_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget18_, vTarget_, qdes;
        Eigen::VectorXd current_foot_state;

        double terminalRewardCoeff_ = -10.;
        double linvelRewardCoeff_ = 0., linvelReward_ = 0.;
        double angvelRewardCoeff_ = 0., angvelReward_ = 0.;
        double airtimeRewardCoeff_ = 0., airtimeReward_ = 0.;
        double manipulcmdRewardCoeff_ = 0., manipulcmdReward_ = 0.;
        double footslipRewardCoeff_ = 0., footslipReward_ = 0.;
        double footclRewardCoeff_ = 0., footclReward_ = 0.;
        double oriRewardCoeff_ = 0., oriReward_ = 0.;
        double jtorqueRewardCoeff_ = 0., jtorqueReward_ = 0.;
        double jpositionRewardCoeff_ = 0., jpositionReward_ = 0.;
        double jspeedRewardCoeff_ = 0., jspeedReward_ = 0.;
        double jaccRewardCoeff_ = 0., jaccReward_ = 0.;
        double actsmooth1RewardCoeff_ = 0., actsmooth1Reward_ = 0.;
        double actsmooth2RewardCoeff_ = 0., actsmooth2Reward_ = 0.;
        double baseRewardCoeff_ = 0., baseReward_ = 0.;
        double jpositionRewardCoeff2_ = 0., jpositionReward2_ =0.;
        double control_dt_= 0.;
        double curriculumFactor_ = 0., curriculumDecayFactor_ = 0., cf = 0.;
        double act_std_val = 0.;

        std::set<size_t> footIndices_;
        std::vector<size_t> footVec_;
        std::vector<size_t> footFrame_;

        Eigen::VectorXd actionMean_, invActionStd_, actionStd_, obDouble_, eef_position_, eef_position_init_, eef_orientation_, eef_orientation_init_, eef_position_init;
        Eigen::VectorXd last_contact, grf_bin_obs, last_swing, last_foot_state, foot_pos_height;
        Eigen::VectorXd command_;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, bodyOrientation_;
        Eigen::Vector3d robot_COM, desired_eef_position_, desired_eef_position;
        Eigen::Vector4d contact_state, t_a, t_s, t_a_i, t_s_i;

        std::deque<Eigen::VectorXd> qdes_history, joint_position_history, joint_velocity_history, eef_position_history, eef_orientation_history;
        std::vector<raisim::Visuals *> desired_command_traj;
        raisim::Visuals * object;
        double positive_reward = 0, negative_reward = 0, cumulative_reward = 0, angle = 0;
        double v_x_max, v_y_max, v_w_max, t_x, t_y, t_w, t_stance;
        int step_counter = 0, nFoot = 4;

        raisim::Vec<3> RF_footPosition, LF_footPosition, RH_footPosition, LH_footPosition;

        std::normal_distribution<double> normDist_;
        thread_local static std::mt19937 gen_;
        bool stance = false;
        int n_balls = 9;
        double clean_randomizer;
        double coordinate_x, coordinate_y;
    };
    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
}