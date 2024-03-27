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
void quatPower(const raisim::Vec<4>& q, double t, raisim::Vec<4>& qt) {
    double theta = 2.0 * std::acos(q[0]);

    // If theta is very small, the quaternion is close to the identity quaternion.
    // In this case, just return the identity quaternion to avoid division by zero.
    if (std::abs(theta) < 1e-6) {
        qt[0] = 1.0;
        qt[1] = 0.0;
        qt[2] = 0.0;
        qt[3] = 0.0;
        return;
    }

    double theta_t = theta * t;
    double sin_theta_t_over_sin_theta = std::sin(theta_t) / std::sin(theta);

    qt[0] = std::cos(theta_t);
    qt[1] = q[1] * sin_theta_t_over_sin_theta;
    qt[2] = q[2] * sin_theta_t_over_sin_theta;
    qt[3] = q[3] * sin_theta_t_over_sin_theta;
}

namespace raisim {

    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable, int env_id) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {
            /// set seed
            setSeed(env_id);
            for (int i = 0; i < 10; i++)
                clean_randomizer = Eigen::VectorXd::Random(1)[0];
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            gen_.seed(seed);
            /// add objects
            world_ = std::make_unique<raisim::World>();
            handmal_ = world_->addArticulatedSystem(resourceDir_+"/handmal_test/urdf/handmal_test.urdf"); //handmal_test = last 3 arm joint is locked.
//            handmal_ = world_->addArticulatedSystem(resourceDir_+"/handmal/urdf/handmal.urdf");
            handmal_->setName("handmal_");
            handmal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            world_->addGround();

            /// config reward coefficients
            rewards_.initializeFromConfigurationFile (cfg["reward"]);
            READ_YAML(double, max_time_, cfg["max_time"]); READ_YAML(double, simulation_dt_, cfg["simulation_dt"]);
            READ_YAML(double, control_dt_, cfg["control_dt"]); READ_YAML(double, actionStd_val_, cfg["action_std"]);
            READ_YAML(double, curriculumFactor_, cfg["curriculumFactor"]); READ_YAML(double, curriculumDecayFactor_, cfg["curriculumDecay"]);
            READ_YAML(double, eef_linvel_limit_, cfg["eef_linvel"]); READ_YAML(double, eef_angvel_limit_, cfg["eef_angvel"]);
            READ_YAML(double, clearance_, cfg["clearance"]);
            cf_ = curriculumFactor_;

            /// get robot data
            gcDim_ = handmal_->getGeneralizedCoordinateDim();
            gvDim_ = handmal_->getDOF();
            nJoints_ = gvDim_ - 6; // 21-6=15

            /// indices of links that should not make contact with ground
            footIndices_.insert(handmal_->getBodyIdx("LF_SHANK"));footIndices_.insert(handmal_->getBodyIdx("RF_SHANK"));
            footIndices_.insert(handmal_->getBodyIdx("LH_SHANK"));footIndices_.insert(handmal_->getBodyIdx("RH_SHANK"));
            footVec_.push_back(handmal_->getBodyIdx("LF_SHANK"));footVec_.push_back(handmal_->getBodyIdx("RF_SHANK"));
            footVec_.push_back(handmal_->getBodyIdx("LH_SHANK"));footVec_.push_back(handmal_->getBodyIdx("RH_SHANK"));
            footFrame_.push_back(handmal_->getFrameIdxByName("LF_ADAPTER_TO_FOOT"));footFrame_.push_back(handmal_->getFrameIdxByName("RF_ADAPTER_TO_FOOT"));
            footFrame_.push_back(handmal_->getFrameIdxByName("LH_ADAPTER_TO_FOOT"));footFrame_.push_back(handmal_->getFrameIdxByName("RH_ADAPTER_TO_FOOT"));
            nFoot_ = 4;
//            footFrame_.push_back(anymal_->getFrameIdxByName("LF_shank_fixed_LF_FOOT"));footFrame_.push_back(anymal_->getFrameIdxByName("RF_shank_fixed_RF_FOOT"));
//            footFrame_.push_back(anymal_->getFrameIdxByName("LH_shank_fixed_LH_FOOT"));footFrame_.push_back(anymal_->getFrameIdxByName("RH_shank_fixed_RH_FOOT"));

            /// initialize containers
            gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
            gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
            pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);
            pTarget15_.setZero(nJoints_);
            gc_init_ << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8, 0.0, -2.6, 0;
//            gc_init_ << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8, 0.0, -2.6, -1.57, 0.0, 2.0, 0.0;
            // in gc_: head(3):base position, segment(3,4):base orientation, segment(7,12):leg, tail(6):arm
            // in gv_: head(3):base linear_velocity, segment(3,3):base angular_velocity, segment(6,12):leg, tail(6):arm
            
            /// set pd gains for leg
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero(); jointPgain.tail(3).setConstant(50.0);
            jointPgain.segment(6,12).setConstant(50.0);
            jointDgain.setZero(); jointDgain.tail(3).setConstant(4.0);
            jointDgain.segment(6,12).setConstant(0.4);
            handmal_->setPdGains(jointPgain, jointDgain);
            handmal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// policy input dimenstion
            obDim_ = 194;
            obDouble_.setZero(obDim_);

            /// action & observation scaling
            actionDim_ = nJoints_;
            actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
            actionMean_ = gc_init_.tail(nJoints_);

            actionStd_leg_.setZero(12); actionStd_arm_.setZero(3);
            actionStd_leg_.setConstant(actionStd_val_);
            actionStd_arm_ << 0.05, 0.05, 0.05;
            actionStd_ << actionStd_leg_, actionStd_arm_;

            /// if visualizable_, add visual sphere
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer(8080);
                obj_ee_curr_command_ = server_->addVisualSphere("manipulation_command_position",0.03,0.1,0.1, 0.1, 1);
                obj_ee_curr_position_ = server_->addVisualSphere("eef_position",0.03,0.1,0.1, 0.1, 1);
                obj_ee_goal_command_ = server_->addVisualSphere("end_eef_position",0.03,0.1,0.1, 0.1, 1);
//                object_workspace_final_ = server_->addVisualSphere("eef_workspace_final",0.1,0,0, 1.0, 0.1);
//                object_workspace_curr_ = server_->addVisualSphere("eef_worksapce",0.1, 1,0,0,0.1);
                std::cout<<"server on. port: 8080"<<std::endl;
            }
        }

        void init() final {
            itr_n_ = 0;
            handmal_->setMass(handmal_->getBodyIdx("kinova_link_1"),0);
            handmal_->setMass(handmal_->getBodyIdx("kinova_link_2"),0);
            handmal_->setMass(handmal_->getBodyIdx("kinova_link_3"),0);
        }

        void kill_server(){
            server_->killServer();
        }

        void reset() final{
            step_counter_ = 0;
            /// set initial pose
            gc_ = gc_init_; gv_.setZero();
            gc_.segment(3,4) += 0.2*Eigen::VectorXd::Random(4); gc_.segment(3,4).normalize();
            gc_.segment(7,12) += 0.2*Eigen::VectorXd::Random(12);gv_.segment(6,12) = 0.25*Eigen::VectorXd::Random(12);
            gv_.head(1) = 0.3*Eigen::VectorXd::Random(1); gv_.segment(1,2) = 0.15*Eigen::VectorXd::Random(2);
            gv_.segment(3,3) = 0.2*Eigen::VectorXd::Random(3);
            gc_.tail(3) += 0.3*Eigen::VectorXd::Random(3); gv_.tail(3) = 0.15*Eigen::VectorXd::Random(3);
            handmal_->setState(gc_, gv_);
            pTarget_.tail(nJoints_) = Eigen::VectorXd::Zero(15) + actionMean_;
            handmal_->setPdTarget(pTarget_, vTarget_);

            for (int i = 0; i < 50; i++){
                if (server_)    server_->lockVisualizationServerMutex();
                world_->integrate();
                if (server_)    server_->unlockVisualizationServerMutex();
            }

            /// clear history
            pTarget_history_.clear();
            joint_position_history_.clear();
            joint_velocity_history_.clear();
            ee_position_history_.clear();
            ee_quat_history_.clear();
            for (int j = 0; j < 10; j++) {
                pTarget_history_.push_back(Eigen::VectorXd::Zero(15));
                joint_position_history_.push_back(Eigen::VectorXd::Zero(15));
                joint_velocity_history_.push_back(Eigen::VectorXd::Zero(15));
                ee_position_history_.push_back(Eigen::VectorXd::Zero((3)));
                ee_quat_history_.push_back(Eigen::VectorXd::Zero(4));
            }

            /// reset foot states
            last_foot_state_.setZero(nFoot_);
            current_foot_state_.setZero(nFoot_);
            grf_bin_.setZero(nFoot_);
            foot_vel_.setZero(nFoot_);
            foot_pos_height_.setZero(nFoot_);
            t_a_.setZero(); t_s_.setZero(); t_a_i_.setConstant(world_->getWorldTime()); t_s_i_.setConstant(world_->getWorldTime());

            updateObservation();
            _set_ee_goal_pose();
            _generate_ee_traj();

            ee_init_position_ = ee_position_;
            ee_final_goal_center << ee_init_position_[0] + 1.5,
                        ee_init_position_[1],
                        ee_init_position_[2] - 0.5; // cartesian world frame

            previous_arm_joint_position_ = gc_.tail(3);

            /// visualization end effector position and goal
            if (visualizable_) {
                obj_ee_curr_command_->setColor(0,0,1,1);
                obj_ee_curr_position_->setColor(0,0,1,1);
                obj_ee_goal_command_->setColor(1,0,1,1);

                obj_ee_curr_command_->setPosition(ee_curr_commnad_[step_counter_].head(3));
                obj_ee_curr_position_->setPosition(ee_position_);
                obj_ee_goal_command_->setPosition(ee_goal_position_);

//                object_workspace_curr_->setPosition(ee_goal_center);
//                object_workspace_curr_->setSphereSize(0.35*cf_);
//                object_workspace_final_->setPosition(ee_final_goal_center);
//                object_workspace_final_->setSphereSize(0.35);

            }
        }

        float step(const Eigen::Ref<EigenVec>& action) final {
            pTargetleg_ = action.cast<double>().head(12).cwiseProduct(actionStd_leg_) + gc_init_.segment(7,12);
            pTargetarm_ = action.cast<double>().tail(3).cwiseProduct(actionStd_arm_) + previous_arm_joint_position_;
            pTarget15_ << pTargetleg_, pTargetarm_;
//            pTarget15_ = action.cast<double>().cwiseProduct(actionStd_) + gc_init_.tail(15);
            pTarget_.tail(nJoints_) = pTarget15_;

            handmal_->setPdTarget(pTarget_, vTarget_);

            auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);

            for(howManySteps_ = 0; howManySteps_ < loopCount; howManySteps_++){
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if(server_) server_->unlockVisualizationServerMutex();
            }

            updateObservation();
            /// compute rewards
            rewards_.reset();
            VecDyn current_torque = handmal_->getGeneralizedForce();

            for (int fi = 0; fi < nFoot_; fi++){
                rewards_.record("airtime",std::clamp(t_s_[fi] - t_a_[fi],-0.3,0.3),true);
                rewards_.record("footclearance",pow((foot_pos_height_[fi]-clearance_),2) * pow(foot_vel_[fi],0.25),true); // clearance = 0.07
                rewards_.record("footslip", grf_bin_[fi] * foot_vel_[fi],true);
            }

            // positive rewards
            if(step_counter_) {
                rewards_.record("cmdposition",
                exp(-((ee_curr_commnad_[step_counter_ - 1].head(3) - ee_position_).squaredNorm()/pow(0.04/cf_,2))));
            }

            // negative rewards
            rewards_.record("jposition",
            (gc_.tail(3)-gc_init_.tail(3)).squaredNorm()); // position(15)->tend not to bend legs
            rewards_.record("jtorque",
            current_torque.e().tail(15).squaredNorm()); // torque(15)->rotate body 90 degree to reduce leg torque
            rewards_.record("jspeed",
            (gv_.tail(15)).squaredNorm());
            rewards_.record("jacc",
            (gv_.tail(15)-joint_velocity_history_[joint_velocity_history_.size()-1]).squaredNorm());
            rewards_.record("actsmooth1",
            (pTarget15_ - pTarget_history_[pTarget_history_.size()-1]).squaredNorm());
            rewards_.record("actsmooth2",
             (pTarget15_ - 2 * pTarget_history_[pTarget_history_.size()-1] + pTarget_history_[pTarget_history_.size()-2]).squaredNorm());
            rewards_.record("base",
            (0.8* pow(bodyLinearVel_[2],2) + 0.2*abs(bodyAngularVel_[0]) + 0.2*abs(bodyAngularVel_[1])));

            pTarget_history_.push_back(pTarget15_);
            joint_position_history_.push_back(pTarget_history_[pTarget_history_.size()-1] - gc_.tail(15));
            joint_velocity_history_.push_back(gv_.tail(15));
            ee_position_history_.push_back(baseRot_.e().transpose() * (ee_position_ - gc_.head(3)));
            ee_quat_history_.push_back(base_quat_ee_rai_.e());

            previous_arm_joint_position_ = gc_.tail(3);

            float positive_reward, negative_reward;
            positive_reward = rewards_["cmdposition"] + rewards_["airtime"];
            negative_reward = cf_ * (rewards_["jtorque"] + rewards_["jspeed"] + rewards_["jacc"]\
             + rewards_["actsmooth1"] + rewards_["actsmooth2"] + rewards_["base"]
             + rewards_["footclearance"] + rewards_["footslip"]);

             step_counter_++;

            if (visualizable_) {
                if(step_counter_==10){
                    obj_ee_curr_command_->setColor(0,1,0,1);
                    obj_ee_curr_position_->setColor(1,0,0,1);
                }
                if(step_counter_%10==0) {
                    obj_ee_curr_command_->setPosition(ee_curr_commnad_[step_counter_].head(3));
                }
                obj_ee_curr_position_->setPosition((ee_position_));
            }

            return float(positive_reward*exp(0.2*negative_reward)/howManySteps_);
        }


        void observe(Eigen::Ref<EigenVec> ob) final {
            obDouble_ <<
                    bodyOrientation_, // body orientation 3
                    gc_.tail(15), // joint state1: joint angle 15
                    gv_.tail(15), // joint state2: joint velocity 15 // 33
                    pTarget_history_[pTarget_history_.size()-1], // desired joint position = action history 15*2=30 // 63
                    pTarget_history_[pTarget_history_.size()-3],
                    joint_position_history_[joint_position_history_.size()-1], // joint history1: joint angle 15*3=45 // 108
                    joint_position_history_[joint_position_history_.size()-3],
                    joint_position_history_[joint_position_history_.size()-5],
                    joint_velocity_history_[joint_velocity_history_.size()-1], // joint history2: joint velocity 15*3=45 // 153
                    joint_velocity_history_[joint_velocity_history_.size()-3],
                    joint_velocity_history_[joint_velocity_history_.size()-5],
                    baseRot_.e().transpose() * (ee_position_ - gc_.head(3)), // end effector position (in body frame) 3 156
                    base_quat_ee_rai_.e(), // end effector quaterninon (in body frame) 4 160
                    ee_position_history_[ee_position_history_.size()-1], // end effector position history: 3*3=9 169
                    ee_position_history_[ee_position_history_.size()-3],
                    ee_position_history_[ee_position_history_.size()-5],
                    ee_quat_history_[ee_quat_history_.size()-1], // end effector orientation(quaternion): 4*3=12 181
                    ee_quat_history_[ee_quat_history_.size()-3],
                    ee_quat_history_[ee_quat_history_.size()-5],
                    current_foot_state_, // 4
                    baseRot_.e().transpose() * (ee_curr_commnad_[step_counter_] - gc_.head(3)), // 3*3=9
                    baseRot_.e().transpose() * (ee_curr_commnad_[step_counter_+1] - gc_.head(3)),
                    baseRot_.e().transpose() * (ee_curr_commnad_[step_counter_+2] - gc_.head(3)); // 194
            /// convert it to float
            ob = obDouble_.cast<float>();
        }

        void updateObservation() {
            handmal_->getState(gc_, gv_);
            _get_ee_curr_pose();

            quat_base_rai[0] = gc_[3]; quat_base_rai[1] = gc_[4]; quat_base_rai[2] = gc_[5]; quat_base_rai[3] = gc_[6];
            raisim::quatToRotMat(quat_base_rai, baseRot_);
            bodyLinearVel_ = baseRot_.e().transpose() * gv_.segment(0, 3);
            bodyAngularVel_ = baseRot_.e().transpose() * gv_.segment(3, 3);

            quatToEuler(quat_base_rai, bodyOrientation_);
            quatMul(quat_base_rai, ee_quat_rai_, base_quat_ee_rai_);

            // Foot contact events
            Eigen::VectorXd swing_bin, foot_pos_bin;
            raisim::Vec<3> footVelocity;
            raisim::Vec<3> footPosition;
            swing_bin.setOnes(nFoot_);
            foot_pos_bin.setZero(nFoot_);

            for (int footIdx_i = 0; footIdx_i < nFoot_; footIdx_i++)
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
                        if (impulse_i > 0) // if contact with ground
                        {
                            grf_bin_[footIdx_i] = 1.0;
                            swing_bin[footIdx_i] = 0.0;
                        }
                    }
                }
                // measure foot position & velocity
                handmal_->getFrameVelocity(footFrame_[footIdx_i], footVelocity);
                handmal_->getFramePosition(footFrame_[footIdx_i], footPosition);
                foot_pos_bin[footIdx_i] = (double)(footPosition[2] > clearance_);
                foot_vel_[footIdx_i] = footVelocity.squaredNorm();
                foot_pos_height_[footIdx_i] = footPosition[2];
            }

            for (int fi = 0; fi < nFoot_; fi++){
                if (foot_pos_bin[fi] == 1)
                    current_foot_state_[fi] = 1.0; // 0 = contact, 1 = ~contact
                else if (swing_bin[fi] == 0)
                    current_foot_state_[fi] = 0.0;
                // if contact changes, update t_s, t_a
                if ((current_foot_state_[fi] + last_foot_state_[fi]) == 1.0){
                    if ((current_foot_state_[fi]) == 1.0 ){
                        t_a_i_[fi] = world_->getWorldTime();
                        t_s_[fi] = world_->getWorldTime() - t_s_i_[fi];
                    }
                    else{
                        t_s_i_[fi] = world_->getWorldTime();
                        t_a_[fi] = world_->getWorldTime() - t_a_i_[fi];
                    }
                }
            }
            last_foot_state_ = current_foot_state_;
        }

        bool isTerminalState(float &terminalReward) final
        {
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
            /// arm mass curriculum
            if(itr_n_>1000) {
                handmal_->setMass(handmal_->getBodyIdx("kinova_link_1"),
                                  handmal_->getMass(handmal_->getBodyIdx("kinova_link_1")) * pow(cf_,1/(pow(curriculumFactor_,1000))));
                handmal_->setMass(handmal_->getBodyIdx("kinova_link_2"),
                                  handmal_->getMass(handmal_->getBodyIdx("kinova_link_2")) * pow(cf_,1/(pow(curriculumFactor_,1000))));
                handmal_->setMass(handmal_->getBodyIdx("kinova_link_3"),
                                  handmal_->getMass(handmal_->getBodyIdx("kinova_link_3")) * pow(cf_,1/(pow(curriculumFactor_,1000))));
            }
        }

        void setSeed(int seed){
            std::srand(seed);
            for (int i = 0; i < 10; i++)
                clean_randomizer = Eigen::VectorXd::Random(1)[0];
        }

        void getTorque(Eigen::Ref<EigenVec> torque) final {
            /// get joint torques
            torque = handmal_->getGeneralizedForce().e().tail(15).cast<float>();
        }

        void setCommand(Eigen::Ref<EigenVec>& command) final {
            /// for test
            setcmd_ = true;
            r_ = command[0]; theta_ = command[1]; phi_ = command[2];
        }

        void _get_ee_curr_pose(){
            handmal_->getFramePosition("kinova_joint_end_effector", ee_position_rai_);
            handmal_->getFrameOrientation("kinova_joint_end_effector", ee_orientation_mat_rai_);

            ee_position_ = ee_position_rai_.e();
            rotMatToQuat(ee_orientation_mat_rai_, ee_quat_rai_);
            ee_quat_ = ee_quat_rai_.e();
        }

        void _set_ee_goal_pose(){
            ee_goal_position_.setZero();
            ee_delta_euler_rai_.setZero();
            std::uniform_real_distribution<> distrib(0, 1);
            if (!setcmd_) {
                r_ = cf_ * (distrib(gen_) * 0.2 + 0.15);
                theta_ = distrib(gen_) * M_PI;
                phi_ = distrib(gen_) * 2 * M_PI;
                ee_delta_position_ << r_ * cos(theta_) * sin(phi_) + 1.5 * cf_,
                        r_ * sin(theta_) * sin(phi_),
                        r_ * cos(phi_) - 0.5 * cf_;
            }
            else {
                ee_delta_position_ << r_ * cos(theta_) * sin(phi_) + 1.5,
                        r_ * sin(theta_) * sin(phi_),
                        r_ * cos(phi_) - 0.5;
            }
//            ee_delta_position_ << r_ * cos(theta_) * sin(phi_),
//                                        r_ * sin(theta_) * sin(phi_),
//                                        r_ * cos(phi_);
            ee_goal_position_ = ee_delta_position_ + ee_init_position_;

            angle_ = distrib(gen_) * M_PI; yaw_angle_ = distrib(gen_) * 2 * M_PI; pitch_angle_ = distrib(gen_) * M_PI;
            ee_delta_euler_rai_[0] = angle_ * cos(yaw_angle_) * sin(pitch_angle_),
            ee_delta_euler_rai_[1] = angle_ * sin(yaw_angle_) * sin(pitch_angle_),
            ee_delta_euler_rai_[2] = angle_ * cos(pitch_angle_);

            eulerVecToQuat(ee_delta_euler_rai_, ee_delta_quat_rai_);

            quatMul(ee_quat_rai_, ee_delta_quat_rai_, ee_goal_quat_rai_);
            ee_goal_quat_ = ee_goal_quat_rai_.e();
        }

        void _generate_ee_traj(){
            double eef_time = ee_delta_position_.norm()/eef_linvel_limit_;
            int moving_points = ceil(eef_time/control_dt_);

            Eigen::VectorXd traj_pose; double t;
            ee_curr_commnad_.clear();

            for(int i=0;i<moving_points+1;i++){
                traj_pose.setZero(7);
                t = double(i)/moving_points;
                quatPower(ee_delta_quat_rai_, t, ee_curr_delta_quat_rai_);
                quatMul(ee_quat_rai_, ee_curr_delta_quat_rai_, ee_curr_command_quat_rai_);
                traj_pose << t * ee_goal_position_ + (1-t) * ee_init_position_,
                                ee_curr_command_quat_rai_.e();
                ee_curr_commnad_.push_back(traj_pose);
            }

            int fixed_points = ceil((max_time_-eef_time)/control_dt_);
            for(int i=0;i<fixed_points+1;i++){
                traj_pose << ee_goal_position_, ee_goal_quat_;
                ee_curr_commnad_.push_back(traj_pose);
            }
//            ee_goal_center << ee_init_position_[0] + 1.5 * cf_,
//                               ee_init_position_[1],
//                               ee_init_position_[2] - 0.5 * cf_;
        }

    private:
        /// for initialization
        int gcDim_, gvDim_, nJoints_;
        int step_counter_=0, howManySteps_=0, itr_n_=0;
        bool visualizable_ = false; bool setcmd_ = false;
        raisim::ArticulatedSystem* handmal_;
        raisim::Visuals* obj_ee_curr_command_;
        raisim::Visuals* obj_ee_curr_position_;
        raisim::Visuals* obj_ee_goal_command_;
//        raisim::Visuals* object_workspace_final_;
//        raisim::Visuals* object_workspace_curr_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget15_, vTarget_, actionMean_, actionStd_, obDouble_;
        Eigen::VectorXd actionStd_leg_, actionStd_arm_, pTargetleg_, pTargetarm_;

        /// coefficients, constants and rewards
        double terminalRewardCoeff_ = -10.;
        double max_time_ = 0.;
        double curriculumFactor_ = 0., curriculumDecayFactor_ = 0., cf_ = 0.;
        double actionStd_val_ = 0.;

        /// for foots
        std::set<size_t> footIndices_;
        std::vector<size_t> footVec_, footFrame_;
        Eigen::VectorXd foot_pos_height_, foot_vel_, grf_bin_;
        int nFoot_ = 4;
        double clearance_ = 0.;
        Eigen::Vector4d t_a_, t_s_, t_a_i_, t_s_i_;
        Eigen::VectorXd current_foot_state_, last_foot_state_;

        /// body and base
        raisim::Mat<3,3> baseRot_;
        Eigen::Vector3d bodyOrientation_, bodyLinearVel_, bodyAngularVel_, previous_arm_joint_position_;
        raisim::Vec<4> quat_base_rai, base_quat_ee_rai_; // w_Q_b, b_Q_e, _rai_: raisim type vector

        /// eef pose
        raisim::Vec<3> ee_position_rai_, ee_delta_euler_rai_;
        raisim::Vec<4> ee_quat_rai_, ee_delta_quat_rai_, ee_goal_quat_rai_, ee_curr_delta_quat_rai_, ee_curr_command_quat_rai_;
        raisim::Mat<3,3> ee_orientation_mat_rai_;
        Eigen::Vector3d ee_position_, ee_delta_position_, ee_goal_position_, ee_init_position_;
        Eigen::Vector4d ee_quat_, ee_goal_quat_;
        std::deque<Eigen::VectorXd> ee_curr_commnad_;
        Eigen::Vector3d ee_goal_center, ee_final_goal_center;
        double r_=0., theta_=0., phi_=0.;
        double angle_=0., pitch_angle_=0., yaw_angle_=0.;
        double eef_linvel_limit_=0., eef_angvel_limit_=0.;

        /// history deque
        std::deque<Eigen::VectorXd> pTarget_history_, joint_position_history_, joint_velocity_history_;
        std::deque<Eigen::VectorXd> ee_position_history_, ee_quat_history_;

        /// etc.
        std::normal_distribution<double> normDist_;
        thread_local static std::mt19937 gen_;
        double clean_randomizer;
    };
    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
}