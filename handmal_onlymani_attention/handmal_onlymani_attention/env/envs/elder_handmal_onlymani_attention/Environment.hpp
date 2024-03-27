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

double angleBetweenVectors(const Eigen::Vector3d& v, const Eigen::Vector3d& w) {
    double dotProduct = v.dot(w);
    double normsProduct = v.norm() * w.norm();
    double cosAngle = dotProduct / normsProduct;
    double angleRadians = std::acos(cosAngle);
    if (normsProduct) return angleRadians;
    else return 0;
}

Eigen::VectorXd softmax(const Eigen::VectorXd& z) {
    Eigen::VectorXd result(z.size());
    double sum = 0.0;
    for (int i = 0; i < z.size(); ++i) {
        result(i) = std::exp(z(i));
        sum += result(i);
    }
    for (int i = 0; i < z.size(); ++i) {
        result(i) /= sum;
    }
    return result;
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
            handmal_ = world_->addArticulatedSystem(resourceDir_+"/handmal_test/urdf/handmal_test.urdf");
//            handmal_ = world_->addArticulatedSystem(resourceDir_+"/handmal/urdf/handmal.urdf");
            handmal_->setName("handmal_");
            handmal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            world_->addGround();
            /// config reward coefficients
            rewards_.initializeFromConfigurationFile (cfg["reward"]);
            READ_YAML(double, max_time_, cfg["max_time"]); READ_YAML(double, simulation_dt_, cfg["simulation_dt"]);
            READ_YAML(double, control_dt_, cfg["control_dt"]); READ_YAML(double, act_std_val_, cfg["action_std"]);
            READ_YAML(double, curriculumFactor_, cfg["curriculumFactor"]); READ_YAML(double, curriculumDecayFactor_, cfg["curriculumDecay"]);
            READ_YAML(double, eef_linvel_, cfg["eef_linvel"]); READ_YAML(double, eef_angvel_, cfg["eef_angvel"]);
            READ_YAML(double, clearance_, cfg["clearance"]);
            cf_ = curriculumFactor_;
            /// get robot data
            gcDim_ = handmal_->getGeneralizedCoordinateDim();
            gvDim_ = handmal_->getDOF();
            nJoints_ = gvDim_ - 6; // 24-6=18
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
            // in gv_: head(3):base linearvelocity, segment(3,3):base angularvelocity, segment(6,12):leg, segment(18,3):large arm joint, tail(3):small arm joint
            /// set pd gains for leg
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero(); jointPgain.tail(3).setConstant(400.0);
            jointPgain.segment(6,12).setConstant(50.0); //default arm's pgain = 40.0, default leg's pgain = 50.0
            jointDgain.setZero(); jointDgain.tail(3).setConstant(4.0);
            jointDgain.segment(6,12).setConstant(0.4); // defualt arm's dgain = 1.0, default leg's dgain = 0.4
            handmal_->setPdGains(jointPgain, jointDgain);
            handmal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
            /// policy input dimenstion
            obDim_ = 238;
            obDouble_.setZero(obDim_);
            /// action & observation scaling
            actionDim_ = nJoints_;
            actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
            actionMean_ = gc_init_.tail(nJoints_);
            rewDim_ = 11;

            actionStd_leg_.setZero(12); actionStd_arm_.setZero(3);
            actionStd_leg_.setConstant(act_std_val_);
            actionStd_arm_ << 0.05, 0.05, 0.05;
            actionStd_ << actionStd_leg_, actionStd_arm_;

            last_foot_state_.setZero(nFoot_);
            itr_n_ = 0;

            handmal_->setMass(handmal_->getBodyIdx("kinova_link_1"),0);
            handmal_->setMass(handmal_->getBodyIdx("kinova_link_2"),0);
            handmal_->setMass(handmal_->getBodyIdx("kinova_link_3"),0);

            /// if visualizable_, add visual sphere
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer(8080);
                object_manipul_ = server_->addVisualSphere("manipulation_command_position",0.03,0.1,0.1, 0.1, 1);
                object_eef_ = server_->addVisualSphere("eef_position",0.03,0.1,0.1, 0.1, 1);
                object_manipul_final_ = server_->addVisualSphere("end_eef_position",0.03,0.1,0.1, 0.1, 1);
                object_workspace_final_ = server_->addVisualSphere("eef_workspace_final",0.1,0,0, 1.0, 0.1);
                object_workspace_ = server_->addVisualSphere("eef_worksapce",0.1, 1,0,0,0.1);
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
            // set initail pose
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

            pTarget_history_.clear();
            joint_position_history_.clear();
            joint_velocity_history_.clear();
            eef_position_history_.clear();
            eef_quat_history_.clear();

            for (int j = 0; j < 10; j++) {
                pTarget_history_.push_back(Eigen::VectorXd::Zero(15));
                joint_position_history_.push_back(Eigen::VectorXd::Zero(15));
                joint_velocity_history_.push_back(Eigen::VectorXd::Zero(15));
                eef_position_history_.push_back(Eigen::VectorXd::Zero((3)));
                eef_quat_history_.push_back(Eigen::VectorXd::Zero(4));
            }
            updateObservation();
            init_eef_position_ = eef_position_;
            setendpose();
            gen_traj();

            previous_arm_joint_position_ = gc_.tail(3);
            previous_bodyLinearVel_.setZero();

            c_eef_position_ << init_eef_position_[0] + 2.0 * cf_,
                            init_eef_position_[1],
                            init_eef_position_[2] - 0.5 * cf_;

            c_eef_position_final_ << init_eef_position_[0] + 2.0,
                                    init_eef_position_[1],
                                    init_eef_position_[2] - 0.5;

            if (visualizable_) {
                    object_manipul_->setPosition(eef_traj_[step_counter_].head(3));
                    object_eef_->setPosition(eef_position_);
                    object_manipul_final_->setPosition(end_eef_position_);

                    object_workspace_->setPosition(c_eef_position_);
                    object_workspace_->setSphereSize(0.35*cf_);

                    object_workspace_final_->setPosition(c_eef_position_final_);
                    object_workspace_final_->setSphereSize(0.35);

                    object_manipul_->setColor(0,0,1,1);
                    object_eef_->setColor(0,0,1,1);
                    object_manipul_final_->setColor(1,0,1,1);
            }
            t_a_.setZero(); t_s_.setZero(); t_a_i_.setConstant(world_->getWorldTime()); t_s_i_.setConstant(world_->getWorldTime());
        }

        float step(const Eigen::Ref<EigenVec>& action, const Eigen::Ref<EigenVec>& attention) final {
            attention_ = attention.cast<double>();
//            std::cout << attention_.size() << std::endl;
            attention_ = softmax(attention_);
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
            VecDyn current_torque = handmal_->getGeneralizedForce();

            if (visualizable_) {
                if(step_counter_==10){
                    object_manipul_->setColor(0,1,0,1);
                    object_eef_->setColor(1,0,0,1);
                }
                if(step_counter_%10==0) {
                    object_manipul_->setPosition(eef_traj_[step_counter_].head(3));
                }
                object_eef_->setPosition((eef_position_));
            }
            /// reward calculation
            rewards_.reset();

            for (int fi = 0; fi < nFoot_; fi++){
                rewards_.record("airtime",attention_[0] * std::clamp(t_s_[fi] - t_a_[fi],-0.3,0.3),true);
                rewards_.record("footclearance",attention_[1] * pow((foot_pos_height_[fi]-clearance_),2) * pow(foot_vel_[fi],0.25),true); // clearance = 0.07
                rewards_.record("footslip", attention_[2] * grf_bin_[fi] * foot_vel_[fi],true);
            }

            // positive rewards
            if(step_counter_) {
                rewards_.record("cmdposition",
                attention_[3] * exp(-((eef_traj_[step_counter_ - 1].head(3) - eef_position_).squaredNorm()/pow(0.04/cf_,2))));
            }

            // negative rewards
            rewards_.record("jposition",
                            attention_[4] * (gc_.tail(3)-gc_init_.tail(3)).squaredNorm()); // position(15)->tend not to bend legs
            rewards_.record("jtorque",
                            attention_[5] * current_torque.e().tail(15).squaredNorm()); // torque(15)->rotate body 90 degree to reduce leg torque
            rewards_.record("jspeed",
                            attention_[6] * (gv_.tail(15)).squaredNorm());
            rewards_.record("jacc",
                            attention_[7] * (gv_.tail(15)-joint_velocity_history_[joint_velocity_history_.size()-1]).squaredNorm());
            rewards_.record("actsmooth1",
                            attention_[8] * (pTarget15_ - pTarget_history_[pTarget_history_.size()-1]).squaredNorm());
            rewards_.record("actsmooth2",
                            attention_[9] * (pTarget15_ - 2 * pTarget_history_[pTarget_history_.size()-1] + pTarget_history_[pTarget_history_.size()-2]).squaredNorm());
            rewards_.record("base",
                            attention_[10] * (0.8* pow(bodyLinearVel_[2],2) + 0.2*abs(bodyAngularVel_[0]) + 0.2*abs(bodyAngularVel_[1])));
//            rewards_.record("base", \
//            (bodyLinearVel_-previous_bodyLinearVel_).norm());

            pTarget_history_.push_back(pTarget15_);
            joint_position_history_.push_back(pTarget_history_[pTarget_history_.size()-1] - gc_.tail(15));
            joint_velocity_history_.push_back(gv_.tail(15));
            eef_position_history_.push_back(eef_position_);
            eef_quat_history_.push_back(eef_quat_);

            previous_arm_joint_position_ = gc_.tail(3);
            previous_bodyLinearVel_ = bodyLinearVel_;
            step_counter_++;

            if(visualizable_){
                if(step_counter_%10==0) {
                }
            }

            float positive_reward, negative_reward;
            positive_reward = rewards_["cmdposition"] + rewards_["airtime"];
            negative_reward = cf_ * (rewards_["jtorque"] + rewards_["jspeed"] + rewards_["jacc"]\
             + rewards_["actsmooth1"] + rewards_["actsmooth2"] + rewards_["base"]
             + rewards_["footclearance"] + rewards_["footslip"]);

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
                    baseRot_.e().transpose() * eef_position_, // 3 156
                    eef_quat_, // 4 // 160
                    baseRot_.e().transpose() * eef_position_history_[eef_position_history_.size()-1], // 3*3=9 169
                    baseRot_.e().transpose() * eef_position_history_[eef_position_history_.size()-3],
                    baseRot_.e().transpose() * eef_position_history_[eef_position_history_.size()-5],
                    eef_quat_history_[eef_quat_history_.size()-1], // 4*3=12 181
                    eef_quat_history_[eef_quat_history_.size()-3],
                    eef_quat_history_[eef_quat_history_.size()-5],
                    baseRot_.e().transpose() * relative_position_.e(), // 3 184
                    relative_quat_.e(), // 4 // 188
                    baseRot_.e().transpose() * end_eef_position_, // 191
                    end_eef_quat_, // 195
                    baseRot_.e().transpose() * eef_traj_[step_counter_], // 7*3=21 216
                    baseRot_.e().transpose() * eef_traj_[step_counter_+1],
                    baseRot_.e().transpose() * eef_traj_[step_counter_+2],
                    baseRot_.e().transpose() * (robot_COM_-RF_footPosition_.e()), // 3*4=12 228
                    baseRot_.e().transpose() * (robot_COM_-LF_footPosition_.e()),
                    baseRot_.e().transpose() * (robot_COM_-RH_footPosition_.e()),
                    baseRot_.e().transpose() * (robot_COM_-LH_footPosition_.e()),
                    bodyLinearVel_, // 231
                    current_foot_state_, // 235
                    foot_pos_height_; // 238
//                    command_, // command 6
            /// convert it to float
            ob = obDouble_.cast<float>();
        }

        void updateObservation() {
            handmal_->getState(gc_, gv_);
            geteefpose();

            raisim::Vec<4> quat;
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

            relative_position_ = end_eef_position_-eef_position_;
            quatInvQuatMul(end_eef_quat_, eef_quat_, relative_quat_);

            // Foot contact events
            Eigen::VectorXd swing_bin, foot_pos_bin;
            raisim::Vec<3> footVelocity;
            raisim::Vec<3> footPosition;

            swing_bin.setOnes(nFoot_);
            foot_pos_bin.setZero(nFoot_);
            grf_bin_.setZero(nFoot_);
            foot_vel_.setZero(nFoot_);
            foot_pos_height_.setZero(nFoot_);
            current_foot_state_ = last_foot_state_;

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
                        if (impulse_i > 0) // 땅에 닿았을 때
                        {
                            grf_bin_[footIdx_i] = 1.0;
                            swing_bin[footIdx_i] = 0.0;
                        }
                    }
                }
                // measure foot velocity
                handmal_->getFrameVelocity(footFrame_[footIdx_i], footVelocity);
                handmal_->getFramePosition(footFrame_[footIdx_i], footPosition);
                foot_pos_bin[footIdx_i] = (double)(footPosition[2] > clearance_);
                foot_vel_[footIdx_i] = footVelocity.squaredNorm();
                foot_pos_height_[footIdx_i] = footPosition[2];
            }

            for (int fi = 0; fi < nFoot_; fi++){
                if (foot_pos_bin[fi] == 1)
                    current_foot_state_[fi] = 1.0; // 0이 땅에 닿았다, 1.0이 땅에서 떨어진 상태
                else if (swing_bin[fi] == 0)
                    current_foot_state_[fi] = 0.0;
                // change contact
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
            torque = handmal_->getGeneralizedForce().e().tail(15).cast<float>();
        }

        void setCommand(Eigen::Ref<EigenVec>& command) final {
            setcmd_ = true;
            r_ = command[0]; theta_ = command[1]; phi_ = command[2];
        }

        void geteefpose(){
            handmal_->getFramePosition("kinova_joint_end_effector", eef_position_rai_);
            handmal_->getFrameOrientation("kinova_joint_end_effector", eef_orientation_rai_);

            eef_position_ = eef_position_rai_.e();
            rotMatToQuat(eef_orientation_rai_, eef_quat_rai_);
            eef_quat_ = eef_quat_rai_.e();
        }

        void setendpose(){
            end_eef_position_.setZero();
            end_eef_euler_delta_rai_.setZero();
            std::uniform_real_distribution<> distrib(0, 1);
            if (!setcmd_) {
                r_ = cf_ * (distrib(gen_) * 0.2 + 0.15);
                theta_ = distrib(gen_) * M_PI;
                phi_ = distrib(gen_) * 2 * M_PI;
                end_eef_position_delta_ << r_ * cos(theta_) * sin(phi_) + 2.0 * cf_,
                        r_ * sin(theta_) * sin(phi_),
                        r_ * cos(phi_) - 0.5 * cf_;
            }
            else {
                end_eef_position_delta_ << r_ * cos(theta_) * sin(phi_) + 2.0,
                        r_ * sin(theta_) * sin(phi_),
                        r_ * cos(phi_) - 0.5;
            }
//            end_eef_position_delta_ << r_ * cos(theta_) * sin(phi_),
//                                        r_ * sin(theta_) * sin(phi_),
//                                        r_ * cos(phi_);
            end_eef_position_ = end_eef_position_delta_ + init_eef_position_;

            angle_ = distrib(gen_) * M_PI; alpha_ = distrib(gen_) * 2 * M_PI; beta_ = distrib(gen_) * M_PI;
            end_eef_euler_delta_rai_[0] = angle_ * cos(alpha_) * sin(beta_),
            end_eef_euler_delta_rai_[1] = angle_ * sin(alpha_) * sin(beta_),
            end_eef_euler_delta_rai_[2] = angle_ * cos(beta_);

            eulerVecToQuat(end_eef_euler_delta_rai_, end_eef_quat_delta_rai_);

            quatMul(eef_quat_rai_, end_eef_quat_delta_rai_, end_eef_quat_rai_);
            end_eef_quat_ = end_eef_quat_rai_.e();
        }

        void gen_traj(){
//            double eef_time = std::max(r_/eef_linvel_, angle_/eef_angvel_);
            double eef_time = end_eef_position_delta_.norm()/eef_linvel_;
            int moving_points = ceil(eef_time/control_dt_);

            Eigen::VectorXd traj_pose; double t;
            eef_traj_.clear();

            for(int i=0;i<moving_points+1;i++){
                traj_pose.setZero(7);
                t = double(i)/moving_points;
                quatPower(end_eef_quat_delta_rai_, t, point_quat_delta_rai_);
                quatMul(eef_quat_rai_, point_quat_delta_rai_, point_quat_rai_);
                traj_pose << t * end_eef_position_ + (1-t) * init_eef_position_,
                                point_quat_rai_.e();
                eef_traj_.push_back(traj_pose);
            }

            int fixed_points = ceil((max_time_-eef_time)/control_dt_);
            for(int i=0;i<fixed_points+1;i++){
                traj_pose << end_eef_position_, end_eef_quat_;
                eef_traj_.push_back(traj_pose);
            }
        }

    private:
        /// for initialization
        int gcDim_, gvDim_, nJoints_;
        int step_counter_=0, howManySteps_=0;
        bool visualizable_ = false; bool setcmd_ = false;
        raisim::ArticulatedSystem* handmal_;
        raisim::Visuals* object_manipul_;
        raisim::Visuals* object_eef_;
        raisim::Visuals* object_manipul_final_;
        raisim::Visuals* object_workspace_final_;
        raisim::Visuals* object_workspace_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget15_, vTarget_, actionMean_, actionStd_, obDouble_;
        Eigen::VectorXd actionStd_leg_, actionStd_arm_, pTargetleg_, pTargetarm_, attention_;
/// Coefficients, constants and rewards
        double terminalRewardCoeff_ = -10.;
        double max_time_ = 0.;
        double curriculumFactor_ = 0., curriculumDecayFactor_ = 0., cf_ = 0.;
        double act_std_val_ = 0.;
/// for foots
        Eigen::Vector3d robot_COM_;
        std::set<size_t> footIndices_;
        std::vector<size_t> footVec_, footFrame_;
        Eigen::VectorXd foot_pos_height_, foot_vel_, grf_bin_;
        raisim::Vec<3> RF_footPosition_, LF_footPosition_, RH_footPosition_, LH_footPosition_;
        int nFoot_ = 4;
        double clearance_ = 0.;
/// eef pose
        raisim::Vec<3> eef_position_rai_, end_eef_euler_delta_rai_, relative_position_;
        raisim::Vec<4> eef_quat_rai_, end_eef_quat_delta_rai_, end_eef_quat_rai_, point_quat_delta_rai_, point_quat_rai_, relative_quat_;
        raisim::Mat<3,3> eef_orientation_rai_;
        double r_=0., theta_=0., phi_=0., angle_=0., alpha_=0., beta_=0.;
        Eigen::Vector3d eef_position_, end_eef_position_delta_, end_eef_position_, init_eef_position_, c_eef_position_, c_eef_position_final_;
        Eigen::Vector3d eef_euler_, end_eef_euler_;
        Eigen::Vector4d eef_quat_, end_eef_quat_;
        std::deque<Eigen::VectorXd> eef_traj_;
        double eef_linvel_=0., eef_angvel_=0.;
/// history deque
        raisim::Mat<3,3> baseRot_;
        Eigen::Vector3d bodyOrientation_, bodyLinearVel_, bodyAngularVel_, previous_arm_joint_position_, previous_bodyLinearVel_;
        std::deque<Eigen::VectorXd> pTarget_history_, joint_position_history_, joint_velocity_history_;
        std::deque<Eigen::VectorXd> eef_position_history_, eef_quat_history_;
        Eigen::Vector4d t_a_, t_s_, t_a_i_, t_s_i_;
        Eigen::VectorXd current_foot_state_, last_foot_state_;
/// etc.
        std::normal_distribution<double> normDist_;
        thread_local static std::mt19937 gen_;
//        Eigen::Matrix3d I_ = Eigen::Matrix3d::Identity();
        double clean_randomizer;
        int itr_n_;
    };
    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
}