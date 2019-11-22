/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Author: Joo Saw
**
** -------------------------------------------------------------------------*/

#pragma once

#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <map>
#include <mutex>
#include <condition_variable>

#include <inference_engine.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "job.hpp"
#include "human_pose.hpp"
#include "human_pose_estimator.hpp"

namespace scheduler {

class Scheduler{
public:
    Scheduler(int numDevices,
              const std::string& modelXmlPath,
              const std::string& modelBinPath,
              const std::string& targetDeviceName);
    ~Scheduler();
    bool resultIsReady(int id);
    void waitResult(int id);
    std::vector<human_pose_estimation::HumanPose> getResult(int id);
    bool queueJob(int id, unsigned char *image, int width, int height);
    void clearResults(int id);
    void terminate(void);
    void joinThreads(void);

    bool matchJobIdToWorkerId;

private:
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network;
    std::vector<human_pose_estimation::HumanPoseEstimator *> workers;
    std::queue<job::Job> jobs;
    std::map<int, std::queue<std::vector<human_pose_estimation::HumanPose>>> results;
    std::vector<std::thread> threads;
    bool terminated;
    std::mutex results_mutex;
    std::condition_variable results_cond;
};

}  // namespace scheduler
