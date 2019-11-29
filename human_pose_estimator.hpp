/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Author: Joo Saw
**
** -------------------------------------------------------------------------*/

#pragma once

#include <time.h>
#include <string>
#include <vector>
#include <queue>
#include <utility>
#include <mutex>
//#include <condition_variable>

#include <inference_engine.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "job.hpp"
#include "human_pose.hpp"



namespace human_pose_estimation {


class Worker {
public:
    Worker(int worker_id_, int queueSize_, std::string &targetDeviceName);
    int queue_available_size(void);

    InferenceEngine::ExecutableNetwork executableNetwork;
    std::unique_ptr<std::pair<InferenceEngine::InferRequest::Ptr, std::shared_ptr<job::Job>>[]> infwork;
    int worker_id;
    int queue_size;
    std::string target_device_name;
    std::shared_ptr<std::mutex> jobs_mutex;
};


class HumanPoseEstimator {
public:
    static const size_t keypointsNumber;

    HumanPoseEstimator(bool matchJobIdToWorkerId_,
                       int queueSize_,
                       int numDevices_,
                       const std::string& modelXmlPath,
                       const std::string& modelBinPath,
                       const std::string& targetDeviceName);
    ~HumanPoseEstimator();
    bool queueJob(int id, struct timeval *timestamp,
                  unsigned char *image, int width, int height);
    bool resultIsReady(int id);
    std::pair<std::shared_ptr<struct timeval>, std::vector<human_pose_estimation::HumanPose>> getResult(int id);

    bool matchJobIdToWorkerId;

private:
    int save_image_as_png(const cv::Mat &img, const char *filename);
    void pollAsyncInferenceResults(void);

    void estimateAsync(int worker_id_, InferenceEngine::InferRequest::Ptr request, std::shared_ptr<job::Job> the_job);
    std::vector<human_pose_estimation::HumanPose> getInferenceResult(InferenceEngine::InferRequest::Ptr request,
                                                                     std::shared_ptr<job::Job> the_job,
                                                                     int worker_id_);
    std::vector<human_pose_estimation::HumanPose> getWaitInferenceResult(InferenceEngine::InferRequest::Ptr request,
                                                                         std::shared_ptr<job::Job> the_job,
                                                                         int worker_id_);
    void getInputWidthHeight(int *width, int *height);
    void set_notify_on_job_completion(std::pair<InferenceEngine::InferRequest::Ptr, std::shared_ptr<job::Job>> *infwork,
                                      std::shared_ptr<std::mutex> jobs_mutex,
                                      int worker_id_);
    void get_scaled_image_dimensions(int width, int height,
                                     int *scaled_width, int *scaled_height);
    unsigned char *scale_yuv2bgr(unsigned char *src_img, int width, int height, int scaled_width, int scaled_height);

    void imageToBuffer(const cv::Mat& scaledImage, uint8_t* buffer) const;
    std::vector<human_pose_estimation::HumanPose> getPoses(InferenceEngine::InferRequest::Ptr request,
                                                           const cv::Size& orgImageSize,
                                                           const cv::Size& scaledImageSize) const;
    std::vector<human_pose_estimation::HumanPose> postprocess(
            const float* heatMapsData, const int heatMapOffset, const int nHeatMaps,
            const float* pafsData, const int pafOffset, const int nPafs,
            const int featureMapWidth, const int featureMapHeight,
            const cv::Size& imageSize, const cv::Size& scaledImageSize) const;
    std::vector<human_pose_estimation::HumanPose> extractPoses(const std::vector<cv::Mat>& heatMaps,
                                        const std::vector<cv::Mat>& pafs) const;
    void resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const;
    void correctCoordinates(std::vector<human_pose_estimation::HumanPose>& poses,
                            const cv::Size& featureMapsSize,
                            const cv::Size& imageSize,
                            const cv::Size& scaledImageSize) const;
    cv::Mat padImage(const cv::Mat& scaledImage) const;

    int minJointsNumber;
    int stride;
    cv::Vec3f meanPixel;
    float minPeaksDistance;
    float midPointsScoreThreshold;
    float foundMidPointsRatioThreshold;
    float minSubsetScore;
    cv::Size inputLayerSize;
    int upsampleRatio;
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network;
    int numDevices;
    std::string inputName;
    std::string pafsBlobName;
    std::string heatmapsBlobName;
    std::vector<std::shared_ptr<Worker>> workers;
    std::mutex results_mutex;
    std::map<int, std::list<std::pair<std::shared_ptr<struct timeval>, std::vector<human_pose_estimation::HumanPose>>>> results;
};

}  // namespace human_pose_estimation
