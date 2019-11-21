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
#include <utility>
#include <mutex>
#include <condition_variable>

#include <inference_engine.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "job.hpp"
#include "human_pose.hpp"


#define INFER_QUEUE_SIZE    2


namespace human_pose_estimation {

class HumanPoseEstimator {
public:
    static const size_t keypointsNumber;

    HumanPoseEstimator(int worker_id_,
                       const std::string& modelXmlPath,
                       const std::string& modelBinPath,
                       const std::string& targetDeviceName);
    ~HumanPoseEstimator();
    bool estimateAsync(job::Job *new_job);
    bool queue_not_full(void);
    int queue_available_size(void);
    int get_worker_id(void);
    bool current_job_is_done(void);
    std::pair<int, std::vector<human_pose_estimation::HumanPose>> getResult(void);
    void getInputWidthHeight(int *width, int *height);

    std::mutex *jobs_mutex;
    std::condition_variable *jobs_cond;

private:
    void set_notify_on_job_completion(InferenceEngine::InferRequest *request) const;
    int get_next_empty_job_index(void);
    void imageToBuffer(const cv::Mat& scaledImage, uint8_t* buffer) const;
    std::vector<human_pose_estimation::HumanPose> getPoses(InferenceEngine::InferRequest *request,
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
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest async_infer_request[INFER_QUEUE_SIZE];
    job::Job *the_job[INFER_QUEUE_SIZE];
    int job_index;
    std::string inputName;
    std::string pafsBlobName;
    std::string heatmapsBlobName;
    int worker_id;
};

}  // namespace human_pose_estimation
