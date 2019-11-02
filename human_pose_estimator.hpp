/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** -------------------------------------------------------------------------*/

#pragma once

#include <string>
#include <vector>

#include <inference_engine.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "human_pose.hpp"


namespace human_pose_estimation {

class HumanPoseEstimator {
public:
    static const size_t keypointsNumber;

    HumanPoseEstimator(const std::string& modelXmlPath,
                       const std::string& modelBinPath,
                       const std::string& targetDeviceName);
    std::vector<HumanPose> estimate(const cv::Mat& image);
    void estimateAsync(const cv::Mat& image);
    bool resultIsReady(void);
    std::vector<HumanPose> getResult(void);
    ~HumanPoseEstimator();

private:
    void preprocess(const cv::Mat& image, uint8_t* buffer) const;
    std::vector<HumanPose> postprocess(
            const float* heatMapsData, const int heatMapOffset, const int nHeatMaps,
            const float* pafsData, const int pafOffset, const int nPafs,
            const int featureMapWidth, const int featureMapHeight,
            const cv::Size& imageSize) const;
    std::vector<HumanPose> extractPoses(const std::vector<cv::Mat>& heatMaps,
                                        const std::vector<cv::Mat>& pafs) const;
    void resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const;
    void correctCoordinates(std::vector<HumanPose>& poses,
                            const cv::Size& featureMapsSize,
                            const cv::Size& imageSize) const;
    bool changeInputWidth(const cv::Size& imageSize);

    int minJointsNumber;
    int stride;
    cv::Vec4i pad;
    cv::Vec3f meanPixel;
    float minPeaksDistance;
    float midPointsScoreThreshold;
    float foundMidPointsRatioThreshold;
    float minSubsetScore;
    cv::Size inputLayerSize;
    int upsampleRatio;
    InferenceEngine::Core ie;
    std::string targetDeviceName;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest request;
    InferenceEngine::CNNNetReader netReader;
    std::string pafsBlobName;
    std::string heatmapsBlobName;
    std::string modelXmlPath;
    std::string modelBinPath;
    cv::Size lastImageSize;
};

}  // namespace human_pose_estimation
