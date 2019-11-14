/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Author: Joo Saw
**
** -------------------------------------------------------------------------*/

#include "peak.hpp"
#include "human_pose_estimator.hpp"


namespace human_pose_estimation {
const size_t HumanPoseEstimator::keypointsNumber = 18;

HumanPoseEstimator::HumanPoseEstimator(const std::string& modelXmlPath_,
                                       const std::string& modelBinPath_,
                                       const std::string& targetDeviceName_)
    : requestCount(0),
      minJointsNumber(3),
      stride(8),
      meanPixel(cv::Vec3f::all(128)),
      minPeaksDistance(3.0f),
      midPointsScoreThreshold(0.05f),
      foundMidPointsRatioThreshold(0.8f),
      minSubsetScore(0.2f),
      inputLayerSize(-1, -1),
      upsampleRatio(4),
      targetDeviceName(targetDeviceName_),
      modelXmlPath(modelXmlPath_),
      modelBinPath(modelBinPath_) {
    netReader.ReadNetwork(modelXmlPath); // model.xml file
    netReader.ReadWeights(modelBinPath); // model.bin file
    network = netReader.getNetwork();
    InferenceEngine::InputInfo::Ptr inputInfo = network.getInputsInfo().begin()->second;
    inputLayerSize = cv::Size(inputInfo->getTensorDesc().getDims()[3], inputInfo->getTensorDesc().getDims()[2]);
    inputInfo->setPrecision(InferenceEngine::Precision::U8);

    InferenceEngine::OutputsDataMap outputInfo = network.getOutputsInfo();
    auto outputBlobsIt = outputInfo.begin();
    pafsBlobName = outputBlobsIt->first;
    heatmapsBlobName = (++outputBlobsIt)->first;

    executableNetwork = ie.LoadNetwork(network, targetDeviceName);
    request = executableNetwork.CreateInferRequest();
}


HumanPoseEstimator::~HumanPoseEstimator() {
}


void HumanPoseEstimator::getInputWidthHeight(int *width, int *height) {
    *width = inputLayerSize.width;
    *height = inputLayerSize.height;
}


bool HumanPoseEstimator::queueIsEmpty(void) {
    if (requestCount > 0)
        return false;
    return true;
}


bool HumanPoseEstimator::resultIsReady(void) {
    if (requestCount < 1)
        return false;
    InferenceEngine::StatusCode state = request.Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
    return (InferenceEngine::StatusCode::OK == state);
}


void HumanPoseEstimator::waitResult(void) {
    request.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}


std::vector<HumanPose> HumanPoseEstimator::estimate(const cv::Mat& scaledImage, const cv::Size& orgImageSize) {
    if (requestCount > 0) {
        std::vector<HumanPose> poses;
        return poses;
    }
    requestCount++;

    auto scaledImageSize = scaledImage.size();
    InferenceEngine::Blob::Ptr input = request.GetBlob(network.getInputsInfo().begin()->first);
    auto buffer = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::U8>::value_type *>();
    cv::Mat paddedImage = padImage(scaledImage);
    imageToBuffer(paddedImage, buffer);

    request.Infer();

    InferenceEngine::Blob::Ptr pafsBlob = request.GetBlob(pafsBlobName);
    InferenceEngine::Blob::Ptr heatMapsBlob = request.GetBlob(heatmapsBlobName);
    CV_Assert(heatMapsBlob->getTensorDesc().getDims()[1] == keypointsNumber + 1);
    InferenceEngine::SizeVector heatMapDims =
            heatMapsBlob->getTensorDesc().getDims();
    std::vector<HumanPose> poses = postprocess(
            heatMapsBlob->buffer(),
            heatMapDims[2] * heatMapDims[3],
            keypointsNumber,
            pafsBlob->buffer(),
            heatMapDims[2] * heatMapDims[3],
            pafsBlob->getTensorDesc().getDims()[1],
            heatMapDims[3], heatMapDims[2], orgImageSize, scaledImageSize);

    requestCount--;

    return poses;
}


void HumanPoseEstimator::estimateAsync(const cv::Mat& scaledImage) {
    if (requestCount > 0)
        return;
    requestCount++;

    cv::Mat paddedImage = padImage(scaledImage);
    InferenceEngine::Blob::Ptr input = request.GetBlob(network.getInputsInfo().begin()->first);
    auto buffer = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::U8>::value_type *>();
    imageToBuffer(paddedImage, buffer);

    request.StartAsync();
}


std::vector<HumanPose> HumanPoseEstimator::getResult(const cv::Size& orgImageSize, const cv::Size& scaledImageSize) {
    if (requestCount < 1) {
        std::vector<HumanPose> poses;
        return poses;
    }
    waitResult();
    InferenceEngine::Blob::Ptr pafsBlob = request.GetBlob(pafsBlobName);
    InferenceEngine::Blob::Ptr heatMapsBlob = request.GetBlob(heatmapsBlobName);
    CV_Assert(heatMapsBlob->getTensorDesc().getDims()[1] == keypointsNumber + 1);
    InferenceEngine::SizeVector heatMapDims =
            heatMapsBlob->getTensorDesc().getDims();
    std::vector<HumanPose> poses = postprocess(
            heatMapsBlob->buffer(),
            heatMapDims[2] * heatMapDims[3],
            keypointsNumber,
            pafsBlob->buffer(),
            heatMapDims[2] * heatMapDims[3],
            pafsBlob->getTensorDesc().getDims()[1],
            heatMapDims[3], heatMapDims[2], orgImageSize, scaledImageSize);
    requestCount--;
    return poses;
}


void HumanPoseEstimator::imageToBuffer(const cv::Mat& scaledImage, uint8_t* buffer) const {
    std::vector<cv::Mat> planes(3);
    for (size_t pId = 0; pId < planes.size(); pId++) {
        planes[pId] = cv::Mat(inputLayerSize, CV_8UC1,
                              buffer + pId * inputLayerSize.area());
    }
    cv::split(scaledImage, planes);
}


cv::Mat HumanPoseEstimator::scaleImage(const cv::Mat& image) {
    cv::Mat scaledImage;
    double scale = inputLayerSize.height / static_cast<double>(image.rows);
    cv::resize(image, scaledImage, cv::Size(), scale, scale, cv::INTER_CUBIC);
    return padImage(scaledImage);
}


cv::Mat HumanPoseEstimator::padImage(const cv::Mat& scaledImage) const {
    cv::Mat paddedImage;
    cv::Size scaledImageSize = scaledImage.size();
    int w_diff = inputLayerSize.width - scaledImageSize.width;
    int h_diff = inputLayerSize.height - scaledImageSize.height;
    int left = w_diff >> 1;
    int right = w_diff - left;
    int top = h_diff >> 1;
    int bottom = h_diff - top;
    cv::copyMakeBorder(scaledImage, paddedImage, top, bottom, left, right,
                       cv::BORDER_CONSTANT, meanPixel);
    return paddedImage;
}


std::vector<HumanPose> HumanPoseEstimator::postprocess(
        const float* heatMapsData, const int heatMapOffset, const int nHeatMaps,
        const float* pafsData, const int pafOffset, const int nPafs,
        const int featureMapWidth, const int featureMapHeight,
        const cv::Size& imageSize, const cv::Size& scaledImageSize) const {
    std::vector<cv::Mat> heatMaps(nHeatMaps);
    for (size_t i = 0; i < heatMaps.size(); i++) {
        heatMaps[i] = cv::Mat(featureMapHeight, featureMapWidth, CV_32FC1,
                              reinterpret_cast<void*>(
                                  const_cast<float*>(
                                      heatMapsData + i * heatMapOffset)));
    }
    resizeFeatureMaps(heatMaps);

    std::vector<cv::Mat> pafs(nPafs);
    for (size_t i = 0; i < pafs.size(); i++) {
        pafs[i] = cv::Mat(featureMapHeight, featureMapWidth, CV_32FC1,
                          reinterpret_cast<void*>(
                              const_cast<float*>(
                                  pafsData + i * pafOffset)));
    }
    resizeFeatureMaps(pafs);

    std::vector<HumanPose> poses = extractPoses(heatMaps, pafs);
    correctCoordinates(poses, heatMaps[0].size(), imageSize, scaledImageSize);
    return poses;
}

class FindPeaksBody: public cv::ParallelLoopBody {
public:
    FindPeaksBody(const std::vector<cv::Mat>& heatMaps, float minPeaksDistance,
                  std::vector<std::vector<Peak> >& peaksFromHeatMap)
        : heatMaps(heatMaps),
          minPeaksDistance(minPeaksDistance),
          peaksFromHeatMap(peaksFromHeatMap) {}

    virtual void operator()(const cv::Range& range) const {
        for (int i = range.start; i < range.end; i++) {
            findPeaks(heatMaps, minPeaksDistance, peaksFromHeatMap, i);
        }
    }

private:
    const std::vector<cv::Mat>& heatMaps;
    float minPeaksDistance;
    std::vector<std::vector<Peak> >& peaksFromHeatMap;
};

std::vector<HumanPose> HumanPoseEstimator::extractPoses(
        const std::vector<cv::Mat>& heatMaps,
        const std::vector<cv::Mat>& pafs) const {
    std::vector<std::vector<Peak> > peaksFromHeatMap(heatMaps.size());
    FindPeaksBody findPeaksBody(heatMaps, minPeaksDistance, peaksFromHeatMap);
    cv::parallel_for_(cv::Range(0, static_cast<int>(heatMaps.size())),
                      findPeaksBody);
    int peaksBefore = 0;
    for (size_t heatmapId = 1; heatmapId < heatMaps.size(); heatmapId++) {
        peaksBefore += static_cast<int>(peaksFromHeatMap[heatmapId - 1].size());
        for (auto& peak : peaksFromHeatMap[heatmapId]) {
            peak.id += peaksBefore;
        }
    }
    std::vector<HumanPose> poses = groupPeaksToPoses(
                peaksFromHeatMap, pafs, keypointsNumber, midPointsScoreThreshold,
                foundMidPointsRatioThreshold, minJointsNumber, minSubsetScore);
    return poses;
}

void HumanPoseEstimator::resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const {
    for (auto& featureMap : featureMaps) {
        cv::resize(featureMap, featureMap, cv::Size(),
                   upsampleRatio, upsampleRatio, cv::INTER_CUBIC);
    }
}

void HumanPoseEstimator::correctCoordinates(std::vector<HumanPose>& poses,
                                            const cv::Size& featureMapsSize,
                                            const cv::Size& imageSize,
                                            const cv::Size& scaledImageSize) const {
    cv::Size fullFeatureMapSize = featureMapsSize * stride / upsampleRatio;

    int w_diff = inputLayerSize.width - scaledImageSize.width;
    int h_diff = inputLayerSize.height - scaledImageSize.height;
    int left = w_diff >> 1;
    int right = w_diff - left;
    int top = h_diff >> 1;
    int bottom = h_diff - top;

    float scaleX = imageSize.width /
            static_cast<float>(fullFeatureMapSize.width - left - right);
    float scaleY = imageSize.height /
            static_cast<float>(fullFeatureMapSize.height - top - bottom);
    for (auto& pose : poses) {
        for (auto& keypoint : pose.keypoints) {
            if (keypoint != cv::Point2f(-1, -1)) {
                keypoint.x *= stride / upsampleRatio;
                keypoint.x -= left;
                keypoint.x *= scaleX;

                keypoint.y *= stride / upsampleRatio;
                keypoint.y -= top;
                keypoint.y *= scaleY;
            }
        }
    }
}


//bool HumanPoseEstimator::changeInputWidth(const cv::Size& scaledImageSize) {
//    bool changed = false;
//    if (inputLayerSize.height != scaledImageSize.height) {
//        inputLayerSize.height  = scaledImageSize.height;
//        changed = true;
//    }
//    if (inputLayerSize.width != scaledImageSize.width) {
//        inputLayerSize.width  = scaledImageSize.width;
//        changed = true;
//    }
//    return changed;
//}


}  // namespace human_pose_estimation
