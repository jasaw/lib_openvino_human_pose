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

#include <opencv2/opencv.hpp>

//#include <ie_device.hpp>
//#include <ie_plugin_config.hpp>
//#include <ie_plugin_dispatcher.hpp>
//#include <ie_plugin_ptr.hpp>
////#include <ie_plugin_cpp.hpp>
//#include <ie_extension.h>


namespace human_pose_estimation {
const size_t HumanPoseEstimator::keypointsNumber = 18;


HumanPoseEstimator::HumanPoseEstimator(int worker_id_,
                                       const std::string& modelXmlPath,
                                       const std::string& modelBinPath,
                                       const std::string& targetDeviceName)
    : minJointsNumber(3),
      stride(8),
      meanPixel(cv::Vec3f::all(128)),
      minPeaksDistance(3.0f),
      midPointsScoreThreshold(0.05f),
      foundMidPointsRatioThreshold(0.8f),
      minSubsetScore(0.2f),
      inputLayerSize(-1, -1),
      upsampleRatio(4),
      job_index(0) {

    jobs_mutex = new std::mutex();
    jobs_cond = new std::condition_variable();
    worker_id = worker_id_;

    //int numDevices = 0; // TODO: take this from constructor parameter

    //// get ALL inference devices
    //std::string allDevices = "MULTI:";
    //std::vector<std::string> availableDevices = ie.GetAvailableDevices();
    //if ((numDevices <= 0) || ((int)availableDevices.size() < numDevices))
    //    numDevices = availableDevices.size();
    //
    //// Debug only
    //std::cout << "Available devices: " << std::endl;
    //for (auto && device : availableDevices) {
    //    std::cout << "\tDevice: " << device << std::endl;
    //    allDevices += device;
    //    allDevices += ((device == availableDevices[availableDevices.size()-1]) ? "" : ",");
    //}

    //InferenceEngine::PluginDispatcher dispatcher({""});
    //InferenceEngine::InferenceEnginePluginPtr _plugin(dispatcher.getPluginByDevice("MYRIAD"));
    //InferenceEngine::InferencePlugin plugin(_plugin);

    // read model
    InferenceEngine::CNNNetReader netReader;
    netReader.ReadNetwork(modelXmlPath); // model.xml file
    netReader.ReadWeights(modelBinPath); // model.bin file

    network = netReader.getNetwork();
    network.setBatchSize(1);

    // prepare input blobs
    InferenceEngine::InputsDataMap input_data_map = network.getInputsInfo();
    InferenceEngine::InputInfo::Ptr inputInfo = input_data_map.begin()->second;
    inputName = input_data_map.begin()->first;
    inputLayerSize = cv::Size(inputInfo->getTensorDesc().getDims()[3], inputInfo->getTensorDesc().getDims()[2]);
    inputInfo->setPrecision(InferenceEngine::Precision::U8);

    // prepare output blobs
    InferenceEngine::OutputsDataMap outputInfo = network.getOutputsInfo();
    auto outputBlobsIt = outputInfo.begin();
    pafsBlobName = outputBlobsIt->first;
    heatmapsBlobName = (++outputBlobsIt)->first;

    // load network to device
    executableNetwork = ie.LoadNetwork(network, targetDeviceName, {});
    //for (int i = 0; i < 2; i++) {
    //    std::cout << "estimator " << worker_id << " targetDeviceName : " << availableDevices.at(i) << std::endl;
    //    //executableNetwork[i] = plugin.LoadNetwork(network, {});
    //    executableNetwork[i] = ie.LoadNetwork(network, availableDevices.at(i), {});
    //    //executableNetwork[i] = ie.LoadNetwork(network, availableDevices.at(i), {{"VPU_FORCE_RESET", "NO"}});
    //}

    // create infer requests
    for (int i = 0; i < INFER_QUEUE_SIZE; i++) {
        async_infer_request[i] = executableNetwork.CreateInferRequest();
        //set_notify_on_job_completion(&async_infer_request[i]);
        the_job[i] = NULL;
    }
}


HumanPoseEstimator::~HumanPoseEstimator() {
    std::cout << "estimator " << worker_id << " destructor called" << std::endl;
    delete jobs_mutex;
    jobs_mutex = NULL;
    delete jobs_cond;
    jobs_cond = NULL;
}


void HumanPoseEstimator::getInputWidthHeight(int *width, int *height) {
    *width = inputLayerSize.width;
    *height = inputLayerSize.height;
}


int HumanPoseEstimator::get_worker_id(void) {
    return worker_id;
}


void HumanPoseEstimator::set_notify_on_job_completion(InferenceEngine::InferRequest *request) const {
    request->SetCompletionCallback(
        [&] {
                std::cout << "estimator " << worker_id << " callback : Inference Completed" << std::endl;
                //jobs_cond->notify_all();
                //std::cout << "estimator " << worker_id << " callback : notified" << std::endl;
            }
        );
}


bool HumanPoseEstimator::queue_not_full(void) {
    std::unique_lock<std::mutex> mlock(*jobs_mutex);
    return get_next_empty_job_index() >= 0;
}


int HumanPoseEstimator::queue_available_size(void) {
    int cnt = 0;
    std::unique_lock<std::mutex> mlock(*jobs_mutex);
    for (int i = 0; i < INFER_QUEUE_SIZE; i++) {
        if ((the_job[i] == NULL) || (!the_job[i]->is_valid()))
            cnt++;
    }
    return cnt;
}


int HumanPoseEstimator::get_next_empty_job_index(void) {
    int tmp_index = job_index;
    for (int i = 0; i < INFER_QUEUE_SIZE; i++) {
        if ((the_job[i] == NULL) || (!the_job[tmp_index]->is_valid()))
            return tmp_index;
        tmp_index++;
        if (tmp_index >= INFER_QUEUE_SIZE)
            tmp_index = 0;
    }
    return -1;
}


// return id negative means no inference result
bool HumanPoseEstimator::estimateAsync(job::Job *new_job) {
    //std::unique_lock<std::mutex> mlock(*jobs_mutex);
    if (!new_job->is_valid())
        return false;

    int next_job_index = get_next_empty_job_index();
    if (next_job_index < 0) // queue full
        return false;

    {
        std::unique_lock<std::mutex> mlock(*jobs_mutex);
        the_job[next_job_index] = new_job;
    }
    cv::Mat paddedImage = padImage(the_job[next_job_index]->scaledImage);
    InferenceEngine::Blob::Ptr input = async_infer_request[next_job_index].GetBlob(inputName);
    auto buffer = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::U8>::value_type *>();
    imageToBuffer(paddedImage, buffer);

    std::cout << "estimator " << worker_id << " : Start async inference, job index " << next_job_index << std::endl;
    async_infer_request[next_job_index].StartAsync();
    return true;
}


bool HumanPoseEstimator::current_job_is_done(void) {
    InferenceEngine::StatusCode state = async_infer_request[job_index].Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
    std::cout << "estimator " << worker_id << " : job index " << job_index << " status is " << state << std::endl;
    return (InferenceEngine::StatusCode::OK == state);
}


std::vector<human_pose_estimation::HumanPose> HumanPoseEstimator::getPoses(InferenceEngine::InferRequest *request,
                                                                           const cv::Size& orgImageSize,
                                                                           const cv::Size& scaledImageSize) const {
    InferenceEngine::Blob::Ptr pafsBlob = request->GetBlob(pafsBlobName);
    InferenceEngine::Blob::Ptr heatMapsBlob = request->GetBlob(heatmapsBlobName);
    CV_Assert(heatMapsBlob->getTensorDesc().getDims()[1] == keypointsNumber + 1);
    InferenceEngine::SizeVector heatMapDims =
            heatMapsBlob->getTensorDesc().getDims();
    std::vector<human_pose_estimation::HumanPose> poses = postprocess(
            heatMapsBlob->buffer(),
            heatMapDims[2] * heatMapDims[3],
            keypointsNumber,
            pafsBlob->buffer(),
            heatMapDims[2] * heatMapDims[3],
            pafsBlob->getTensorDesc().getDims()[1],
            heatMapDims[3], heatMapDims[2], orgImageSize, scaledImageSize);

    return poses;
}


std::pair<int, std::vector<human_pose_estimation::HumanPose>> HumanPoseEstimator::getResult(void) {
    std::vector<human_pose_estimation::HumanPose> poses;
    int id = job::Job::invalid_job_id;

    std::unique_lock<std::mutex> mlock(*jobs_mutex);
    if (current_job_is_done()) {

        std::cout << "estimator " << worker_id << " : Inference job index " << job_index << " completed, calling wait" << std::endl;

        if (InferenceEngine::StatusCode::OK == async_infer_request[job_index].Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {

            std::cout << "estimator " << worker_id << " : Getting results from job index " << job_index << std::endl;

            cv::Size scaledImageSize = the_job[job_index]->scaledImage.size();
            poses = getPoses(&async_infer_request[job_index],
                             the_job[job_index]->fullImageSize,
                             scaledImageSize);
            id = the_job[job_index]->id;
            delete the_job[job_index];
            the_job[job_index] = NULL;
            job_index++;
            if (job_index >= INFER_QUEUE_SIZE)
                job_index = 0;
        }
    }
    mlock.unlock();

    return std::make_pair(id, poses);
}


void HumanPoseEstimator::imageToBuffer(const cv::Mat& scaledImage, uint8_t* buffer) const {
    std::vector<cv::Mat> planes(3);
    for (size_t pId = 0; pId < planes.size(); pId++) {
        planes[pId] = cv::Mat(inputLayerSize, CV_8UC1,
                              buffer + pId * inputLayerSize.area());
    }
    cv::split(scaledImage, planes);
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


std::vector<human_pose_estimation::HumanPose> HumanPoseEstimator::postprocess(
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

    std::vector<human_pose_estimation::HumanPose> poses = extractPoses(heatMaps, pafs);
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

std::vector<human_pose_estimation::HumanPose> HumanPoseEstimator::extractPoses(
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
    std::vector<human_pose_estimation::HumanPose> poses = groupPeaksToPoses(
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

void HumanPoseEstimator::correctCoordinates(std::vector<human_pose_estimation::HumanPose>& poses,
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








}  // namespace human_pose_estimation
