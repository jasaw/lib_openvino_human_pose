/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Author: Joo Saw
**
** -------------------------------------------------------------------------*/

#include <iostream>
extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/parseutils.h>
#include <libswscale/swscale.h>
}
#include "log.hpp"
#include "peak.hpp"
#include "human_pose_estimator.hpp"

#include <opencv2/opencv.hpp> // for IMWRITE_PNG_COMPRESSION



namespace human_pose_estimation {

const size_t HumanPoseEstimator::keypointsNumber = 18;


void HumanPoseEstimator::get_scaled_image_dimensions(int width, int height,
                                                     int *scaled_width, int *scaled_height)
{
    int input_height = 0;
    int input_width  = 0;
    getInputWidthHeight(&input_width, &input_height);
    double scale_h = (double)input_height / height;
    double scale_w = (double)input_width  / width;
    double scale   = MIN(scale_h, scale_w);
    *scaled_width  = (int)(width * scale);
    *scaled_height = (int)(height * scale);
}


// caller must av_freep returned image
unsigned char *HumanPoseEstimator::scale_yuv2bgr(unsigned char *src_img, int width, int height, int scaled_width, int scaled_height)
{
    uint8_t *src_data[4] = {0};
    uint8_t *dst_data[4] = {0};
    int src_linesize[4] = {0};
    int dst_linesize[4] = {0};
    int src_w = width;
    int src_h = height;
    int dst_w = scaled_width;
    int dst_h = scaled_height;
    enum AVPixelFormat src_pix_fmt = AV_PIX_FMT_YUV420P;
    enum AVPixelFormat dst_pix_fmt = AV_PIX_FMT_BGR24;
    struct SwsContext *sws_ctx = NULL;

    // create scaling context
    sws_ctx = sws_getContext(src_w, src_h, src_pix_fmt,
                             dst_w, dst_h, dst_pix_fmt,
                             SWS_BICUBIC, NULL, NULL, NULL);
    if (!sws_ctx)
    {
        std::ostringstream stringStream;
        stringStream << "Impossible to create scale context for image conversion fmt:"
                     << av_get_pix_fmt_name(src_pix_fmt) << " s:" << src_w << "x" << src_h
                     << " -> fmt:" << av_get_pix_fmt_name(dst_pix_fmt) << " s:" << dst_w << "x" << dst_h;
        errMessage = stringStream.str();
        return NULL;
    }

    int srcNumBytes = av_image_fill_arrays(src_data, src_linesize, src_img,
                                           src_pix_fmt, src_w, src_h, 1);
    if (srcNumBytes < 0)
    {
        std::ostringstream stringStream;
        stringStream << "Failed to fill image arrays: code " << srcNumBytes;
        errMessage = stringStream.str();
        sws_freeContext(sws_ctx);
        return NULL;
    }

    int dst_bufsize;
    if ((dst_bufsize = av_image_alloc(dst_data, dst_linesize,
                       dst_w, dst_h, dst_pix_fmt, 1)) < 0)
    {
        std::ostringstream stringStream;
        stringStream << "Failed to allocate dst image";
        errMessage = stringStream.str();
        sws_freeContext(sws_ctx);
        return NULL;
    }

    // convert to destination format
    sws_scale(sws_ctx, (const uint8_t * const*)src_data,
              src_linesize, 0, src_h, dst_data, dst_linesize);

    sws_freeContext(sws_ctx);
    return dst_data[0];
}


HumanPoseEstimator::HumanPoseEstimator(bool matchJobIdToWorkerId_,
                                       int queueSize_,
                                       int numDevices_,
                                       const std::string& modelXmlPath,
                                       const std::string& modelBinPath,
                                       const std::string& targetDeviceName)
    : minJointsNumber(3),
      stride(8),
      meanPixel(cv::Vec3f::all(255)),
      minPeaksDistance(3.0f),
      midPointsScoreThreshold(0.05f),
      foundMidPointsRatioThreshold(0.8f),
      minSubsetScore(0.2f),
      inputLayerSize(-1, -1),
      upsampleRatio(4) {

    (void)targetDeviceName;

    matchJobIdToWorkerId = matchJobIdToWorkerId_;
    numDevices = numDevices_;

    // get ALL inference devices
    std::vector<std::string> availableDevices = ie.GetAvailableDevices();
    if ((numDevices <= 0) || ((int)availableDevices.size() < numDevices))
        numDevices = availableDevices.size();

    // Debug only
    std::cout << "Available devices: " << std::endl;
    for (auto && device : availableDevices) {
        std::cout << "\tDevice: " << device << std::endl;
    }

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
    for (int i = 0; i < numDevices; i++) {
        auto w = std::make_shared<Worker>(i, queueSize_, availableDevices.at(i));
        std::cout << "estimator " << i << " targetDeviceName : " << w->target_device_name << std::endl;
        w->executableNetwork = ie.LoadNetwork(network, w->target_device_name, {});
        // TODO: move this loop into worker object
        for (int j = 0; j < w->queue_size; j++) {
            w->infwork[j].first = w->executableNetwork.CreateInferRequestPtr();
            set_notify_on_job_completion(&w->infwork[j], w->jobs_mutex, w->worker_id);
        }
        workers.push_back(w);
    }
}


HumanPoseEstimator::~HumanPoseEstimator() {
}


void HumanPoseEstimator::getInputWidthHeight(int *width, int *height) {
    *width = inputLayerSize.width;
    *height = inputLayerSize.height;
}


void HumanPoseEstimator::set_notify_on_job_completion(std::pair<InferenceEngine::InferRequest::Ptr, std::shared_ptr<job::Job>> *infwork,
                                                      std::shared_ptr<std::mutex> jobs_mutex,
                                                      int worker_id_) {
    infwork->first->SetCompletionCallback(
        [&, infwork, jobs_mutex, worker_id_] {
                std::cout << "estimator " << worker_id_ << " callback for job ID " << infwork->second->id << std::endl;

                // Get inference result
                std::unique_lock<std::mutex> mlock(*jobs_mutex);
                if ((infwork->second) && (infwork->second->is_valid())) {
                    std::vector<human_pose_estimation::HumanPose> poses = getInferenceResult(infwork->first,
                                                                                             infwork->second,
                                                                                             worker_id_);
                    int job_id = infwork->second->id;
                    std::shared_ptr<job::Job> no_job(nullptr);
                    no_job.swap(infwork->second);
                    mlock.unlock();

                    std::unique_lock<std::mutex> resmlock(results_mutex);
                    std::map<int, std::queue<std::vector<human_pose_estimation::HumanPose>>>::iterator it;
                    it = results.find(job_id);
                    if (it != results.end()) {
                        // found it, append to queue
                        it->second.push(poses);
                    } else {
                        // not found, create a new queue
                        std::queue<std::vector<human_pose_estimation::HumanPose>> result_q;
                        result_q.push(poses);
                        results.insert(std::make_pair(job_id, result_q));
                    }
                    resmlock.unlock();
                } else {
                    // invalid job ?!
                    std::cout << "Invalid Job ID" << std::endl;
                    if (InferenceEngine::StatusCode::OK == infwork->first->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY))
                        infwork->first->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
                }
            }
        );
}


int HumanPoseEstimator::save_image_as_png(const cv::Mat &img, const char *filename)
{
    //cv::Size imageSize = img.size();
    //std::cout << "imageSize.width = " << imageSize.width << std::endl;
    //std::cout << "imageSize.height = " << imageSize.height << std::endl;
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);

    try {
        cv::imwrite(filename, img, compression_params);
    }
    catch (std::runtime_error& ex) {
        errMessage = "failed to convert image to PNG format: ";
        errMessage.append(ex.what());
        return -1;
    }
    return 0;
}


bool HumanPoseEstimator::queueJob(int id, unsigned char *image, int width, int height)
{
    bool ret = false;

    if (!job::id_is_valid(id))
        return ret;

    std::shared_ptr<Worker> nominated_worker = std::shared_ptr<Worker>(nullptr);
    if (!matchJobIdToWorkerId) {
        // mode 1: push job to next available queue
        // find which worker has the most empty queue
        int max_avail_size = 0;
        for (auto& worker : this->workers) {
            int q_size = worker->queue_available_size();
            if (q_size > max_avail_size) {
                max_avail_size = q_size;
                nominated_worker = worker;
            }
        }
    } else {
        // mode 2: push job to worker with matchin ID
        for (auto& worker : this->workers) {
            if (worker->worker_id == id) {
                if (worker->queue_available_size() > 0)
                    nominated_worker = worker;
                break;
            }
        }
    }

    if (nominated_worker) {
        int scaled_width = 0;
        int scaled_height = 0;
        get_scaled_image_dimensions(width, height, &scaled_width, &scaled_height);

        //// resize model input
        //if ((scaled_width != inputLayerSize.width) || (scaled_height != inputLayerSize.height)) {
        //    inputLayerSize.width = scaled_width;
        //    inputLayerSize.height = scaled_height;
        //    auto input_shapes = network.getInputShapes();
        //    std::string input_name;
        //    InferenceEngine::SizeVector input_shape;
        //    std::tie(input_name, input_shape) = *input_shapes.begin();
        //    input_shape[2] = inputLayerSize.height;
        //    input_shape[3] = inputLayerSize.width;
        //    input_shapes[input_name] = input_shape;
        //    network.reshape(input_shapes);
        //
        //    nominated_worker->executableNetwork = ie.LoadNetwork(network, nominated_worker->target_device_name, {});
        //    for (int j = 0; j < nominated_worker->queue_size; j++) {
        //        nominated_worker->infwork[j].first = nominated_worker->executableNetwork.CreateInferRequestPtr();
        //        set_notify_on_job_completion(&nominated_worker->infwork[j], nominated_worker->jobs_mutex, nominated_worker->worker_id);
        //    }
        //}

        unsigned char *scaled_img = scale_yuv2bgr(image, width, height, scaled_width, scaled_height);
        if (scaled_img) {
            try {
                std::unique_lock<std::mutex> mlock(*nominated_worker->jobs_mutex);
                for (int i = 0; i < nominated_worker->queue_size; i++) {
                    if ((!nominated_worker->infwork[i].second) || (!nominated_worker->infwork[i].second->is_valid())) {
                        nominated_worker->infwork[i].second = std::make_shared<job::Job>(id, width, height, scaled_width, scaled_height, scaled_img);
                        mlock.unlock();
                        estimateAsync(nominated_worker->worker_id,
                                      nominated_worker->infwork[i].first,
                                      nominated_worker->infwork[i].second);
                        ret = true;
                        break;
                    }
                }
            }
            catch (const std::exception &ex) {
                errMessage = "failed to queue inference: ";
                errMessage.append(ex.what());
            }
            av_freep(&scaled_img);
        }
        return ret;
    }

    return ret;
}


// return id negative means no inference result
void HumanPoseEstimator::estimateAsync(int worker_id_, InferenceEngine::InferRequest::Ptr request, std::shared_ptr<job::Job> the_job) {
    //save_image_as_png(the_job->scaledImage, "scaled_input.png");
    cv::Mat paddedImage = padImage(the_job->scaledImage);
    //save_image_as_png(paddedImage, "padded_input.png");
    InferenceEngine::Blob::Ptr input = request->GetBlob(inputName);
    auto buffer = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::U8>::value_type *>();
    imageToBuffer(paddedImage, buffer);

    std::cout << "estimator " << worker_id_ << " : Start async inference for job ID " << the_job->id << std::endl;
    request->StartAsync();
}


void HumanPoseEstimator::pollAsyncInferenceResults(void)
{
    for (auto& worker : this->workers) {
        for (int i = 0; i < worker->queue_size; i++) {
            std::unique_lock<std::mutex> mlock(*worker->jobs_mutex);
            if ((worker->infwork[i].second) && (worker->infwork[i].second->is_valid())) {

                InferenceEngine::StatusCode state = worker->infwork[i].first->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
                if (InferenceEngine::StatusCode::OK == state) {
                    std::vector<human_pose_estimation::HumanPose> poses = getWaitInferenceResult(worker->infwork[i].first,
                                                                                                 worker->infwork[i].second,
                                                                                                 worker->worker_id);
                    int job_id = worker->infwork[i].second->id;
                    std::shared_ptr<job::Job> no_job(nullptr);
                    no_job.swap(worker->infwork[i].second);
                    mlock.unlock();

                    std::unique_lock<std::mutex> resmlock(results_mutex);
                    std::map<int, std::queue<std::vector<human_pose_estimation::HumanPose>>>::iterator it;
                    it = results.find(job_id);
                    if (it != results.end()) {
                        // found it, append to queue
                        it->second.push(poses);
                    } else {
                        // not found, create a new queue
                        std::queue<std::vector<human_pose_estimation::HumanPose>> result_q;
                        result_q.push(poses);
                        results.insert(std::make_pair(job_id, result_q));
                    }
                    resmlock.unlock();
                }
            }
        }
    }
}


bool HumanPoseEstimator::resultIsReady(int id)
{
    //pollAsyncInferenceResults();
    std::unique_lock<std::mutex> mlock(results_mutex);
    std::map<int, std::queue<std::vector<human_pose_estimation::HumanPose>>>::iterator it;
    it = results.find(id);
    return (it != results.end());
}


std::vector<human_pose_estimation::HumanPose> HumanPoseEstimator::getResult(int id)
{
    std::vector<human_pose_estimation::HumanPose> poses;
    std::unique_lock<std::mutex> mlock(results_mutex);
    std::map<int, std::queue<std::vector<human_pose_estimation::HumanPose>>>::iterator it;
    it = results.find(id);
    if (it != results.end()) {
        poses = it->second.front();
        it->second.pop();
        if (it->second.empty())
            results.erase(id);
    }
    return poses;
}


std::vector<human_pose_estimation::HumanPose> HumanPoseEstimator::getPoses(InferenceEngine::InferRequest::Ptr request,
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


std::vector<human_pose_estimation::HumanPose> HumanPoseEstimator::getWaitInferenceResult(InferenceEngine::InferRequest::Ptr request,
                                                                                         std::shared_ptr<job::Job> the_job,
                                                                                         int worker_id_) {
    std::vector<human_pose_estimation::HumanPose> poses;
    InferenceEngine::StatusCode state = request->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
    std::cout << "estimator " << worker_id_ << " status is " << state << std::endl;
    if (InferenceEngine::StatusCode::OK == state) {
        // process result
        std::cout << "estimator " << worker_id_ << " : Inference job completed, calling wait" << std::endl;
        if (InferenceEngine::StatusCode::OK == request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {

            std::cout << "estimator " << worker_id_ << " : Getting results for job ID " << the_job->id << std::endl;

            cv::Size scaledImageSize = the_job->scaledImage.size();
            poses = getPoses(request,
                             the_job->fullImageSize,
                             scaledImageSize);
        }
    }
    return poses;
}


std::vector<human_pose_estimation::HumanPose> HumanPoseEstimator::getInferenceResult(InferenceEngine::InferRequest::Ptr request,
                                                                                     std::shared_ptr<job::Job> the_job,
                                                                                     int worker_id_) {
    std::cout << "estimator " << worker_id_ << " : Getting results for job ID " << the_job->id << std::endl;

    cv::Size scaledImageSize = the_job->scaledImage.size();
    std::vector<human_pose_estimation::HumanPose> poses = getPoses(request,
                                                                   the_job->fullImageSize,
                                                                   scaledImageSize);
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



Worker::Worker(int worker_id_, int queueSize_, std::string &targetDeviceName)
{
    worker_id = worker_id_;
    queue_size = queueSize_;
    target_device_name = targetDeviceName;
    jobs_mutex = std::make_shared<std::mutex>();
    infwork = std::make_unique<std::pair<InferenceEngine::InferRequest::Ptr, std::shared_ptr<job::Job>>[]>(queue_size);
}


int Worker::queue_available_size(void)
{
    int cnt = 0;
    std::unique_lock<std::mutex> mlock(*jobs_mutex);
    for (int i = 0; i < queue_size; i++) {
        if ((!infwork[i].second) || (!infwork[i].second->is_valid()))
            cnt++;
    }
    return cnt;
}


}  // namespace human_pose_estimation
