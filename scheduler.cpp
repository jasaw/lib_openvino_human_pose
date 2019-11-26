/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Author: Joo Saw
**
** -------------------------------------------------------------------------*/

#include <unistd.h>
#include <iostream>
extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/parseutils.h>
#include <libswscale/swscale.h>
}

#include "log.hpp"
#include "scheduler.hpp"

#if 0
namespace scheduler {

static void get_scaled_image_dimensions(human_pose_estimation::HumanPoseEstimator *estimator,
                                        int width, int height,
                                        int *scaled_width, int *scaled_height)
{
    int input_height = 0;
    int input_width  = 0;
    estimator->getInputWidthHeight(&input_width, &input_height);
    double scale_h = (double)input_height / height;
    double scale_w = (double)input_width  / width;
    double scale   = MIN(scale_h, scale_w);
    *scaled_width  = (int)(width * scale);
    *scaled_height = (int)(height * scale);
}


// caller must av_freep returned image
static unsigned char *scale_yuv2bgr(unsigned char *src_img, int width, int height, int scaled_width, int scaled_height)
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


static void worker_loop(std::map<int, std::queue<std::vector<human_pose_estimation::HumanPose>>>& combined_results,
                        std::mutex& results_mutex,
                        std::condition_variable& results_cond,
                        bool& terminated,
                        human_pose_estimation::HumanPoseEstimator *estimator)
{
    while (!terminated) {
        // wait for inference to be done, then getResult and stick into results
        //{
        //    std::unique_lock<std::mutex> mlock(*(estimator->jobs_mutex));
        //    while (!terminated) {
        //        if (estimator->current_job_is_done()) // inference is done
        //            break;
        //        std::cout << "estimator " << estimator->get_worker_id() << " waiting on lock" << std::endl;
        //        estimator->jobs_cond->wait(mlock, [&estimator]{ return estimator->current_job_is_done(); });
        //    }
        //}
                sleep(1);

        if (terminated)
            return;
        if (estimator->current_job_is_done()) {
            // get result from estimator
            try {
                std::pair<int, std::vector<human_pose_estimation::HumanPose>> curr_result = estimator->getResult();
                if (job::id_is_valid(curr_result.first)) {
                    std::unique_lock<std::mutex> mlock(results_mutex);
                    std::map<int, std::queue<std::vector<human_pose_estimation::HumanPose>>>::iterator it;
                    it = combined_results.find(curr_result.first);
                    if (it != combined_results.end()) {
                        // found it, append to queue
                        it->second.push(curr_result.second);
                    } else {
                        // not found, create a new queue
                        std::queue<std::vector<human_pose_estimation::HumanPose>> result_q;
                        result_q.push(curr_result.second);
                        combined_results.insert(std::make_pair(curr_result.first, result_q));
                    }
                    mlock.unlock();
                    results_cond.notify_all();
                }
            }
            catch (const std::exception &ex) {
                errMessage = "failed to get inference result: ";
                errMessage.append(ex.what());
                std::cerr << errMessage << std::endl;
            }
        }
    }
}





Scheduler::Scheduler(int numDevices,
                     const std::string& modelXmlPath,
                     const std::string& modelBinPath,
                     const std::string& targetDeviceName)
{
    (void)targetDeviceName;
    terminated = false;
    matchJobIdToWorkerId = false;

    // get ALL inference devices
    std::vector<std::string> availableDevices = ie.GetAvailableDevices();
    if ((numDevices <= 0) || ((int)availableDevices.size() < numDevices))
        numDevices = availableDevices.size();

    // Debug only
    std::cout << "Available devices: " << std::endl;
    for (auto && device : availableDevices) {
        std::cout << "\tDevice: " << device << std::endl;
    }

    for (int i = 0; i < numDevices; i++) {
        workers.push_back(new human_pose_estimation::HumanPoseEstimator(i,
                                                                        modelXmlPath,
                                                                        modelBinPath,
                                                                        availableDevices.at(i)));
    }

    for (auto& worker : this->workers) {
        threads.push_back(std::thread(worker_loop,
                                      std::ref(this->results),
                                      std::ref(this->results_mutex),
                                      std::ref(this->results_cond),
                                      std::ref(this->terminated),
                                      worker));
    }
}


Scheduler::~Scheduler() {
    terminate();
    // need to notify everybody ???
    for (auto& worker : this->workers) {
        worker->jobs_cond->notify_all();
    }
    results_cond.notify_all();
    joinThreads();
    for (auto& worker : this->workers)
        delete worker;
}


bool Scheduler::resultIsReady(int id)
{
    std::unique_lock<std::mutex> mlock(results_mutex);
    std::map<int, std::queue<std::vector<human_pose_estimation::HumanPose>>>::iterator it;
    it = results.find(id);
    return (it != results.end());
}


void Scheduler::waitResult(int id)
{
    std::map<int, std::queue<std::vector<human_pose_estimation::HumanPose>>>::iterator it;
    std::unique_lock<std::mutex> mlock(results_mutex);
    while (!terminated) {
        it = results.find(id);
        if (it != results.end())
            break;
        results_cond.wait(mlock);
    }
}


std::vector<human_pose_estimation::HumanPose> Scheduler::getResult(int id)
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


bool Scheduler::queueJob(int id, unsigned char *image, int width, int height)
{
    bool ret = false;

    if (!job::id_is_valid(id))
        return ret;

    human_pose_estimation::HumanPoseEstimator *nominated_worker = NULL;

    if (!matchJobIdToWorkerId) {
        // mode 1: push job to next available queue
        // find which worker has the most empty queue worker.space queue_available_size()
        int max_queue_size = 0;
        for (auto& worker : this->workers) {
            int q_size = worker->queue_available_size();
            if (q_size > max_queue_size) {
                max_queue_size = q_size;
                nominated_worker = worker;
            }
        }
    } else {
        // mode 2: push job to worker with matchin ID
        for (auto& worker : this->workers) {
            if (worker->get_worker_id() == id) {
                nominated_worker = worker;
                break;
            }
        }
    }

    if (nominated_worker) {

        //std::cout << "found nominated worker" << std::endl;

        int scaled_width = 0;
        int scaled_height = 0;
        get_scaled_image_dimensions(nominated_worker, width, height, &scaled_width, &scaled_height);
        unsigned char *scaled_img = scale_yuv2bgr(image, width, height, scaled_width, scaled_height);
        if (scaled_img) {
            try {
                job::Job *new_job = new job::Job(id, width, height, scaled_width, scaled_height, scaled_img);
        //std::cout << "calling nominated_worker->estimateAsync" << std::endl;
                ret = nominated_worker->estimateAsync(new_job);
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


void Scheduler::clearResults(int id)
{
    std::unique_lock<std::mutex> mlock(results_mutex);
    results.erase(id);
}


void Scheduler::terminate(void)
{
    terminated = true;
}

void Scheduler::joinThreads(void)
{
    for (std::thread & th : threads){
        th.join();
    }
}


}  // namespace scheduler

#endif // 0
