/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Copyright: Joo Aun Saw
**
** -------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include "human_pose_estimator.hpp"
#include "log.hpp"
#include "alt_detect.h"

static human_pose_estimation::HumanPoseEstimator *sched = NULL;


const char *alt_detect_err_msg(void)
{
    return log_err_msg();
}


//static const char *humanPoseIdToName(int id)
//{
//    // Result for BODY_25 (25 body parts consisting of COCO + foot)
//    const std::map<int, std::string> POSE_BODY_25_BODY_PARTS = {
//        {0,  "Nose"},
//        {1,  "Neck"},
//        {2,  "RShoulder"},
//        {3,  "RElbow"},
//        {4,  "RWrist"},
//        {5,  "LShoulder"},
//        {6,  "LElbow"},
//        {7,  "LWrist"},
//        {8,  "MidHip"},
//        {9,  "RHip"},
//        {10, "RKnee"},
//        {11, "RAnkle"},
//        {12, "LHip"},
//        {13, "LKnee"},
//        {14, "LAnkle"},
//        {15, "REye"},
//        {16, "LEye"},
//        {17, "REar"},
//        {18, "LEar"},
//        {19, "LBigToe"},
//        {20, "LSmallToe"},
//        {21, "LHeel"},
//        {22, "RBigToe"},
//        {23, "RSmallToe"},
//        {24, "RHeel"},
//        {25, "Background"}
//    };
//    std::map<int, std::string>::const_iterator it;
//    it = POSE_BODY_25_BODY_PARTS.find(id);
//    if (it != POSE_BODY_25_BODY_PARTS.end())
//        return it->second.c_str();
//    return NULL;
//}


static void humanPoseToLines(const std::pair<std::shared_ptr<struct timeval>, std::vector<human_pose_estimation::HumanPose>>& rp,
                             float score_threshold,
                             alt_detect_result_t *alt_detect_result)
{
    const std::vector<std::pair<int, int> > limbKeypointsIds = {
        {1, 2},  {1, 5},   {2, 3},
        {3, 4},  {5, 6},   {6, 7},
        {1, 8},  {8, 9},   {9, 10},
        {1, 11}, {11, 12}, {12, 13},
        {1, 0},  {0, 14},  {14, 16},
        {0, 15}, {15, 17}
    };

    const cv::Point2f absentKeypoint(-1.0f, -1.0f);

    int num_poses = rp.second.size();
    alt_detect_result->objs = new alt_detect_obj_t[num_poses];
    if (alt_detect_result->objs == NULL) {
        errMessage = "failed to allocate memory for results";
        return;
    }
    memset(alt_detect_result->objs, 0, sizeof(alt_detect_obj_t)*num_poses);
    alt_detect_result->timestamp.tv_sec = rp.first->tv_sec;
    alt_detect_result->timestamp.tv_usec = rp.first->tv_usec;
    alt_detect_result->num_objs = 0;

    for (const auto& pose : rp.second) {
        alt_detect_obj_t *cur_obj = &alt_detect_result->objs[alt_detect_result->num_objs];
        cur_obj->score = pose.score;
        if (cur_obj->score < score_threshold)
            continue;
        cur_obj->lines = new alt_detect_line_t[human_pose_estimation::HumanPoseEstimator::keypointsNumber];
        memset(cur_obj->lines, 0, sizeof(alt_detect_line_t)*human_pose_estimation::HumanPoseEstimator::keypointsNumber);
        cur_obj->num_lines = 0;
        for (const auto& limbKeypointsId : limbKeypointsIds) {
            std::pair<cv::Point2f, cv::Point2f> limbKeypoints(pose.keypoints[limbKeypointsId.first],
                    pose.keypoints[limbKeypointsId.second]);
            if (limbKeypoints.first == absentKeypoint
                    || limbKeypoints.second == absentKeypoint) {
                continue;
            }
            int line_index = cur_obj->num_lines;
            cur_obj->lines[line_index].p[0].x = limbKeypoints.first.x;
            cur_obj->lines[line_index].p[0].y = limbKeypoints.first.y;
            cur_obj->lines[line_index].p[0].id = limbKeypointsId.first;
            cur_obj->lines[line_index].p[1].x = limbKeypoints.second.x;
            cur_obj->lines[line_index].p[1].y = limbKeypoints.second.y;
            cur_obj->lines[line_index].p[1].id = limbKeypointsId.second;
            cur_obj->num_lines++;
        }
        alt_detect_result->num_objs++;
    }
}


#define CLIP(X) ( (X) > 255 ? 255 : (X) < 0 ? 0 : X)
// YCbCr -> RGB
#define CYCbCr2R(Y, Cb, Cr) CLIP( Y + ( 91881 * Cr >> 16 ) - 179 )
#define CYCbCr2G(Y, Cb, Cr) CLIP( Y - (( 22544 * Cb + 46793 * Cr ) >> 16) + 135)
#define CYCbCr2B(Y, Cb, Cr) CLIP( Y + (116129 * Cb >> 16 ) - 226 )

static cv::Mat Yuv420ToBgr(unsigned char *pBuffer, int width, int height)
{
    cv::Mat result(height,width,CV_8UC3);
    unsigned char y;
    unsigned char cb;
    unsigned char cr;
    unsigned char r;
    unsigned char g;
    unsigned char b;

    long ySize = width * height;
    long uSize = ySize >> 2;

    unsigned char *output = result.data;
    unsigned char *pY = pBuffer;
    unsigned char *pU = pY+ySize;
    unsigned char *pV = pU+uSize;

    for (int yy = 0; yy < height; yy++) {
        for (int x = 0; x < width; x++) {
            y = pY[yy*width+x];
            cb = pU[(yy>>1)*(width>>1) + (x>>1)];
            cr = pV[(yy>>1)*(width>>1) + (x>>1)];
            b = CYCbCr2B(y,cb,cr);
            g = CYCbCr2G(y,cb,cr);
            r = CYCbCr2R(y,cb,cr);
            *output++=b;
            *output++=g;
            *output++=r;
        }
    }
    return result;
}


static int save_image_as_png(cv::Mat &img, const char *filename)
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


int alt_detect_save_yuv420(unsigned char *image, int width, int height, const char *filename)
{
    cv::Mat img = Yuv420ToBgr(image, width, height);
    return save_image_as_png(img, filename);
}


int alt_detect_render_save_yuv420(unsigned char *image, int width, int height,
                                  alt_detect_result_t *alt_detect_result,
                                  const char *filename)
{
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
        cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
        cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
        cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
        cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
        cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85)
    };
    const int stickWidth = 4;

    cv::Mat img = Yuv420ToBgr(image, width, height);
    cv::Mat pane = img.clone();
    for (int i = 0; i < alt_detect_result->num_objs; i++) {
        alt_detect_obj_t *cur_obj = &alt_detect_result->objs[i];
        for (int j = 0; j < cur_obj->num_lines; j++) {
            alt_detect_line_t *cur_line = &cur_obj->lines[j];
            cv::Point2f Keypoint1(cur_line->p[0].x, cur_line->p[0].y);
            cv::Point2f Keypoint2(cur_line->p[1].x, cur_line->p[1].y);
            std::pair<cv::Point2f, cv::Point2f> limbKeypoints(Keypoint1, Keypoint2);
            float meanX = (limbKeypoints.first.x + limbKeypoints.second.x) / 2;
            float meanY = (limbKeypoints.first.y + limbKeypoints.second.y) / 2;
            cv::Point difference = limbKeypoints.first - limbKeypoints.second;
            double length = std::sqrt(difference.x * difference.x + difference.y * difference.y);
            int angle = static_cast<int>(std::atan2(difference.y, difference.x) * 180 / CV_PI);
            std::vector<cv::Point> polygon;
            cv::ellipse2Poly(cv::Point2d(meanX, meanY), cv::Size2d(length / 2, stickWidth),
                             angle, 0, 360, 1, polygon);
            cv::fillConvexPoly(pane, polygon, colors[cur_line->p[1].id]);
        }
    }
    cv::addWeighted(img, 0.4, pane, 0.6, 0, img);
    return save_image_as_png(img, filename);
}


// image in YUV420 format
// return 0 on success
int alt_detect_process_yuv420(int id, struct timeval *timestamp,
                              unsigned char *image, int width, int height)
{
    if (sched->queueJob(id, timestamp, image, width, height))
        return 0;
    return -1;
}


int alt_detect_result_ready(int id)
{
    if (sched->resultIsReady(id))
        return 1;
    return 0;
}


// caller frees memory by calling alt_detect_free_results
int alt_detect_get_result(int id, float score_threshold,
                          alt_detect_result_t *alt_detect_result)
{
    if (alt_detect_result == NULL)
        return -1;
    std::pair<std::shared_ptr<struct timeval>, std::vector<human_pose_estimation::HumanPose>> rp = sched->getResult(id);
    alt_detect_free_result(alt_detect_result);
    humanPoseToLines(rp, score_threshold, alt_detect_result);
    return alt_detect_result->num_objs;
}


// safe to call with null pointer
void alt_detect_free_result(alt_detect_result_t *alt_detect_result)
{
    if (alt_detect_result) {
        if (alt_detect_result->objs) {
            for (int i = 0; i < alt_detect_result->num_objs; i++) {
                if (alt_detect_result->objs[i].lines) {
                    delete alt_detect_result->objs[i].lines;
                    alt_detect_result->objs[i].lines = NULL;
                }
                alt_detect_result->objs[i].num_lines = 0;
                if (alt_detect_result->objs[i].points) {
                    delete alt_detect_result->objs[i].points;
                    alt_detect_result->objs[i].points = NULL;
                }
                alt_detect_result->objs[i].num_points = 0;
            }
            delete alt_detect_result->objs;
            alt_detect_result->objs = NULL;
        }
        alt_detect_result->num_objs = 0;
    }
}


int alt_detect_init(const char *config_file)
{
    std::string _modelXmlPath("human-pose-estimation-0001.xml");
    std::string _modelBinPath("human-pose-estimation-0001.bin");
    std::string _targetDeviceName("MYRIAD");
    // Setting matchJobIdToWorkerId to true locks one job source to one worker
    // to guarantee sequential output for the job. Each job source needs to have
    // a unique ID.
    // e.g. worker 3 will only process jobs with source ID 3.
    // Setting matchJobIdToWorkerId to false will queue the job the next
    // available worker. Output is not guaranteed to be sequential as some
    // workers may be faster than others.
    bool matchJobIdToWorkerId = false;
    int _numDevices = 0; // zero means use all available inference devices
    int queueSize = 1; // per worker
    struct stat st;

    if (sched)
        return -1;

    try {
        // read model XML and BIN and target device from config file
        if (config_file) {
            std::ifstream cFile(config_file);
            if (cFile.is_open()) {
                std::string line;
                while (getline(cFile, line)) {
                    line.erase(std::remove_if(line.begin(), line.end(), isspace),
                                         line.end());
                    if(line[0] == '#' || line.empty())
                        continue;
                    auto delimiterPos = line.find("=");
                    std::string name = line.substr(0, delimiterPos);
                    std::string value = line.substr(delimiterPos + 1);
                    //std::cout << name << " " << value << '\n';
                    if (name == "MODEL_XML") {
                        _modelXmlPath = value;
                    } else if (name == "MODEL_BIN") {
                        _modelBinPath = value;
                    } else if (name == "TARGET_DEVICE") {
                        _targetDeviceName = value;
                    } else if (name == "NUM_DEVICES") {
                        _numDevices = std::stoi(value);
                    } else if (name == "MATCH_JOB_WORKER_ID") {
                        if (value == "true")
                            matchJobIdToWorkerId = true;
                    } else if (name == "WORKER_QUEUE_SIZE") {
                        queueSize = std::stoi(value);
                    }
                }
            } else {
                errMessage = "failed to open config file: ";
                errMessage.append(config_file);
            }
            //std::cout << "loaded config file "<< config_file << std::endl;
        }

        if (stat(_modelXmlPath.c_str(), &st) != 0)
        {
            errMessage = "model xml file " + _modelXmlPath + " does not exist";
            return -1;
        }
        if (stat(_modelBinPath.c_str(), &st) != 0)
        {
            errMessage = "model bin file " + _modelBinPath + " does not exist";
            return -1;
        }

        sched = new human_pose_estimation::HumanPoseEstimator(matchJobIdToWorkerId,
                                                              queueSize,
                                                              _numDevices,
                                                              _modelXmlPath,
                                                              _modelBinPath,
                                                              _targetDeviceName);
    }

    catch (const std::exception &ex) {
        errMessage = "failed to initialize scheduler for human pose estimator: ";
        errMessage.append(ex.what());
        return -1;
    }
    return 0;
}


void alt_detect_uninit(void)
{
    if (sched)
    {
        delete sched;
        sched = NULL;
    }
}
