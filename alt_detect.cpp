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
#include <map>
#include <sys/stat.h>
extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/parseutils.h>
#include <libswscale/swscale.h>
}
#include <opencv2/opencv.hpp>
#include "human_pose_estimator.hpp"
#include "alt_detect.h"

static human_pose_estimation::HumanPoseEstimator *estimator = NULL;
static std::string errMessage;


static void get_scaled_image_dimensions(int width, int height, int *scaled_width, int *scaled_height)
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


const char *alt_detect_err_msg(void)
{
    return errMessage.c_str();
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


static void humanPoseToLines(const std::vector<human_pose_estimation::HumanPose>& poses,
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

    int num_poses = poses.size();
    alt_detect_result->objs = new alt_detect_obj_t[num_poses];
    if (alt_detect_result->objs == NULL) {
        errMessage = "failed to allocate memory for results";
        return;
    }
    memset(alt_detect_result->objs, 0, sizeof(alt_detect_obj_t)*num_poses);
    alt_detect_result->num_objs = 0;

    for (const auto& pose : poses) {
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
int alt_detect_process_yuv420(unsigned char *image, int width, int height)
{
    int ret = -1;
    int scaled_width = 0;
    int scaled_height = 0;
    get_scaled_image_dimensions(width, height, &scaled_width, &scaled_height);
    unsigned char *scaled_img = scale_yuv2bgr(image, width, height, scaled_width, scaled_height);
    if (scaled_img) {
        cv::Mat scaled_mat(scaled_height, scaled_width, CV_8UC3, scaled_img);
        //save_image_as_png(scaled_mat, "libopenvinohumanpose.png");
        try {
            estimator->estimateAsync(scaled_mat);
            ret = 0;
        }
        catch (const std::exception &ex) {
            errMessage = "failed to queue inference: ";
            errMessage.append(ex.what());
        }
        av_freep(&scaled_img);
    }
    return ret;
}


int alt_detect_queue_empty(void)
{
    if (estimator->queueIsEmpty())
        return 1;
    return 0;
}


int alt_detect_result_ready(void)
{
    try {
        if (estimator->resultIsReady())
            return 1;
    }
    catch (const std::exception &ex) {
        errMessage = "failed to get inference result status: ";
        errMessage.append(ex.what());
        return -1;
    }
    return 0;
}


// caller frees memory by calling alt_detect_free_results
int alt_detect_get_result(float score_threshold,
                          int width, int height,
                          alt_detect_result_t *alt_detect_result)
{
    if (alt_detect_result == NULL)
        return -1;

    try {
        if (estimator->resultIsReady()) {
            int scaled_width = 0;
            int scaled_height = 0;
            get_scaled_image_dimensions(width, height, &scaled_width, &scaled_height);
            cv::Size orgImageSize(width, height);
            cv::Size scaledImageSize(scaled_width, scaled_height);
            std::vector<human_pose_estimation::HumanPose> poses = estimator->getResult(orgImageSize, scaledImageSize);
            alt_detect_free_result(alt_detect_result);
            humanPoseToLines(poses, score_threshold, alt_detect_result);
        }
    }
    catch (const std::exception &ex) {
        errMessage = "failed to get inference result: ";
        errMessage.append(ex.what());
        return -1;
    }
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
    int _numDevices = 1;
    struct stat st;

    if (estimator)
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

        estimator = new human_pose_estimation::HumanPoseEstimator(_modelXmlPath, _modelBinPath, _targetDeviceName);
    }

    catch (const std::exception &ex) {
        errMessage = "failed to initialize human pose estimator: ";
        errMessage.append(ex.what());
        return -1;
    }
    return 0;
}


void alt_detect_uninit(void)
{
    if (estimator)
    {
        delete estimator;
        estimator = NULL;
    }
}
