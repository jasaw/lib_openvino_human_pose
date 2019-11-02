/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Author: Joo Saw
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
#include "alt_detect.h"

static human_pose_estimation::HumanPoseEstimator *estimator = NULL;


static void humanPoseToLines(const std::vector<human_pose_estimation::HumanPose>& poses, alt_detect_result_t *alt_detect_result)
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
        std::cerr << "Error: failed to allocate memory for results" << std::endl;
        return;
    }
    alt_detect_result->num_objs = 0;

    for (const auto& pose : poses) {
        alt_detect_obj_t *cur_obj = &alt_detect_result->objs[alt_detect_result->num_objs];
        cur_obj->score = pose.score;
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

static cv::Mat YuvToBgr(unsigned char *pBuffer, int width, int height)
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


#if 0
static void debug_write_raw_image(cv::Mat &img)
{
    cv::Size imageSize = img.size();

    std::cout << "imageSize.width = " << imageSize.width << std::endl;
    std::cout << "imageSize.height = " << imageSize.height << std::endl;

    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);

    try {
        cv::imwrite("libopenvinohumanpose.png", img, compression_params);
    }
    catch (std::runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    }
}
#endif


// image in YUV420 format
// return 0 on success
int alt_detect_process(unsigned char *image, int width, int height)
{
    cv::Mat img = YuvToBgr(image, width, height);

    //debug_write_raw_image(img);

    estimator->estimateAsync(img);
    return 0;
}


int alt_detect_result_ready(void)
{
    if (estimator->resultIsReady())
        return 1;
    return 0;
}


// caller frees memory by calling alt_detect_free_results
int alt_detect_get_result(alt_detect_result_t *alt_detect_result)
{
    if (alt_detect_result == NULL)
        return 0;

    memset(alt_detect_result, 0, sizeof(alt_detect_result_t));
    if (estimator->resultIsReady()) {
        std::vector<human_pose_estimation::HumanPose> poses = estimator->getResult();
        humanPoseToLines(poses, alt_detect_result);
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
    struct stat st;

    if (estimator)
        return -1;

    // read model XML and BIN and target device from config file
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
            }
        }

    } else {
        std::cerr << "Couldn't open " << config_file << " config file" << std::endl;
    }
    std::cout << "loaded config file "<< config_file << std::endl;

    if (stat(_modelXmlPath.c_str(), &st) != 0)
    {
        std::cerr << "Error: " << _modelXmlPath << " does not exist" << std::endl;
        return -1;
    }
    if (stat(_modelBinPath.c_str(), &st) != 0)
    {
        std::cerr << "Error: " << _modelBinPath << " does not exist" << std::endl;
        return -1;
    }

    estimator = new human_pose_estimation::HumanPoseEstimator(_modelXmlPath, _modelBinPath, _targetDeviceName);
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
