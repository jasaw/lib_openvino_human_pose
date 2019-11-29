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
#include <opencv2/imgproc/imgproc.hpp>

namespace job {

bool id_is_valid(int id);

class Job {
public:
    Job(void);
    Job(int id_, std::shared_ptr<struct timeval> timestamp_, int fullImageWidth, int fullImageHeight,
        int scaledImageWidth, int scaledImageHeight, unsigned char *scaledImage);
    bool is_valid(void);
    int id;
    const cv::Mat scaledImage;
    cv::Size fullImageSize;
    // TODO: add job timestamp as results may be out of order
    static const int invalid_job_id;
    std::shared_ptr<struct timeval> timestamp;
};

}  // namespace job
