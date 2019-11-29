/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Author: Joo Saw
**
** -------------------------------------------------------------------------*/

#include "job.hpp"

namespace job {
const int Job::invalid_job_id = -1;

bool id_is_valid(int id) {
    return id >= 0;
}


Job::Job(void) : id(invalid_job_id)
{
}


Job::Job(int id_, std::shared_ptr<struct timeval> timestamp_, int fullImageWidth, int fullImageHeight,
         int scaledImageWidth, int scaledImageHeight, unsigned char *scaledImage)
    : id(id_),
      scaledImage(cv::Mat(scaledImageHeight, scaledImageWidth, CV_8UC3, scaledImage)),
      fullImageSize(cv::Size(fullImageWidth, fullImageHeight)),
      timestamp(timestamp_)
{
}


bool Job::is_valid(void)
{
    return id_is_valid(id);
}


}  // namespace job
