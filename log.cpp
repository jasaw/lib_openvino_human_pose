/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Author: Joo Saw
**
** -------------------------------------------------------------------------*/

#include "log.hpp"

std::string errMessage;

const char *log_err_msg(void)
{
    return errMessage.c_str();
}
