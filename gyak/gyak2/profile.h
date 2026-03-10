#ifndef PROFILE_H
#define PROFILE_H

#include <CL/cl.h>

double event_ms(cl_event ev);
void   write_csv_hello_profile(const char* path,
                               double t_write_ms,
                               double t_kernel_ms,
                               double t_read_ms);

#endif //PROFILE_H