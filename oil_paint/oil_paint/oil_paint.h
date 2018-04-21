#ifndef OIL_PAINT_H
#define OIL_PAINT_H

#include <device_launch_parameters.h>
uchar4* oil(uchar4* d_in, unsigned char* d_intensity, size_t numRows, size_t numCols, int radius);
#endif
