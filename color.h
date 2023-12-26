#ifndef COLOR_H
#define COLOR_H

#include <iostream>

#include "vec3.h"

using color = vec3;

/*
    write pixel to outputstream
    pixel_color parameter should be the sum of 
    the color of all surrounding pixels.
    
    we find the color we want to display by finding
    the average of all the samples we take.

    @param out the output stream
    @param pixel_color the RGB color vector
    @param samples_per_pixel the number of samples taken for this pixel
*/
void write_color(std::ostream &out, color pixel_color, int samples_per_pixel) {
    double r = pixel_color.x();
    double g = pixel_color.y();
    double b = pixel_color.z();

    double scale = 1.0 / samples_per_pixel;
    r *= scale;
    g *= scale;
    b *= scale;

    static const interval intensity(0.00, 0.999);
    out << static_cast<int>(255.999 * intensity.clamp(r)) << " "
        << static_cast<int>(255.999 * intensity.clamp(g)) << " "
        << static_cast<int>(255.999 * intensity.clamp(b)) << std::endl;
}

#endif