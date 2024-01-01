#ifndef COLOR_H
#define COLOR_H

#include <iostream>

#include "vec3.h"

using color = vec3;

/*
    transform color from linear to gamma space

    @param linear_component the linear color

    @return the gamma corrected color
*/
inline float linear_to_gamma(float linear_component) {
    return sqrt(linear_component);
}

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
    float r = pixel_color.x();
    float g = pixel_color.y();
    float b = pixel_color.z();

    float scale = 1.0 / samples_per_pixel;
    r *= scale;
    g *= scale;
    b *= scale;

    // transform linear to gamma
    // gamma correction
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);


    static const interval intensity(0.00, 0.999);
    out << static_cast<int>(255.999 * intensity.clamp(r)) << " "
        << static_cast<int>(255.999 * intensity.clamp(g)) << " "
        << static_cast<int>(255.999 * intensity.clamp(b)) << std::endl;
}

#endif