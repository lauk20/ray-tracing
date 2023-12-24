#ifndef COLOR_H
#define COLOR_H

#include <iostream>

#include "vec3.h"

using color = vec3;

/*
    write pixel to outputstream

    @param out the output stream
    @param pixel_color the RGB color vector
*/
void write_color(std::ostream &out, color pixel_color) {
    out << static_cast<int>(255.999 * pixel_color.x()) << " "
        << static_cast<int>(255.999 * pixel_color.y()) << " "
        << static_cast<int>(255.999 * pixel_color.z()) << std::endl;
}

#endif