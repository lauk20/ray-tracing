#include <iostream>

#include "color.h"
#include "vec3.h"

int main() {
    // dimension of the image
    int width = 256;
    int height = 256;

    // write ppm file to stdout

    std::cout << "P3" << std::endl << width << " " << height << std::endl << 255 << std::endl;

    for (int i = 0; i < height; i++) {
        std::clog << "\rScanlines remaining: " << (height - i) << " " << std::flush;
        for (int j = 0; j < width; j++) {
            color pixel_color = color(double(j) / (width - 1), double(i) / (height - 1), 0);
            write_color(std::cout, pixel_color);
        }
    }

    std::clog << "\rDone                          \n";
}