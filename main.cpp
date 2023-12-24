#include <iostream>

int main() {
    // dimension of the image
    int width = 256;
    int height = 256;

    // write ppm file to stdout

    std::cout << "P3" << std::endl << width << " " << height << std::endl << 255 << std::endl;

    for (int i = 0; i < height; i++) {
        std::clog << "\rScanlines remaining: " << (height - i) << " " << std::flush;
        for (int j = 0; j < width; j++) {
            double red = static_cast<double>(j) / (width - 1);
            double green = static_cast<double>(i) / (height - 1);
            double blue = 0;

            int cred = static_cast<int>(255.999 * red);
            int cgreen = static_cast<int>(255.999 * green);
            int cblue = static_cast<int>(255.999 * blue);

            std::cout << cred << " " << cgreen << " " << cblue << std::endl;
        }
    }
}