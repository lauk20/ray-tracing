#include <iostream>

#include "constants.h"
#include "color.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"

/*
    get color for a given scene ray

    @param ray the ray
    
    @return color of the ray
*/
color ray_color(const ray &r, const hittable &world) {
    hit_record rec;
    if (world.hit(r, interval(0, infinity), rec)) {
        return 0.5 * (rec.normal + color(1,1,1));
    }

    vec3 unit_direction = unit_vector(r.direction());
    double a = 0.5 * (unit_direction.y() + 1.0);
    // linear interpolate the color based on y-direction
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

int main() {
    // dimension of the image and
    // aspect ratio of the image
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    // calculate image height from width and aspect ratio
    // the height of the image needs to be at least 1
    int image_height = static_cast<int>(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // world of objects
    hittable_list world;
    world.add(make_shared<sphere>(point3(0, 0, -1), 0.5));
    world.add(make_shared<sphere>(point3(0, -100.5, -1), 100));

    // set up camera
    double focal_length = 1.0; // distance from camera to viewport
    double viewport_height = 2.0;
    double viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);
    point3 camera_center = point3(0, 0, 0);

    // vectors across and down the viewport edges
    vec3 viewport_u = vec3(viewport_width, 0, 0);
    vec3 viewport_v = vec3(0, -viewport_height, 0);

    // delta vectors between pixels
    vec3 pixel_delta_u = viewport_u / image_width;
    vec3 pixel_delta_v = viewport_v / image_height;

    // location of the upper left corner of the viewport
    point3 viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
    // location of the upper left pixel within the viewport
    // we decided that the pixel inset is 0.5 of the pixel deltas
    point3 pixel00_location = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // render
    // write ppm file to stdout

    std::cout << "P3" << std::endl << image_width << " " << image_height << std::endl << 255 << std::endl;

    for (int i = 0; i < image_height; i++) {
        std::clog << "\rScanlines remaining: " << (image_height - i) << " " << std::flush;
        for (int j = 0; j < image_width; j++) {
            // get center of pixel based on pixel deltas
            point3 pixel_center = pixel00_location + (j * pixel_delta_u) + (i * pixel_delta_v);
            // get direction of the vector camera to pixel center
            vec3 ray_direction = pixel_center - camera_center;
            // get ray from camera center to the pixel center
            ray r(camera_center, ray_direction);
            
            // calculate color of the ray
            color pixel_color = ray_color(r, world);
            // write color in ppm format
            write_color(std::cout, pixel_color);
        }
    }

    std::clog << "\rDone                          " << std::endl;
}