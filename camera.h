#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>

#include "constants.h"
#include "color.h"
#include "hittable.h"

class camera {
    public:
        // dimension of the image and
        // aspect ratio of the image
        double aspect_ratio = 16.0 / 9.0;
        int image_width = 400;

        // samples for each pixel
        int samples_per_pixel = 10;

        // max ray bounces
        int max_depth = 10;

        /*
            render the scene
            output image in ppm format to std

            @param world the list of hittable objects
        */
        void render(const hittable &world) {
            // initialize viewport and image properties
            initialize();

            // begin render
            std::cout << "P3" << std::endl << image_width << " " << image_height << std::endl << 255 << std::endl;

            for (int i = 0; i < image_height; i++) {
                std::clog << "\rScanlines remaining: " << (image_height - i) << " " << std::flush;
                for (int j = 0; j < image_width; j++) {
                    color pixel_color(0, 0, 0);
                    // add color samples
                    for (int sample = 0; sample < samples_per_pixel; sample++) {
                        ray r = get_ray(j, i);
                        pixel_color += ray_color(r, max_depth, world);
                    }

                    write_color(std::cout, pixel_color, samples_per_pixel);
                }
            }

            std::clog << "\rDone                          " << std::endl;
        }
    
    private:
        int image_height;
        point3 camera_center;
        point3 pixel00_location;
        vec3 pixel_delta_u;
        vec3 pixel_delta_v;

        /*
            initialize camera and image properties
        */
        void initialize() {
            // calculate image height from width and aspect ratio
            // the height of the image needs to be at least 1
            image_height = static_cast<int>(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            // set up camera
            double focal_length = 1.0; // distance from camera to viewport
            double viewport_height = 2.0;
            double viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);
            camera_center = point3(0, 0, 0);

            // vectors across and down the viewport edges
            vec3 viewport_u = vec3(viewport_width, 0, 0);
            vec3 viewport_v = vec3(0, -viewport_height, 0);

            // delta vectors between pixels
            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            // location of the upper left corner of the viewport
            point3 viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
            // location of the upper left pixel within the viewport
            // we decided that the pixel inset is 0.5 of the pixel deltas
            pixel00_location = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
        }

        /*
            get color for a given scene ray

            @param ray the ray
            
            @return color of the ray
        */
        color ray_color(const ray&r, int depth, const hittable &world) const {
            hit_record rec;

            if (depth <= 0) {
                return color(0, 0, 0);
            }

            // we use interval(0.001, infinity) instead of
            // interval(0, infinity) because of floating point errors
            // the bounced ray may be "inside" the object causing the
            // program to think it hit the edge of the object again
            // we fix this by ignoring rays that hit the object when
            // their t parameter in P = A + Bt is small.
            // accounting for the floating point errors
            // this error is called "shadow acne"
            if (world.hit(r, interval(0.001, infinity), rec)) {
                // get the diffuse reflection of the object
                vec3 direction = random_on_hemisphere(rec.normal);
                // bounce the ray until it doesn't hit anything
                // and get 1/2 of that color
                return 0.5 * ray_color(ray(rec.p, direction), depth - 1, world);
            }

            // background ("sky")
            vec3 unit_direction = unit_vector(r.direction());
            double a = 0.5 * (unit_direction.y() + 1.0);
            // linear interpolate the color based on y-direction
            return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
        }

        /*
            get randomly sampled ray from camera to pixel at i, j

            @param i pixel location column
            @param j pixel location row

            @return ray from camera to the randomly sampled point
        */
        ray get_ray(int i, int j) const {
            point3 pixel_center = pixel00_location + (i * pixel_delta_u) + (j * pixel_delta_v);
            // pixel center offset by a random amount within the square area
            // of the pixel_center
            point3 pixel_sample = pixel_center + pixel_sample_square();

            point3 ray_origin = camera_center;
            vec3 ray_direction = pixel_sample - ray_origin;

            return ray(ray_origin, ray_direction);
        }

        /*
            get random point in square surrounding pixel at the origin
            this is used as an offset when get_ray is called.

            @return point in square surrounding pixel at the origin
        */
        point3 pixel_sample_square() const {
            double px = -0.5 + random_double();
            double py = -0.5 + random_double();

            return (px * pixel_delta_u) + (py * pixel_delta_u);
        }
};

#endif