#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>

#include "constants.h"
#include "color.h"
#include "hittable.h"
#include "material.h"

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

        // vertical field of view
        double vfov = 90;
        // where camera is looking from
        point3 lookfrom = point3(0, 0, -1);
        // where camera is looking at
        point3 lookat = point3(0, 0, 0);
        // camera's up direction, relative to the world
        vec3 vup = vec3(0, 1, 0);

        double defocus_angle = 0; // angle of the defocus "cone" (base at lens, peak of cone at viewport center)
        double focus_dist = 10; // distance from lookfrom point to plane of perfect focus

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
        vec3 u, v, w; // basis vector for camera frame
        vec3 defocus_disk_u; // defocus disk horizontal radius
        vec3 defocus_disk_v; // defocus disk vertical radius

        /*
            initialize camera and image properties
        */
        void initialize() {
            // calculate image height from width and aspect ratio
            // the height of the image needs to be at least 1
            image_height = static_cast<int>(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            camera_center = lookfrom;

            // set up viewport dimensions
            double theta = degrees_to_radians(vfov);
            double h = tan(theta/2);
            double viewport_height = 2 * h * focus_dist;
            double viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);
            
            // calculate basis vectors for camera coordinate frame
            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            // vectors across and down the viewport edges
            vec3 viewport_u = viewport_width * u;
            vec3 viewport_v = viewport_height * -v;

            // delta vectors between pixels
            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            // location of the upper left corner of the viewport
            point3 viewport_upper_left = camera_center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
            // location of the upper left pixel within the viewport
            // we decided that the pixel inset is 0.5 of the pixel deltas
            pixel00_location = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

            // calculate camera defocus disk basis vectors
            double defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;
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
            // the bounced ray may start from "inside" the object causing the
            // program to think it hit the edge of the object again
            // we fix this by ignoring rays that hit the object when
            // their t parameter in P = A + Bt is small.
            // accounting for the floating point errors
            // this error is called "shadow acne"
            if (world.hit(r, interval(0.001, infinity), rec)) {
                //
                // get the diffuse reflection of the object
                // using randomized uniform scattering
                // vec3 direction = random_on_hemisphere(rec.normal);
                // ** UNCOMMENT IF WANT TO USE UNIFORM DIFFUSE SCATTER **

                // get diffuse reflection
                // using Lambertian distribution
                // vec3 direction = rec.normal + random_unit_vector();
                // ** UNCOMMENT IF WANT TO USE LAMBERTIAN
                // ** LAMBERTIAN IS NOW IN ITS OWN MATERIAL DERIVED CLASS

                // the scattered ray
                ray scattered;
                // attentuation is how much of each RGB value 
                // the object retains for each scatter/hit
                color attenuation;
                if (rec.mat->scatter(r, rec, attenuation, scattered))
                    return attenuation * ray_color(scattered, depth - 1, world);

                // bounce the ray until it doesn't hit anything
                // and get 1/2 of that color
                //return 0.5 * ray_color(ray(rec.p, direction), depth - 1, world);

                return color(0, 0, 0);
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

            // get random sample camera ray for pixel at i,j originating from camera defocus disk
            point3 ray_origin = (defocus_angle <= 0) ? camera_center : defocus_disk_sample();
            vec3 ray_direction = pixel_sample - ray_origin;

            return ray(ray_origin, ray_direction);
        }

        /*
            get defocus disk ray origin

            @return point3 origin of a defocused ray
        */
        point3 defocus_disk_sample() const {
            // return random point in camera defocus disk
            vec3 p = random_in_unit_disk();
            return camera_center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
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