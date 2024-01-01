#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>

#include "constants.h"
#include "color.h"
#include "cuda_tools.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"

/*
    get color for a given scene ray

    @param ray the ray
    
    @return color of the ray
*/
__device__ color ray_color(const ray&r, hittable_list ** world, ray &scattered, curandState * rand_state) {
    hit_record rec;

    // we use interval(0.001, infinity) instead of
    // interval(0, infinity) because of floating point errors
    // the bounced ray may start from "inside" the object causing the
    // program to think it hit the edge of the object again
    // we fix this by ignoring rays that hit the object when
    // their t parameter in P = A + Bt is small.
    // accounting for the floating point errors
    // this error is called "shadow acne"
    if ((*world)->hit(r, interval(0.001f, FLT_MAX), rec)) {
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

        // attentuation is how much of each RGB value 
        // the object retains for each scatter/hit
        color attenuation;
        if (rec.mat->scatter(r, rec, attenuation, scattered, rand_state))
            return attenuation;

        // bounce the ray until it doesn't hit anything
        // and get 1/2 of that color
        //return 0.5 * ray_color(ray(rec.p, direction), depth - 1, world);

        scattered = ray(point3(0, 0, 0), vec3(0, 0, 0));
        return color(0, 0, 0);
    }

    // background ("sky")
    vec3 unit_direction = unit_vector(r.direction());
    float a = 0.5f * (unit_direction.y() + 1.0f);
    scattered = ray(point3(0, 0, 0), vec3(0, 0, 0));
    // linear interpolate the color based on y-direction
    return (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
}

/*
    get random point in square surrounding pixel at the origin
    this is used as an offset when get_ray is called.

    @return point in square surrounding pixel at the origin
*/
__device__ point3 pixel_sample_square(const vec3 &pixel_delta_u, const vec3 &pixel_delta_v, curandState * rand_state) {
    float px = -0.50f + curand_uniform(rand_state) - 0.00001f;
    float py = -0.50f + curand_uniform(rand_state) - 0.00001f;

    return (px * pixel_delta_u) + (py * pixel_delta_v);
}

/*
    get defocus disk ray origin

    @return point3 origin of a defocused ray
*/
__device__ point3 defocus_disk_sample(const vec3 &camera_center, const vec3 &defocus_disk_u, const vec3 &defocus_disk_v, curandState * rand_state) {
    // return random point in camera defocus disk
    vec3 p = random_in_unit_disk(rand_state);
    return camera_center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
}

/*
    get randomly sampled ray from camera to pixel at i, j

    @param i pixel location column
    @param j pixel location row

    @return ray from camera to the randomly sampled point
*/
__device__ ray get_ray(int i, int j, const vec3 &camera_center, const vec3 &pixel00_location, const vec3 &pixel_delta_u,
                       const vec3 &pixel_delta_v, const vec3 &defocus_disk_u, const vec3 &defocus_disk_v, float defocus_angle, curandState * rand_state) {
    point3 pixel_center = pixel00_location + (i * pixel_delta_u) + (j * pixel_delta_v);
    // pixel center offset by a random amount within the square area
    // of the pixel_center
    point3 pixel_sample = pixel_center + pixel_sample_square(pixel_delta_u, pixel_delta_v, rand_state);

    // get random sample camera ray for pixel at i,j originating from camera defocus disk
    point3 ray_origin = (defocus_angle <= 0) ? camera_center : defocus_disk_sample(camera_center, defocus_disk_u, defocus_disk_v, rand_state);
    vec3 ray_direction = pixel_sample - ray_origin;

    return ray(ray_origin, ray_direction);
}

__global__ void render_kernel(color * raster, hittable_list ** world, int max_depth, int image_width, int image_height,
                              int samples_per_pixel, vec3 center, vec3 pixel00_location, float defocus_angle, vec3 pixel_delta_u,
                              vec3 pixel_delta_v, vec3 defocus_disk_u, vec3 defocus_disk_v, curandState * rand_state, int num_samples) {
    int id = getThreadID();
    if (id >= num_samples) return;

    curandState * local_rand_state = rand_state + id;
    int pixel_index = id / samples_per_pixel;
    int i = pixel_index % image_width; // col
    int j = pixel_index / image_width; // row

    ray r;
    ray scattered;
    color attenuation(1, 1, 1);

    r = get_ray(i, j, center, pixel00_location, pixel_delta_u, pixel_delta_v, defocus_disk_u, defocus_disk_v, defocus_angle, local_rand_state);
    for (int depth = 0; depth < max_depth; depth++) {
        //printf("indepth %d\n", depth);
        attenuation = attenuation * ray_color(r, world, scattered, local_rand_state);
        //printf("afterattenuation %d\n", depth);
        r = scattered;
        if (r.direction().near_zero())
            break;
        if (depth == max_depth - 1)
            attenuation = color(0, 0, 0);
    }
    //printf("ended\n");
    atomicAdd(&raster[pixel_index].m[0], attenuation.m[0]);
    atomicAdd(&raster[pixel_index].m[1], attenuation.m[1]);
    atomicAdd(&raster[pixel_index].m[2], attenuation.m[2]);
    //raster[pixel_index] = raster[pixel_index] + attenuation;
}

class camera {
    public:
        // dimension of the image and
        // aspect ratio of the image
        float aspect_ratio = 16.0 / 9.0;
        int image_width = 400;

        // samples for each pixel
        int samples_per_pixel = 10;

        // max ray bounces
        int max_depth = 10;

        // vertical field of view
        float vfov = 90;
        // where camera is looking from
        point3 lookfrom = point3(0, 0, -1);
        // where camera is looking at
        point3 lookat = point3(0, 0, 0);
        // camera's up direction, relative to the world
        vec3 vup = vec3(0, 1, 0);

        float defocus_angle = 0; // angle of the defocus "cone" (base at lens, peak of cone at viewport center)
        float focus_dist = 10; // distance from lookfrom point to plane of perfect focus

        /*
            render the scene
            output image in ppm format to std

            @param world the list of hittable objects
        */
        void render(hittable_list ** world) {
            // initialize viewport and image properties
            initialize();
            int n = image_width * image_height * samples_per_pixel;
            dim3 grid_size((n + 127) / 128);
            dim3 block_size(128);

            render_kernel<<<grid_size, block_size>>>(raster, world, max_depth, image_width, image_height, samples_per_pixel, camera_center,
                                                    pixel00_location, defocus_angle, pixel_delta_u, pixel_delta_v, defocus_disk_u, defocus_disk_v, rand_state, n);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            // begin render
            std::cout << "P3" << std::endl << this->image_width << " " << this->image_height << std::endl;
            std::cout << 255 << std::endl;

            for (int j = 0; j < image_height; j++) {
                for (int i = 0; i < image_width; i++) {
                    size_t pixel_index = j * image_width + i;
                    color pixel_color = raster[pixel_index];
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
        vec3 * raster = nullptr;
        size_t raster_size = 0;
        curandState * rand_state = nullptr;

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
            float theta = degrees_to_radians(vfov);
            float h = tan(theta/2);
            float viewport_height = 2 * h * focus_dist;
            float viewport_width = viewport_height * (static_cast<float>(image_width) / image_height);
            
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
            float defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;

            // allocate CUDA memory
            int pixel_count = image_width * image_height;
            raster_size = pixel_count * sizeof(color);
            checkCudaErrors(cudaMallocManaged((void**)&raster, raster_size));
            int random_count = pixel_count * samples_per_pixel;
            checkCudaErrors(cudaMallocManaged((void**)&rand_state, random_count * sizeof(curandState)));
            dim3 grid_size = (random_count + 127) / 128;
            dim3 block_size = 128;
            initRandState<<<grid_size, block_size>>>(rand_state, random_count);
            checkCudaErrors(cudaDeviceSynchronize());
        }
};

#endif