#include <iostream>

#include "camera.h"
#include "hittable_list.h"
#include "sphere.h"

__global__ void create_world(hittable_list ** world, curandState * rand_state) {
    int id = getThreadID();
    if (id > 0) return;
    curand_init(12345, id, 0, rand_state);

    *world = new hittable_list();

    lambertian * ground_material = new lambertian(color(0.5, 0.5, 0.5));

    (*world)->add(new sphere(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = random_double(rand_state);
            point3 center(a + 0.9 * random_double(rand_state), 0.2, b + 0.9 * random_double(rand_state));

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material * sphere_material;

                if (choose_mat < 0.8) {
                    color albedo = color(random_double(rand_state), random_double(rand_state), random_double(rand_state));
                    sphere_material = new lambertian(albedo);
                    (*world)->add(new sphere(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    color albedo = color(0.5, 1, random_double(rand_state));
                    float fuzz = random_double(0, 0.5, rand_state);
                    sphere_material = new metal(albedo, fuzz);
                    (*world)->add(new sphere(center, 0.2, sphere_material));
                } else {
                    sphere_material = new dielectric(1.5);
                    (*world)->add(new sphere(center, 0.2, sphere_material));
                }
            }
        }
    }

    dielectric * material1 = new dielectric(1.5);
    (*world)->add(new sphere(point3(0, 1, 0), 1, material1));
    lambertian * material2 = new lambertian(color(0.4, 0.2, 0.1));
    (*world)->add(new sphere(point3(-4, 1, 0), 1, material2));
    metal * material3 = new metal(color(0.7, 0.6, 0.5), 0);
    (*world)->add(new sphere(point3(4, 1, 0), 1, material3));
}

int main() {
    hittable_list ** world = nullptr;
    curandState * rand_state;
    checkCudaErrors(cudaMallocManaged((void**)&rand_state, sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void**)&world, sizeof(hittable_list*)));
    create_world<<<1, 1>>>(world, rand_state);
    checkCudaErrors(cudaDeviceSynchronize());

    // camera
    camera cam;
    cam.aspect_ratio = 16.0/9.0;
    cam.image_width = 1200;
    cam.samples_per_pixel = 10;
    cam.max_depth = 50;

    cam.vfov = 20;
    cam.lookfrom = point3(13, 2, 3);
    cam.lookat = point3(0, 0, 0);
    cam.vup = vec3(0, 1, 0);

    cam.defocus_angle = 0.0;
    cam.focus_dist = 10.0;

    // render scene
    cam.render(world);
    checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}