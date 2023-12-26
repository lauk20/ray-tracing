#include <iostream>

#include "camera.h"
#include "hittable_list.h"
#include "sphere.h"

int main() {
    // world of objects
    hittable_list world;
    world.add(make_shared<sphere>(point3(0, 0, -1), 0.5));
    world.add(make_shared<sphere>(point3(0, -100.5, -1), 100));

    // camera
    camera cam;
    cam.aspect_ratio = 16.0/9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;

    // render scene
    cam.render(world);

    return 0;
}