#include <iostream>

#include "color.h"
#include "ray.h"
#include "vec3.h"

/*
    determine whether a ray hits a sphere and return the parameter
    t of the ray equation P = A + Bt that gives the closest hit point
    Derived from the equation (x-C_x)^2 + (y-C_y)^2 + (z-C_z)^2 = r^2
    and then representing the equation using vectors where the center is
    C = (C_x, C_y, C_z) and the point on the sphere is P = (x, y, z).
    The dot prouct of (P-C) and (P-C) = the initial equation.
    So, (P-C) DOT (P-C) = r^2.
    Then using the fact that our ray can be represented using the equation
    P(t) = A + tb, we plug in and solve for t to determine the number of
    solutions (intersections of the ray and sphere).

    @param center center of the sphere
    @param radius radius of the sphere
    @param r ray to determine if it hits the sphere

    @return parameter t of P = A + Bt of the closest hit point,
            -1 if the ray does not hit the sphere.
*/
double hit_sphere(const point3 &center, double radius, const ray &r) {
    vec3 center_to_origin = r.origin() - center;
    double a = r.direction().length_squared();
    double half_b = dot(r.direction(), center_to_origin);
    double c = center_to_origin.length_squared() - radius * radius;
    double discriminant = half_b * half_b - a * c;

    if (discriminant < 0) {
        return -1.0;
    } else {
        return (-half_b - std::sqrt(discriminant)) / a;
    }
}

/*
    get color for a given scene ray

    @param ray the ray
    
    @return color of the ray
*/
color ray_color(const ray &r) {
    // calculate the sphere normal 
    // from the center of the sphere to the hit point
    double t = hit_sphere(point3(0, 0, -1), 0.5, r);
    if (t > 0.0) {
        vec3 N = unit_vector(r.at(t) - vec3(0, 0, -1));
        return 0.5 * color(N.x() + 1, N.y() + 1, N.z() + 1);
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
            color pixel_color = ray_color(r);
            // write color in ppm format
            write_color(std::cout, pixel_color);
        }
    }

    std::clog << "\rDone                          " << std::endl;
}