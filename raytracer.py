import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import random

def curve(x, b = 2):
    return 1/(1 + (x/(1 - x)) ** (-b))

def reflect_ray(rayDirection, surfaceNormal):
    return sub_vecs(rayDirection, scale_vec(surfaceNormal, 2 * dot_product(rayDirection, surfaceNormal)))

def length(a, b):
    dist = sub_vecs(a, b)
    return np.sqrt(dot_product(dist, dist))

def magnitude(vec):
    return np.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])

def square_magnitude(vec):
    return vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]

def normalize(vec):
    mag = magnitude(vec)
    return (vec[0]/mag, vec[1]/mag, vec[2]/mag)

def dot_product(vec1, vec2):
    return (vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2])

def scale_vec(vec, s):
    return (vec[0] * s, vec[1] * s, vec[2] * s)

def sub_vecs(vec1, vec2):
    return (vec1[0] - vec2[0], vec1[1] - vec2[1], vec1[2] - vec2[2])

def add_vecs(vec1, vec2):
    return (vec1[0] + vec2[0], vec1[1] + vec2[1], vec1[2] + vec2[2])

def mul_vecs(vec1, vec2):
    return (vec1[0] * vec2[0], vec1[1] * vec2[1], vec1[2] * vec2[2])

def lerp(a, b, t):
    return a + (b - a) * t

def lerp_vecs(vec1, vec2, t):
    return (
        lerp(vec1[0], vec2[0], t),
        lerp(vec1[1], vec2[1], t),
        lerp(vec1[2], vec2[2], t)
    )


def randomize_direction(normal, amount):
    a = random.random() * 2 - 1
    b = random.random() * 2 -1
    c = random.random() * 2 - 1
    return normalize((normal[0] + a * amount, normal[1] + b * amount, normal[2] + c * amount))


def random_in_unit_sphere():
    p = (random.random() * 2 - 1, random.random() * 2 - 1, random.random() * 2 - 1)
    while (square_magnitude(p) >= 1.0):
        p = (random.random() * 2 - 1, random.random() * 2 - 1, random.random() * 2 - 1)
    return p

def rgb_to_real(x):
    """ Converts a color value from 0-255 to 0-1"""
    return (x[0]/255.0, x[1]/255.0, x[2]/255.0)

def real_to_rgb(rgb_float):
    """ Converts a color value from 0.0-1.0 to 0-255"""
    return ((int) (rgb_float[0] * 255), (int) (rgb_float[1] * 255), (int) (rgb_float[2] * 255))

class Light(object):
    def __init__(self, position, color, max_range):
        self.position = position
        self.color = color
        self.maxRange = max_range


class Scene(object):
    """
    Contains all of the models and lighting.
    """
    def __init__(self, light, ambient_light_value):
        self._models = []
        self._ambient_light_value = ambient_light_value
        self._light = light

    def get_light_color(self, ray_origin, ray_direction, max_length=1000):
        for model in self._models:
            (did_intersect, intersection_distance, surface_normal) = model.get_intersection(ray_origin, ray_direction, max_length)
            if (did_intersect and intersection_distance < max_length): 
                return self._ambient_light_value
        return add_vecs(self._light.color, self._ambient_light_value)


    def get_color(self, ray_origin, ray_direction, max_length=1000, bounces=10):

        """ Gets the color of the ray, taking reflections into account.

        This function calculates the appropriate color for the ray.
        You can set the desiered number of reflection bounces, and 
        the length of the ray.
        """

        if (bounces <= 0): return (0, 0, 0)

        closest_intersection_distance = float("inf")
        closeset_color = (1,1,1)
        closest_surface_normal = (1, 0, 0)
        
        # Finds the closest item to the camera, this ensures far away
        # objects aren't overlapping closer objects.
        for model in self._models:
            (did_intersect, intersection_distance, surface_normal) = model.get_intersection(ray_origin, ray_direction, max_length)
            if (did_intersect and intersection_distance < closest_intersection_distance):
                closest_intersection_distance = intersection_distance
                intersection_point = add_vecs(ray_origin, scale_vec(ray_direction, intersection_distance))
                closeset_color = model.get_color_at_point(intersection_point)
                closest_surface_normal = surface_normal
                closest_center_item = model.position
        if (closest_intersection_distance == float("inf")):
            return self._ambient_light_value
        
        intersection_point = add_vecs(ray_origin, scale_vec(ray_direction, closest_intersection_distance))
        surface_normal = closest_surface_normal

        heading_to_light = sub_vecs(self._light.position, intersection_point)
        direction_to_light = normalize(heading_to_light)
        surface_dot = dot_product(surface_normal, direction_to_light)
        surface_angle_with_light = max(0, surface_dot)

        light_dir = scale_vec(direction_to_light, -1)

        # Camera view direction, this can probably be pulled out, to enable the camera
        # direction to be changed, but it's fine like this for now.
        view_dir = (0, 0, 1)

        # This just implements the following formulat, then uses that to calculate the specular
        # intensity.
        # r = 2n(n \dot l) - l
        inner1 = scale_vec(surface_normal, 2)
        inner2 = dot_product(surface_normal, light_dir)
        reflection_vector = sub_vecs(scale_vec(inner1, inner2), light_dir) 
        specular = scale_vec((1, 1, 1), min(max(0.0, dot_product(reflection_vector, view_dir)) ** 200, 1000))
        
        # This just pushes the intersecion point off the suface a little bit, to 
        # prevent spurious errors from happening duiring lighting calculations.
        o_intersection = add_vecs(intersection_point, scale_vec(surface_normal, 0.001))

        current_light_color = self.get_light_color(o_intersection, direction_to_light, magnitude(heading_to_light))
        light_color = scale_vec(current_light_color, surface_angle_with_light)
        
        ob_color = mul_vecs(closeset_color, light_color)
        reflection_ray = reflect_ray(ray_direction, surface_normal) 

        reflection_color = self.get_color(o_intersection, reflection_ray, max_length, bounces-1)
        
        ob_color = add_vecs(scale_vec(reflection_color, 0.5), ob_color)
        ob_color = add_vecs(ob_color, specular)

        return ob_color

    def add_model(self, model):
        self._models.append(model)



class Model(object):
    """
    Base scene object class.
    """
    def __init__(self, position, color):
        self._color = color
        self.position = position
    def get_intersection(self, rayOrigin, rayDirection, maxLength):
        return (False, 0, (1, 0, 0))
    def get_color(self):
        return self._color
    def get_color_at_point(self, point):
        return self._color

    
class Sphere(Model):
    """
    A simple sphere model.
    """
    def __init__(self, color, position, radius):
        self._radius = radius
        self._sqrRadius =radius * radius
        super().__init__(position, color)

    def get_intersection(self, ray_origin, ray_direction, maxLength):
        t0 = 0
        t1 = 1 # solution for t if the ray intersects
        L = sub_vecs(self.position, ray_origin)

        #L = sub_vecs(self.position, ray_origin)
        tca = dot_product(L, ray_direction)
        d2 = dot_product(L, L) - tca * tca
        if d2 > self._sqrRadius: 
            return (False, 0, (1, 0, 0))
        thc = np.sqrt(self._sqrRadius - d2)
        t0 = tca - thc
        t1 = tca + thc

        if (t0 > t1):
            temp = t0
            t0 = t1
            t1 = temp

        if (t0 < 0):
            t0 = t1
            if (t0 < 0): 
                return (False, 0, (1, 0,0))
        t = t0

        intersection_point = add_vecs(ray_origin, scale_vec(ray_direction , t))
        surface_normal = normalize(sub_vecs(intersection_point, self.position))
        return (True, t, surface_normal)

class Plane(Model):
    """
    A plane model
    """
    def __init__(self, color, position, surface_normal):
        self._surface_normal = surface_normal
        super().__init__(position, color)

    def get_intersection(self, ray_origin, ray_direction, maxLength):

        nom = dot_product(self._surface_normal, sub_vecs(self.position, ray_origin))
        base = dot_product(self._surface_normal, ray_direction)
        if (base <= 0): return (False, 0, (1, 0, 0))

        intersection_point = add_vecs(ray_origin, scale_vec(ray_direction, nom/base))


        return (True, length(ray_origin, intersection_point), scale_vec(self._surface_normal, -1))
    
    def get_color_at_point(self, point):
        size = 20
        step1 = int((point[0] + 1000) / size) % 2
        step2 = int((point[2] + 1000) / size) % 2
        if (step1 == step2):return (0.9, 0.9, 0.9)
        else: return (0.4, 0.4, 0.4)
    
        

class PixelGrid(object):

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._pixels = [(0,0,0) for x in range(width * height)]

    def set_pixel(self, x, y, color):
        self._pixels[y * self.width + x] = color

    def get_pixel(self, x, y):
        return self._pixels[y * self.width + x]

    def get_pixels(self):
        return self._pixels
    
    def anti_alias(self):
        kernel_size = 3
        output = [(0, 0, 0) for x in range(self.width * self.height)]
        kernel = [0.5, 1, 0.5, 1, 3, 1, 0.5, 1, 0.5]
        inv_kernal_total = 1.0/(sum(kernel))
        for i in range(0, self.width - kernel_size):
            for j in range(0, self.height - kernel_size):
                new_pixel_value = (0, 0, 0)
                for k in range(0, kernel_size):
                    for l in range(0, kernel_size):
                        target_pixel = self._pixels[(j + l) * self.width + i + k]
                        target_pixel = scale_vec(target_pixel, kernel[l * kernel_size + k])
                        new_pixel_value = add_vecs(new_pixel_value, target_pixel)
                output[j * self.width + i] = scale_vec(new_pixel_value, inv_kernal_total)
        self._pixels = output


class Camera(object):
    """
    Camera object, this renders the scene.
    """
    def __init__(self, pixel_data, focal_length, apature_angle):
        self._pixelData = pixel_data
        self._position =(0,0,0) 
        self._focal_length = focal_length
        self._apature_angle = apature_angle
        pass
    def render(self, scene):
        xOffset = -(self._pixelData.width * self._apature_angle * 0.5)
        yOffset = -(self._pixelData.height * self._apature_angle * 0.5)
        
        jitterMatrix = [
            -1.0/4.0,  3.0/4.0,
            3.0/4.0,  1.0/3.0,
            -3.0/4.0, -1.0/4.0,
            1.0/4.0, -3.0/4.0,
        ]
        for x in (pbar := tqdm(range(self._pixelData.width))):
            pbar.set_description("Rendering")
            for y in range(self._pixelData.height):

                color_value = (0, 0, 0)
                for sample in range(0, 4):
                    ray_offset = (xOffset + (x + jitterMatrix[sample])  * self._apature_angle, 
                                  yOffset + (y + jitterMatrix[sample + 1]) * self._apature_angle, 
                                  self._focal_length)
                    ray_direction = normalize(ray_offset)
                    ray_origin = self._position
                    color_value = add_vecs(color_value, scene.get_color(ray_origin, ray_direction))

                self._pixelData.set_pixel(x, y, scale_vec(color_value, 0.25))


    def render_no_anti_alias(self, scene):
        xOffset = -(self._pixelData.width * self._apature_angle * 0.5)
        yOffset = -(self._pixelData.height * self._apature_angle * 0.5)
        
        for x in (pbar := tqdm(range(self._pixelData.width))):
            pbar.set_description("Rendering")
            for y in range(self._pixelData.height):
                ray_offset = (xOffset + x * self._apature_angle, yOffset + y * self._apature_angle, self._focal_length)
                ray_direction = normalize(ray_offset)
                ray_origin = self._position
                color = scene.get_color(ray_origin, ray_direction)

                self._pixelData.set_pixel(x, y, color)

def create_image(pixel_data, should_normalize_pixel_data=False):

    if (should_normalize_pixel_data):
        max_value = 0;
        for pixel in pixel_data.get_pixels():
            if (pixel[0] > max_value): max_value = pixel[0]
            elif (pixel[1] > max_value): max_value = pixel[1]
            elif (pixel[2] > max_value): max_value = pixel[2]

        pixels = pixel_data.get_pixels()
        for i in range(len(pixels)):
            pass
            pixels[i] = (pixels[i][0]/max_value, pixels[i][1]/max_value, pixels[i][2]/max_value)


    image = Image.new("RGB", (pixel_data.width, pixel_data.height), color=(200, 100, 20))
    img = image.load()
    for i in (pbar := tqdm(range(pixel_data.width))):
        pbar.set_description("Generating Image")
        for j in range(pixel_data.height):
            img[i,j] = real_to_rgb(pixel_data.get_pixel(i,j))
    image.show()


def main(size):
    print(f"Generating with size: {size}x{size}")

    # Defines some simple colors, for use in the models.
    c1 = rgb_to_real((23, 190, 187))
    c2 = rgb_to_real((205, 83, 52))
    c3 = rgb_to_real((250, 150, 214))
    c4 = rgb_to_real((248, 198, 48))
    c5 = rgb_to_real((46, 40, 42))

    # The main scene light.
    light = Light((150, -200, 30), (0.9,0.9,0.9), 10000)

    # Create the scene, this is what we'll add everything to.
    scene = Scene(light, c5)
    #scene = Scene(light, (0.4, 0.4, 0.5))

    # Create some models, then add them to the scene
    plane1 = Plane((0,1,1), (0,-10,10), (0, 1, 0))
    sphereTest = Sphere(c1, (10, -5, 100), 10)
    sphereTest2 = Sphere(c2, (-20, -14, 150), 20)
    sphereTest3 = Sphere(c3, (-5, -20, 120), 10)
    sphereTest4 = Sphere(c4, (0, 2, 80), 3)

    scene.add_model(sphereTest)
    scene.add_model(sphereTest2)
    scene.add_model(sphereTest3)
    scene.add_model(sphereTest4)
    scene.add_model(plane1)

    # Create a pixel grid, and a camera, passing the pixel grid
    # into the camera constructor.
    pixel_grid = PixelGrid(size, size)
    camera = Camera(pixel_grid, 60, 300.0/pixel_grid.width * 0.1)

    # This renders the camera, then creates the final image from the pixel grid.
    camera.render(scene)
    create_image(pixel_grid)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, required=True)
    args = parser.parse_args()
    main(args.size)

if __name__ == "__main__":
    parse_args()
