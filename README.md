# Python Raytracer

![Demo Image](/example.png)

This is a simple raytracer implementation written in Python. It can render basic 3D scenes by tracing rays from the camera through each pixel of the output image and calculating the color of the object that the ray intersects with.

To run the raytracer, simply execute the following command in your terminal:

```
python raytracer.py --size 100
```

The `--size` argument specifies the size of the output image in pixels. For example, setting `--size` to 100 will produce a 100x100 image. The resulting image will be opened in a your default image viewer.

The implementation supports basic lighting and shading models, such as diffuse and specular reflections, as well as shadows. It currently includes several geometric primitives, spheres and planes.

Feel free to fork this repository and experiment with the code yourself. Happy rendering!
