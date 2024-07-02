#ifndef PIXEL_H
#define PIXEL_H

struct Pixel {
    unsigned char b, g, r;

    Pixel(unsigned char b, unsigned char g, unsigned char r)
        : b(b), g(g), r(r) {}
};

#endif