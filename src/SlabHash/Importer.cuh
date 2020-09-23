#ifndef IMPORTER_H
#define IMPORTER_H

#include <string>

struct Voxel {
    int indices[3];
    static const float SIZE = 0.5f;
};

struct Triangle {
    float vertices[3][3];
};

// Bounding box is defined in terms of the starting voxel index, and the number of
// voxels that it includes in each direction
// Then there is a 3 dimensional array of atleast size, which would be used to mark
// if a triangle occupies that particular voxel
struct BoundingBox {
    int start_i[3];
    int size[3];
    bool ***occupied;

    __device__ void setOccupied(Voxel v);
};

struct Mesh {
    Triangle *triangles = nullptr;
    int numTriangles = 0;
    BoundingBox box;
};

Mesh import(std::string fileName);

#endif /* IMPORTER_H */