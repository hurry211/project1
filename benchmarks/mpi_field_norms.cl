#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// One work item integrates one physical P1 triangle of an actual rank export.
__kernel void field_norms(__global const double *nodes, __global const int *cells,
                          __global const double *rule, int points, __global double2 *squares)
{
    int e = get_global_id(0);
    double3 a = vload3(cells[3*e], nodes);
    double3 b = vload3(cells[3*e+1], nodes) - a;
    double3 c = vload3(cells[3*e+2], nodes) - a;
    double det = b.x*c.y - c.x*b.y;
    double ux = (b.z*c.y - c.z*b.y)/det;
    double uy = (c.z*b.x - b.z*c.x)/det;
    double2 sum = (double2)(0.0);
    for (int q = 0; q < points; ++q) {
        double3 r = vload3(q, rule);
        double3 v = a + r.x*b + r.y*c;
        double sx = sinpi(v.x), sy = sinpi(v.y);
        double du = v.z - sx*sy;
        double dx = ux - M_PI*cospi(v.x)*sy;
        double dy = uy - M_PI*sx*cospi(v.y);
        sum += r.z * (double2)(du*du, dx*dx + dy*dy);
    }
    squares[e] = sum * det;
}
