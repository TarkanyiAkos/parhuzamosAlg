__kernel void sample_kernel(__global int* buffer, int n)
{
    int value = 11;
    #ifdef SET_ME
    value = SET_ME;
    #endif
    if (get_global_id(0) < n) {
        buffer[get_global_id(0)] = value;
    }
}
