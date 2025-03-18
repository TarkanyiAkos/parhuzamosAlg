#pragma OPENCL EXTENSION cl_khr_int64 : enable

//modular exponentiation: (base^exp) mod mod
ulong modexp(ulong base, ulong exp, ulong mod) {
    ulong result = 1;
    base = base % mod;
    while(exp > 0) {
        if(exp & 1)
            result = (result * base) % mod;
        exp = exp >> 1; //shift bit by 1
        base = (base * base) % mod;
    }
    return result;
}

//Miller-Rabin teszt egy tanura
__kernel void millerRabinTest(__global const ulong *p, __global const ulong *witness, const ulong d, const int s, __global int *results) {
    int i = get_global_id(0);   //item id
    ulong candidate = p[0];     //jelolt primszam a globalis memoriabol
    ulong a = witness[i];      //ehhez tartozo tanu szam
    ulong x = modexp(a, d, candidate);  //x = a^d mod candidate kiszamolasa
    if(x == 1 || x == candidate - 1) {   //ha x = 1 vagy candidate-1, a teszt sikeres erre  a tanura
        results[i] = 1;        //valoszinuleg prim
        return;
    }
    //loop: tovabbi tesztek
    for(int r = 1; r < s; r++) {
        x = (x * x) % candidate;
        if(x == candidate - 1) { //Ha modositott x = candidate-1, a teszt sikeres
            results[i] = 1;     //Valszeg prim
            return;
        }
        if(x == 1) {     //Ha modositott x = 1 hamarabb mint a masik, a teszt sikeres
            results[i] = 0;  //x osszetett
            return;
        }
    }
    results[i] = 0;  //ha egyik feltetel sincs meg, osszetett
}