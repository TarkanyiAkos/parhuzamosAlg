//(a^b mod m) kiszámítása.
ulong modexp(ulong base, ulong exp, ulong mod) {
    ulong result = 1;
    base = base % mod; //mivel úgy is csak a maradékot nézzük, azt már a hatványozás előtt vehetjük
    while (exp > 0) {
        if (exp & 1) //ha az aktuális bit 1 = páratlan -> hozzá kel szoroznunk az alapot az eredményhez
            result = (result * base) % mod;
        exp = exp >> 1; //következő bit
        base = (base * base) % mod; //hatvány
    }
    return result;
}

//a Miller–Rabin teszt egy iterácoója 1 tanúval
__kernel void millerRabinTest(__global const ulong *p, __global const ulong *witness, const ulong d, const int s, __global int *results) {
    int i = get_global_id(0); //minden szál azonosítója, ez dönti el hogy melyik tanúval dolgozik
    ulong candidate = p[0]; //a vizsgált szám minden szál számára ugyanaz, globálisból olvassák be
    ulong a = witness[i]; //az i-edik tanú szám beolvasása

    //első lépés a Miller–Rabin tesztben: a^d mod candidate kiszámítása
    ulong x = modexp(a, d, candidate);

    //ha x = 1 vagy candidate-1 -> az adott tanú nem bizonyítja hogy a szám összetett lenne
    if (x == 1 || x == candidate - 1) {
        results[i] = 1; //valószínűleg prím e tanú szerint
        return;
    }

    //különben még s-1 próbálkozás x négyzetre emelésével
    //ha ezalatt eléri a candidate-1 -t, akkor prím
    for (int r = 1; r < s; r++) {
        x = (x * x) % candidate; //x új értéke: x^2 mod candidate
        if (x == candidate - 1) {
            results[i] = 1; //tanú nem bizonyítja hogy x összetett
            return;
        }
        if (x == 1) {
			//előbb jutottunk el 1-hez, x összetett
            results[i] = 0;
            return;
        }
    }

    //nem teljesült feltétel
	//x összetett
    results[i] = 0;
}
