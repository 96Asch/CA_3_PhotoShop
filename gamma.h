/*
 * Skeleton code for use with Computer Architecture assignment 3, LIACS,
 * Leiden University.
 */

#ifndef GAMMA_H
#define GAMMA_H

/* Look-up-table that converts 8-bit linear integer values to 2.2-gamma
 * corrected values.
 */
#ifdef __CUDACC__
__device__
#endif
static unsigned short GAMMA[256] = {
    /*0*/     0, 21, 28, 34, 39, 43, 46, 50, 53, 56,
    /*10*/    59, 61, 64, 66, 68, 70, 72, 74, 76, 78,
    /*20*/    80, 82, 84, 85, 87, 89, 90, 92, 93, 95,
    /*30*/    96, 98, 99, 101, 102, 103, 105, 106, 107, 109,
    /*40*/    110, 111, 112, 114, 115, 116, 117, 118, 119, 120,
    /*50*/    122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    /*60*/    132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
    /*70*/    142, 143, 144, 144, 145, 146, 147, 148, 149, 150,
    /*80*/    151, 151, 152, 153, 154, 155, 156, 156, 157, 158,
    /*90*/    159, 160, 160, 161, 162, 163, 164, 164, 165, 166,
    /*100*/   167, 167, 168, 169, 170, 170, 171, 172, 173, 173,
    /*110*/   174, 175, 175, 176, 177, 178, 178, 179, 180, 180,
    /*120*/   181, 182, 182, 183, 184, 184, 185, 186, 186, 187,
    /*130*/   188, 188, 189, 190, 190, 191, 192, 192, 193, 194,
    /*140*/   194, 195, 195, 196, 197, 197, 198, 199, 199, 200,
    /*150*/   200, 201, 202, 202, 203, 203, 204, 205, 205, 206,
    /*160*/   206 ,207 ,207 ,208 ,209 ,209 ,210 ,210 ,211 ,212,
    /*170*/   212 ,213 ,213 ,214 ,214 ,215 ,215 ,216 ,217 ,217,
    /*180*/   218 ,218 ,219 ,219 ,220 ,220 ,221 ,221 ,222 ,223,
    /*190*/   223, 224, 224, 225, 225, 226, 226, 227, 227, 228,
    /*200*/   228, 229, 229, 230, 230, 231, 231, 232, 232, 233,
    /*210*/   233, 234, 234, 235, 235, 236, 236, 237, 237, 238,
    /*220*/   238, 239, 239, 240, 240, 241, 241, 242, 242, 243,
    /*230*/   243, 244, 244, 245, 245, 246, 246, 247, 247, 248,
    /*240*/   248, 249, 249, 249, 250, 250, 251, 251, 252, 252,
    /*250*/   253, 253, 254, 254, 255, 255 };

#endif
