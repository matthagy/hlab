
#include "opt.h"
#include "vector.h"
#define vec_t Vec3D
#include "xperiodic.h"


void
CE_count_neighbors(double r_neighbor,
                   int N, vec_t * CEX_RESTRICT positions, 
                   vec_t box_size,
                   int * CEX_RESTRICT neighbors)
{
	double r_neighbor_sqr;
	vec_t position_i, position_j, r, box_size_2;
	int i,j, i_N;

        r_neighbor_sqr = r_neighbor * r_neighbor;
	Vec3_MUL(box_size_2, box_size, 0.5);
	for (i=0; i<N; i++) {
		i_N = i*N;
		position_i = positions[i];
		for (j=i+1; j<N; j++) {
			position_j = positions[j];
                        int mirrored = 0;
                        XPERIODIC_SEPARATION_VECTOR(r, position_i, position_j,
                                                    box_size, box_size_2, mirrored);
                        if (Vec3_SQR(r) <= r_neighbor_sqr) {
                                neighbors[i] ++;
                                neighbors[j] ++;
                        }
		}
	}
}

