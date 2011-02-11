
#include <math.h>
#include <assert.h>

#include "opt.h"
#include "mem.h"
#include "debug.h"

void
CHLAB_acc_periodic_rs(int * restrict rs, int N_rs, double r_min, double r_prec,
                      const double * restrict positions, int N_positions,
                      const double * restrict box_size)
{
  double x_h = 0.5 * box_size[0];
  double y_h = 0.5 * box_size[1];
  double z_h = 0.5 * box_size[1];

  for (int i=0; i<N_positions; i++) {
    double x_i = positions[i*3    ];
    double y_i = positions[i*3 + 1];
    double z_i = positions[i*3 + 2];

    for (int j=i+1; j<N_positions; j++) {
      double x_r = fabs(positions[j*3    ] - x_i);
      double y_r = fabs(positions[j*3 + 1] - y_i);
      double z_r = fabs(positions[j*3 + 2] - z_i);

      if (unlikely(x_r > x_h)) x_r -= 2.0 * x_h;
      if (unlikely(y_r > y_h)) y_r -= 2.0 * y_h;
      if (unlikely(z_r > z_h)) z_r -= 2.0 * z_h;

      double r = sqrt(x_r*x_r + y_r*y_r + z_r*z_r);
      
      int inx = (int)floor((r-r_min) / r_prec);

      if (inx < 0) Fatal("bad r %.4g for r_min=%.4g r_prec=%.4g N_rs=%dxo", r, r_min, r_prec, N_rs);

      if (inx < N_rs) rs[inx] ++;
    }
  }
}


void
CHLAB_acc_periodic_orient(double * restrict r_orient_acc, int * restrict r_Ns, 
                          int N_rs, double r_min, double r_prec,
                          const double * restrict positions,
                          const double * restrict orients,
                          int N_positions,
                          const double * restrict box_size)
{
  double x_h = 0.5 * box_size[0];
  double y_h = 0.5 * box_size[1];
  double z_h = 0.5 * box_size[1];

  for (int i=0; i<N_positions; i++) {
    double x_i = positions[i*3    ];
    double y_i = positions[i*3 + 1];
    double z_i = positions[i*3 + 2];

    double o_x_i = orients[i*3    ];
    double o_y_i = orients[i*3 + 1];
    double o_z_i = orients[i*3 + 2];

    for (int j=i+1; j<N_positions; j++) {
      double x_r = fabs(positions[j*3    ] - x_i);
      double y_r = fabs(positions[j*3 + 1] - y_i);
      double z_r = fabs(positions[j*3 + 2] - z_i);

      if (unlikely(x_r > x_h)) x_r -= 2.0 * x_h;
      if (unlikely(y_r > y_h)) y_r -= 2.0 * y_h;
      if (unlikely(z_r > z_h)) z_r -= 2.0 * z_h;

      double r = sqrt(x_r*x_r + y_r*y_r + z_r*z_r);
      
      int inx = (int)floor((r-r_min) / r_prec);

      if (inx < 0) Fatal("bad r %.4g for r_min=%.4g r_prec=%.4g N_rs=%d", r, r_min, r_prec, N_rs);

      if (inx < N_rs) {
        r_Ns[inx] ++;

        double o_x_j = orients[j*3    ];
        double o_y_j = orients[j*3 + 1];
        double o_z_j = orients[j*3 + 2];

        double o = o_x_i * o_x_j + o_y_i * o_y_j + o_z_i * o_z_j;
        r_orient_acc[inx] += o;
      }
    }
  }
}
