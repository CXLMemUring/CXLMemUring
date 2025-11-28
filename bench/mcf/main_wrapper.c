/* Main wrapper for MCF benchmark */

#include <stdio.h>
#include <stdlib.h>
#include "defines.h"

/* External main_ptr function from mcf.c */
extern int main_ptr(void);

/* B constant from pbeampp.h */
#define B 50

/* BASKET structure for arc pricing */
typedef struct basket {
    arc_t *a;
    cost_t cost;
    cost_t abs_cost;
    long number;
} BASKET;

/* remote function - compute reduced cost for arc and add to basket if candidate */
void remote(arc_t *arc, long *basket_size, BASKET *perm[])
{
    cost_t red_cost;

    /* Compute reduced cost: c_ij - pi_i + pi_j */
    red_cost = arc->cost - arc->tail->potential + arc->head->potential;

    /* Check dual infeasibility:
     * - If arc at lower bound (ident == 1) and red_cost < 0 -> candidate
     * - If arc at upper bound (ident == 2) and red_cost > 0 -> candidate
     */
    if (((red_cost < 0) && (arc->ident == AT_LOWER)) ||
        ((red_cost > 0) && (arc->ident == AT_UPPER))) {

        /* Add to basket if space available */
        if (*basket_size < B) {
            (*basket_size)++;
            perm[*basket_size]->a = arc;
            perm[*basket_size]->cost = red_cost;
            perm[*basket_size]->abs_cost = (red_cost >= 0) ? red_cost : -red_cost;
        }
    }
}

int main(int argc, char *argv[])
{
    /* Call the original main_ptr function */
    return main_ptr();
}
