/**************************************************************************
PBEAMPP.C of ZIB optimizer MCF, SPEC version

This software was developed at ZIB Berlin. Maintenance and revisions 
solely on responsibility of Andreas Loebel

Dr. Andreas Loebel
Ortlerweg 29b, 12207 Berlin

Konrad-Zuse-Zentrum fuer Informationstechnik Berlin (ZIB)
Scientific Computing - Optimization
Takustr. 7, 14195 Berlin-Dahlem

Copyright (c) 1998-2000 ZIB.           
Copyright (c) 2000-2002 ZIB & Loebel.  
Copyright (c) 2003-2005 Andreas Loebel.
**************************************************************************/
/*  LAST EDIT: Sun Nov 21 16:22:04 2004 by Andreas Loebel (boss.local.de)  */
/*  $Id: pbeampp.c,v 1.10 2005/02/17 19:42:32 bzfloebe Exp $  */



#define K 300
#define B  50



#include "pbeampp.h"

#ifdef MCF_PROFILING
#include "mcf_profiler.h"
#endif

#ifdef MCF_VORTEX_OFFLOAD
#include "mcf_vortex_offload.h"

// Staging buffers for SoA transformation (from liveness analysis)
static int64_t *g_arc_costs = NULL;
static int64_t *g_tail_pots = NULL;
static int64_t *g_head_pots = NULL;
static int32_t *g_arc_idents = NULL;
static int g_offload_initialized = 0;
static long g_num_arcs = 0;

// Initialize offload buffers based on liveness analysis
static void init_offload_buffers(long num_arcs) {
    if (g_offload_initialized && g_num_arcs >= num_arcs) return;

    if (g_arc_costs) free(g_arc_costs);
    if (g_tail_pots) free(g_tail_pots);
    if (g_head_pots) free(g_head_pots);
    if (g_arc_idents) free(g_arc_idents);

    g_arc_costs = (int64_t*)malloc(num_arcs * sizeof(int64_t));
    g_tail_pots = (int64_t*)malloc(num_arcs * sizeof(int64_t));
    g_head_pots = (int64_t*)malloc(num_arcs * sizeof(int64_t));
    g_arc_idents = (int32_t*)malloc(num_arcs * sizeof(int32_t));
    g_num_arcs = num_arcs;
    g_offload_initialized = 1;
}
#endif


#ifdef _PROTO_
int bea_is_dual_infeasible( arc_t *arc, cost_t red_cost )
#else
int bea_is_dual_infeasible( arc, red_cost )
    arc_t *arc;
    cost_t red_cost;
#endif
{
    return(    (((red_cost < 0) & (arc->ident == AT_LOWER)) | ((red_cost > 0) & (arc->ident == AT_UPPER))) );
}







typedef struct basket
{
    arc_t *a;
    cost_t cost;
    cost_t abs_cost;
} BASKET;

static long basket_size;
static BASKET basket[B+K+1];
static BASKET *perm[B+K+1];

#define NR_GROUP_STATE (basket[0].cost)
#define GROUP_POS_STATE (basket[0].abs_cost)


void remote(arc_t *arc, long *basket_size, BASKET *perm[]);

#ifdef _PROTO_
void sort_basket( long min, long max )
#else
void sort_basket( min, max )
    long min, max;
#endif
{
    long l, r;
    cost_t cut;
    BASKET *xchange;

    l = min; r = max;

    cut = perm[ (long)( (l+r) / 2 ) ]->abs_cost;

    do
    {
        while( perm[l]->abs_cost > cut )
            l++;
        while( cut > perm[r]->abs_cost )
            r--;
            
        if( l < r )
        {
            xchange = perm[l];
            perm[l] = perm[r];
            perm[r] = xchange;
        }
        if( l <= r )
        {
            l++; r--;
        }

    }
    while( l <= r );

    if( min < r )
        sort_basket( min, r );
    if( (l < max) & (l <= B) )
        sort_basket( l, max ); 
}






static long initialize = 1;

#ifdef _PROTO_
arc_t *primal_bea_mpp( long m,  arc_t *arcs, arc_t *stop_arcs, 
                              cost_t *red_cost_of_bea )
#else
arc_t *primal_bea_mpp( m, arcs, stop_arcs, red_cost_of_bea )
    long m;
    arc_t *arcs;
    arc_t *stop_arcs;
    cost_t *red_cost_of_bea;
#endif
{
    long i, next, old_group_pos;
    arc_t *arc;
    cost_t red_cost;
    long arcs_priced = 0;

#ifdef MCF_PROFILING
    mcf_profile_primal_bea_mpp_start();
#endif

    if( initialize )
    {
        for( i=1; i < K+B+1; i++ )
            perm[i] = &(basket[i]);
        basket_size = 0;
        initialize = 0;
        NR_GROUP_STATE = 0;
        GROUP_POS_STATE = 0;
    }
    else
    {
        for( i = 2, next = 0; (i <= B) & (i <= basket_size); i++ )
        {
            arc = perm[i]->a;
            red_cost = arc->cost - arc->tail->potential + arc->head->potential;
            if( (((red_cost < 0) & (arc->ident == AT_LOWER)) | ((red_cost > 0) & (arc->ident == AT_UPPER))) )
            {
                next++;
                perm[next]->a = arc;
                perm[next]->cost = red_cost;
                perm[next]->abs_cost = red_cost;
            }
                }   
        basket_size = next;
        }

    if( NR_GROUP_STATE == 0 )
    {
        NR_GROUP_STATE = ( (m-1) / K ) + 1;
        if( NR_GROUP_STATE == 0 )
            NR_GROUP_STATE = 1;
        GROUP_POS_STATE = 0;
    }

    old_group_pos = (long)GROUP_POS_STATE;

    do
    {
        /* price next group */
        arc = arcs + (long)GROUP_POS_STATE;

        // CPU path (original code) - always use for correctness
        for( ; arc < stop_arcs; arc += (long)NR_GROUP_STATE )
        {
            remote(arc, &basket_size, perm);
            arcs_priced++;
        }

        GROUP_POS_STATE = (cost_t)((long)GROUP_POS_STATE + 1);
        if( (long)GROUP_POS_STATE == (long)NR_GROUP_STATE )
            GROUP_POS_STATE = 0;

    /* Avoid short-circuit; evaluate both sides explicitly */
    } while ( ((basket_size < B) & ((long)GROUP_POS_STATE != old_group_pos)) );
	
    if( basket_size == 0 )
    {
        initialize = 1;
        *red_cost_of_bea = 0;
#ifdef MCF_PROFILING
        mcf_profile_primal_bea_mpp_end(arcs_priced);
#endif
        return (arc_t *)0;  // Return NULL to indicate optimality
    }
    
    sort_basket( 1, basket_size );

    *red_cost_of_bea = perm[1]->cost;
#ifdef MCF_PROFILING
    mcf_profile_primal_bea_mpp_end(arcs_priced);
#endif
    return( perm[1]->a );
}



