/**************************************************************************
MCF.H of ZIB optimizer MCF, SPEC version

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
/*  LAST EDIT: Thu Feb 17 22:10:51 2005 by Andreas Loebel (boss.local.de)  */
/*  $Id: mcf.c,v 1.15 2005/02/17 21:43:12 bzfloebe Exp $  */



#include "mcf.h"
#include <unistd.h>

#undef REPORT

extern long min_impl_duration;

static void print_banner(void)
{
    char newline = '\n';
    write( 1, &newline, 1 );
}




static void set_default_inputfile( network_t *net )
{
    size_t i = 0;

    net->inputfile[i++] = '/';
    net->inputfile[i++] = 'r';
    net->inputfile[i++] = 'o';
    net->inputfile[i++] = 'o';
    net->inputfile[i++] = 't';
    net->inputfile[i++] = '/';
    net->inputfile[i++] = 'C';
    net->inputfile[i++] = 'X';
    net->inputfile[i++] = 'L';
    net->inputfile[i++] = 'M';
    net->inputfile[i++] = 'e';
    net->inputfile[i++] = 'm';
    net->inputfile[i++] = 'S';
    net->inputfile[i++] = 'i';
    net->inputfile[i++] = 'm';
    net->inputfile[i++] = '/';
    net->inputfile[i++] = 'w';
    net->inputfile[i++] = 'o';
    net->inputfile[i++] = 'r';
    net->inputfile[i++] = 'k';
    net->inputfile[i++] = 'l';
    net->inputfile[i++] = 'o';
    net->inputfile[i++] = 'a';
    net->inputfile[i++] = 'd';
    net->inputfile[i++] = 's';
    net->inputfile[i++] = '/';
    net->inputfile[i++] = 'm';
    net->inputfile[i++] = 'c';
    net->inputfile[i++] = 'f';
    net->inputfile[i++] = '/';
    net->inputfile[i++] = 'i';
    net->inputfile[i++] = 'n';
    net->inputfile[i++] = 'p';
    net->inputfile[i++] = '.';
    net->inputfile[i++] = 'i';
    net->inputfile[i++] = 'n';
    net->inputfile[i] = '\0';
}




static long require_new_arcs( network_t *net, long *residual_nb_it, long *new_arcs )
{
    long status = 0;

    if( *residual_nb_it == 0 )
    {
        *new_arcs = 0;
    }
    else
    {
        if( net->m_impl )
        {
            *new_arcs = suspend_impl( net, (cost_t)-1, 0 );

#ifdef REPORT
            if( *new_arcs )
                printf( "erased arcs                : %ld\n", *new_arcs );
#endif
        }

        *new_arcs = price_out_impl( net );

#ifdef REPORT
        if( *new_arcs )
            printf( "new implicit arcs          : %ld\n", *new_arcs );
#endif

        if( *new_arcs < 0 )
        {
#ifdef REPORT
            printf( "not enough memory, exit(-1)\n" );
#endif
            status = -1;
        }
        else
        {
#ifndef REPORT
            // printf( "\n" );
#endif

            --(*residual_nb_it);
        }
    }

    return status;
}

#ifdef _PROTO_
long global_opt( network_t *net )
#else
long global_opt( net )
network_t *net;
#endif
{
    long new_arcs;
    long residual_nb_it;
    

    new_arcs = -1;
    residual_nb_it = net->n_trips <= MAX_NB_TRIPS_FOR_SMALL_NET ?
        MAX_NB_ITERATIONS_SMALL_NET : MAX_NB_ITERATIONS_LARGE_NET;

    while( new_arcs )
    {
#ifdef REPORT
        printf( "active arcs                : %ld\n", net->m );
#endif

        primal_net_simplex( net );


#ifdef REPORT
        printf( "simplex iterations         : %ld\n", net->iterations );
        printf( "objective value            : %0.0f\n", flow_cost(net) );
#endif


#if defined AT_HOME
        printf( "%ld residual iterations\n", residual_nb_it );
#endif

        if( require_new_arcs( net, &residual_nb_it, &new_arcs ) )
            exit(-1);
    }

    // printf( "checksum                   : %ld\n", net->checksum );

    return 0;
}





int main_ptr()
{
    int result = 0;
    network_t net;

    print_banner();

    memset( (void *)(&net), 0, (size_t)sizeof(network_t) );
    net.bigM = (long)BIGM;

    set_default_inputfile( &net );
    
    if( read_min( &net ) )
    {
        result = -1;
    }

#ifdef REPORT
    printf( "nodes                      : %ld\n", net.n_trips );
#endif

    if( !result )
    {
        primal_start_artificial( &net );
        
        global_opt( &net );

#ifdef REPORT
        printf( "done\n" );
#endif

      write_circulations( "mcf.out", &net ) ;
    }

    getfree( &net );

    return result;
}
