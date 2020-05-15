/* Implementation of PSO using OpenMP.
 *
 * Author: Naga Kandasamy
 * Date: May 2, 2020
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "pso.h"

/* Solve PSO */
int pso_solve_omp(char *function, swarm_t *swarm, 
                    float xmax, float xmin, int max_iter, int thr_cnt)
{
    int i, j, iter, g;
    float w, c1, c2;
    float r1, r2;
    float curr_fitness;
    particle_t *particle, *gbest;

    w = 0.79;
    c1 = 1.49;
    c2 = 1.49;
    iter = 0;
    g = -1;
    while (iter < max_iter) {
#pragma omp parallel for num_threads(thr_cnt) if(thr_cnt > 1)
        for (i = 0; i < swarm->num_particles; i++) {
            particle = &swarm->particle[i];
            gbest = &swarm->particle[particle->g];  /* Best performing particle from last iteration */ 
            for (j = 0; j < particle->dim; j++) {   /* Update this particle's state */
                r1 = (float)rand()/(float)RAND_MAX;
                r2 = (float)rand()/(float)RAND_MAX;
                /* Update particle velocity */
                particle->v[j] = w * particle->v[j]\
                                 + c1 * r1 * (particle->pbest[j] - particle->x[j])\
                                 + c2 * r2 * (gbest->x[j] - particle->x[j]);
                /* Clamp velocity */
                if ((particle->v[j] < -fabsf(xmax - xmin)) || (particle->v[j] > fabsf(xmax - xmin))) 
                    particle->v[j] = uniform(-fabsf(xmax - xmin), fabsf(xmax - xmin));

                /* Update particle position */
                particle->x[j] = particle->x[j] + particle->v[j];
                if (particle->x[j] > xmax)
                    particle->x[j] = xmax;
                if (particle->x[j] < xmin)
                    particle->x[j] = xmin;
            } /* State update */
            
            /* Evaluate current fitness */
            pso_eval_fitness(function, particle, &curr_fitness);

            /* Update pbest */
            if (curr_fitness < particle->fitness) {
                particle->fitness = curr_fitness;
                for (j = 0; j < particle->dim; j++)
                    particle->pbest[j] = particle->x[j];
            }
        }

        /* Identify best performing particle */
        g = pso_get_best_fitness(swarm);

        for (i = 0; i < swarm->num_particles; i++) {
            particle = &swarm->particle[i];
            particle->g = g;
        }

#ifdef SIMPLE_DEBUG
        /* Print best performing particle */
        fprintf(stderr, "\nIteration %d:\n", iter);
        pso_print_particle(&swarm->particle[g]);
#endif
        iter++;
    } /* End of iteration */

    return g;
}

int optimize_using_omp(char *function, int dim, int swarm_size, 
                       float xmin, float xmax, int num_iter, int num_threads)
{
    /* Initialize PSO */
    swarm_t *swarm;
    srand(time(NULL));
    swarm = pso_init(function, dim, swarm_size, xmin, xmax, num_threads);
    if (swarm == NULL) {
        fprintf(stderr, "Unable to initialize PSO\n");
        exit(EXIT_FAILURE);
    }

#ifdef VERBOSE_DEBUG
    pso_print_swarm(swarm);
#endif
    
    /* Solve PSO */
    int g;
    g = pso_solve_omp(function, swarm, xmax, xmin, num_iter, num_threads);
    if (g >= 0) {
        fprintf(stderr, "Solution:\n");
        pso_print_particle(&swarm->particle[g]);
    }
    
    pso_free(swarm);
    return g;
}
