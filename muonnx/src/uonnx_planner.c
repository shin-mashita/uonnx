#include "uonnx_planner.h"

Plan * plan_create(char * tensor_name, size_t idx)
{
    Plan * p = (Plan *)malloc(sizeof(Plan));

    if(p)
    {
        p->tensor_name = (char *)(NULL);
        p->index = (size_t)(NULL);

        if(tensor_name && idx)
        {
            p->tensor_name = strdup(tensor_name);
            p->index = idx;
        }

        return p;
    }

    return NULL;
}

void free_plan(Plan * p)
{
    if(p)
    {
        if(p->tensor_name) free(p->tensor_name);
        free(p);
    }
}

Planner * planner_init(int n_plans, size_t max_bytes, int max_tensors)
{
    int i = 0;
    Planner * planner = (Planner *)malloc(sizeof(Planner));

    if(!planner)
    {
        return NULL;
    }

    planner->plans = (Plan **)malloc(sizeof(Plan *)*n_plans);
    if(!planner->plans)
    {
        free(planner);
        return NULL;
    }

    for(i = 0; i < n_plans; i++)
    {
        planner->plans[i] = plan_create((char *)NULL, (size_t)NULL);
    }

    planner->n_plans = n_plans;
    planner->max_arena_size = max_bytes;
    planner->max_arena_n_tensors = max_tensors;

    return planner;
}

void free_planner(Planner * planner)
{
    int i = 0;
    Plan * p;

    if(planner)
    {
        if(planner->plans)
        {
            for(i = 0; i < planner->n_plans; i++)
            {
                p = planner->plans[i];
                free_plan(p);
            }
            
            free(planner->plans);
        }

        free(planner);
    }
}

void planner_add(char * tensor_name, size_t idx,Planner * planner)
{
    int i = 0;

    for(i = 0; i < planner->n_plans; i++)
    {
        if(planner->plans[i]->tensor_name == NULL)
        {
            planner->plans[i]->tensor_name = strdup(tensor_name);
            planner->plans[i]->index = idx;

            return;
        }
    }
}

size_t get_plan(char * tensor_name, Planner * planner)
{
    int i = 0;

    for(i = 0; i < planner->n_plans; i++)
    {
        if(strcmp(planner->plans[i]->tensor_name, tensor_name)==0)
        {
            return planner->plans[i]->index;
        }
    }

    return 0;
}