#include "uonnx_planner.h"

PlannerProto * load_planner(const char * filename)
{
    Planner__Planner * planner = NULL;
    FILE * fp;
	void * buf;
	size_t l, len;

    fp = fopen(filename, "rb");

    if(fp)
    {
        fseek(fp, 0L, SEEK_END);
		l = ftell(fp);
		fseek(fp, 0L, SEEK_SET);
		if(l > 0)
        {
            buf = malloc(l);
            if(buf)
            {
                for(len = 0; len < l; len += fread(buf + len, 1, l - len, fp));
                planner = planner__planner__unpack(NULL, len, buf);
                free(buf);
            }
        }
        fclose(fp);
    }

    return planner;
}

void free_plannerproto(PlannerProto * planner)
{
    planner__planner__free_unpacked(planner, NULL);
}

static inline Plan *plan_create(char *tensor_name, size_t idx)
{
    Plan *p = (Plan *)malloc(sizeof(Plan));

    if (p)
    {
        p->tensor_name = (char *)(NULL);
        p->index = (size_t)(NULL);

        if (tensor_name && idx)
        {
            p->tensor_name = strdup(tensor_name);
            p->index = idx;
        }

        return p;
    }

    return NULL;
}

static inline void free_plan(Plan *p)
{
    if (p)
    {
        if (p->tensor_name)
            free(p->tensor_name);
        free(p);
    }
}

Planner *planner_init(int n_plans, size_t max_bytes, int max_tensors)
{
    int i = 0;
    Planner *planner = (Planner *)malloc(sizeof(Planner));

    if (!planner)
    {
        return NULL;
    }

    planner->plans = (Plan **)malloc(sizeof(Plan *) * n_plans);
    if (!planner->plans)
    {
        free(planner);
        return NULL;
    }

    for (i = 0; i < n_plans; i++)
    {
        planner->plans[i] = plan_create((char *)NULL, (size_t)NULL);
    }

    planner->n_plans = n_plans;
    planner->max_arena_size = max_bytes;
    planner->max_arena_n_tensors = max_tensors;

    return planner;
}

void free_planner(Planner *planner)
{
    int i = 0;
    Plan *p;

    if (planner)
    {
        if (planner->plans)
        {
            for (i = 0; i < planner->n_plans; i++)
            {
                p = planner->plans[i];
                free_plan(p);
            }

            free(planner->plans);
        }
        free(planner);
        planner = NULL;
    }
}

void planner_add(char *tensor_name, size_t idx, Planner *planner)
{
    int i = 0;

    for (i = 0; i < planner->n_plans; i++)
    {
        if (planner->plans[i]->tensor_name == NULL)
        {
            printf("%s\n", tensor_name);
            planner->plans[i]->tensor_name = strdup(tensor_name);
            planner->plans[i]->index = idx;

            return;
        }
    }
}

size_t get_plan(char *tensor_name, Planner *planner)
{
    int i = 0;

    for (i = 0; i < planner->n_plans; i++)
    {
        if (strcmp(planner->plans[i]->tensor_name, tensor_name) == 0)
        {
            return planner->plans[i]->index;
        }
    }

    return 0;
}

static inline int isInitializer(char *name, GraphProto *gproto)
{
    int i = 0;
    for (i = 0; i < gproto->n_initializer; i++)
    {
        if (strcmp(gproto->initializer[i]->name, name) == 0)
        {
            return 1;
        }
    }
    return 0;
}

static inline size_t get_size_from_ValueInfoProto(ValueInfoProto *vp)
{
    size_t sz = 1;
    int i = 0;

    for (i = 0; i < vp->type->tensor_type->shape->n_dim; i++)
    {
        if (vp->type->tensor_type->shape->dim[i]->value_case == ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE)
        {
            sz *= vp->type->tensor_type->shape->dim[i]->dim_value;
        }
        else
        {
            sz *= 1;
        }
    }
    sz *= onnx_tensor_type_sizeof((TensorType)vp->type->tensor_type->elem_type);

    return sz;
}

static inline size_t get_size_from_name(char * tensor_name, GraphProto * gproto)
{
    int i = 0;

    for(i = 0; i < gproto->n_input; i++)
    {
        if(strcmp(gproto->input[i]->name, tensor_name)==0) 
            return get_size_from_ValueInfoProto(gproto->input[i]);
    }

    for(i = 0; i < gproto->n_output; i++)
    {
        if(strcmp(gproto->output[i]->name, tensor_name)==0) 
            return get_size_from_ValueInfoProto(gproto->output[i]);
    }

    for(i = 0; i < gproto->n_value_info; i++)
    {
        if(strcmp(gproto->value_info[i]->name, tensor_name)==0)
            return get_size_from_ValueInfoProto(gproto->value_info[i]);
    }

    return -1;
}

static inline void planner_sort(char ** seq, int n, GraphProto * gproto)
{
    int i = 0, j = 0;
    size_t bef, aft;
    char * temp;

    for (i = 0; i < n; i++)
        for (j = 0; j < n - 1; j++)
        {
            bef = get_size_from_name(seq[j], gproto);
            aft = get_size_from_name(seq[j+1], gproto);
            if(bef == -1 || aft == -1) return;

            if (bef < aft) 
            {
                temp = seq[j];
                seq[j] = seq[j+1];
                seq[j+1] = temp;
            }
        }
}

static inline int get_tensor_expiry(char * tname, GraphProto * gproto)
{
    int i = 0, j = 0;
    
    for(i = gproto->n_node - 1; i >= 0; i--)
    {
        for(j = 0; j < gproto->node[i]->n_input; j++)
        {
            if(strcmp(tname, gproto->node[i]->input[j])==0) return i;
        }
        
        for(j = 0; j < gproto->node[i]->n_output; j++)
        {
            if(strcmp(tname, gproto->node[i]->output[j])==0) return i;
        }
    }

    return -1;
}

typedef struct memblk 
{
    char * name;
    size_t begin;
    size_t end;
} MemBlk;

static inline size_t get_start_addr_from_memblk(size_t sz, MemBlk * memblks, int nblks)
{
    int i = 0;
    size_t start_addr = 0;

    for(i = 0; i < nblks; i++)
    {
        if(memblks[i].name != NULL)
        {
            if(start_addr + sz > memblks[i].begin && start_addr < memblks[i].end)
            {
                start_addr = memblks[i].end;
                i = 0;
            }
        }
    }        

    return start_addr;
}

static inline int get_free_idx_in_memblks(MemBlk * memblks, int nblks)
{
    int i = 0;

    for(i = 0; i < nblks; i++)
    {
        if(memblks[i].name == NULL) return i;
    }

    return -1;
}

static inline int inMemBlks(char * name, MemBlk * memblks, int nblks)
{
    int i = 0;
    
    for(i = 0; i < nblks; i++)
    {
        if(memblks[i].name != NULL)
            if(strcmp(memblks[i].name, name)==0) return 1;
    }
    
    
    return 0;
}

static inline void viewMemBlks(MemBlk * memblks, int nblks)
{
    int i = 0;
    
    for(i = 0; i < nblks; i++)
    {
        printf("    Memblk %s from %d to %d\n", memblks[i].name, memblks[i].begin, memblks[i].end);
    }
}

static inline int get_n_plans(GraphProto * gproto)
{
    int n_plans = 0, i = 0, j = 0;

    for(i = 0; i < gproto->n_node; i++)
    {
        for(j = 0; j < gproto->node[i]->n_input; j++)
        {
            if(!isInitializer(gproto->node[i]->input[j], gproto)) n_plans++;
        }
    }
    return n_plans + gproto->n_output;
}

Planner * planner_init_from_proto(GraphProto * gproto)
{
    // Initialize value
    int i = 0, j = 0;
    int n_plans = get_n_plans(gproto);
    char ** seq = malloc(sizeof(char*)*n_plans);
    int nseq = 0;

    MemBlk * memblks = malloc(sizeof(MemBlk)*n_plans);
    char * to_plan_names[n_plans];
    size_t to_plan_addridx[n_plans];

    int n_toplan = 0;
    int max_bytes = 0;

    Planner *planner;

    // Reinitialize all memblks
    for(i = 0; i < n_plans; i++)
    {
        memblks[i].name = NULL;
        memblks[i].begin = 0;
        memblks[i].end = 0;
    }
    size_t blk_size = 0;
    int blk_idx = 0;

    printf("n_plans %d\n", n_plans);

    /**SCRATCH*/

    // for(i = 0; i < gproto->n_node; i++)
    // {
    //     printf("NODE %d: %s\n", i, gproto->node[i]->name);
    //     for(j = 0; j < gproto->node[i]->n_input; j++)
    //     {
    //         if(isInitializer(gproto->node[i]->input[j], gproto)) printf("   [INIT] %s\n", gproto->node[i]->input[j]);
    //         else printf("   %s \n", gproto->node[i]->input[j]);
    //     }
    // }

    // printf("%s\n", gproto->value_info[0]->name);
    /**END SCRATCH*/

    printf("%d\n", gproto->n_node);

    for(i = 0; i < gproto->n_node; i++)
    {
        printf("Node %d\n", i);
        // Add all non-initializers not assigned with addresses (not in memblks)
        for(j = 0; j < gproto->node[i]->n_input; j++)
        {
            if(!isInitializer(gproto->node[i]->input[j], gproto) && !inMemBlks(gproto->node[i]->input[j], memblks, n_plans))
            {
                seq[nseq] = gproto->node[i]->input[j];
                nseq++;
            }
        }

        for(j = 0; j < gproto->node[i]->n_output; j++)
        {
            if(!isInitializer(gproto->node[i]->output[j], gproto) && !inMemBlks(gproto->node[i]->output[j], memblks, n_plans))
            {
                seq[nseq] = gproto->node[i]->output[j];
                nseq++;
            }
        }

        // Sort all in seq.
        planner_sort(seq, nseq, gproto);

        // Assign non-initializer tensors with addresses

        for(j = 0; j < nseq; j++)
        {
            blk_size = get_size_from_name(seq[j], gproto);
            blk_idx = get_free_idx_in_memblks(memblks, n_plans);

            memblks[blk_idx].begin = get_start_addr_from_memblk(blk_size, memblks, n_plans);
            memblks[blk_idx].end = memblks[blk_idx].begin + blk_size;
            memblks[blk_idx].name = seq[j];

            if(memblks[blk_idx].end > max_bytes) max_bytes = memblks[blk_idx].end;

            printf("%d: \"%s\"\n",n_toplan, memblks[blk_idx].name);
            to_plan_names[n_toplan] = memblks[blk_idx].name;
            to_plan_addridx[n_toplan] = memblks[blk_idx].begin;
            n_toplan++;
        }

        for(j = 0; j < n_plans; j++)
        {
            if(memblks[j].name != NULL)
            {
                if(get_tensor_expiry(memblks[j].name, gproto) <= i)
                {
                    // printf("Removing %s\n", memblks[j].name);
                    memblks[j].name = NULL;
                    memblks[j].begin = 0;
                    memblks[j].end = 0;
                }
            }     
        }
        nseq = 0;
    }

    planner = planner_init(n_plans, max_bytes + 8, n_plans + gproto->n_initializer);
    for(i = 0; i < n_plans; i++)
    {
        planner_add(to_plan_names[i], to_plan_addridx[i], planner);
    }

    // free(to_plan_addridx);
    // free(to_plan_names);
    free(memblks);
    free(seq);

    return planner;
}