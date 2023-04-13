import onnx
import os
from argparse import ArgumentParser

def get_ops(model):
    return list(set([node.op_type for node in model.graph.node]))

def get_ops_dtype(model_file):
    model = onnx.load(model_file)
    weights = {weight.name: weight.data_type for weight in model.graph.initializer}
    node_dict = {n.name: {"op":n.op_type,"dtype":""} for n in model.graph.node}

    for n in model.graph.node:
        for _input in n.input:
            if _input in weights:
                node_dict[n.name]["dtype"] = weights[_input]
                break

    dtype_map = {
        0: "UNDEFINED",
        1: "FLOAT",
        2: "UINT8",
        3: "INT8",
        4: "UINT16",
        5: "INT16",
        6: "INT32",
        7: "INT64",
        8: "STRING",
        9: "BOOL",
        10: "FLOAT16",
        11: "DOUBLE",
        12: "UINT32",
        13: "UINT64",
        14: "COMPLEX64",
        15: "COMPLEX128",
        16: "BFLOAT16",
    }

    L = []
    for name in node_dict:
        if node_dict[name]["dtype"] in dtype_map:
            L.append((node_dict[name]["op"] + "," , dtype_map[node_dict[name]["dtype"]]))
        else:
            L.append((node_dict[name]["op"] + "," , "NO_OP"))
    return list(set(L))

def get_opset(model):

    # Get the model's graph
    graph = model.graph

    # Get the operators and their versions in the model
    operators = []
    for node in graph.node:
        operator_dict = {}
        operator_dict["op_type"] = node.op_type
        #operator_set_index = node.domain
        operator_set = model.opset_import[0]
        operator_dict["op_version"] = operator_set.version
        schema = onnx.defs.get_schema(node.op_type, operator_set.version)
        operator_dict["since_version"] = schema.since_version
        operators.append(operator_dict)

    operators_nodupe = []

    for x in operators:
        if x not in operators_nodupe:
            operators_nodupe.append(x)

    return operators_nodupe

def generate_resolver(ops):
    hbody1 = \
"""
#ifndef __ONNX_CUSTOM_RESOLVER_H__
#define __ONNX_CUSTOM_RESOLVER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "onnx_config.h"
#include "onnx_dtypes.h"

void * resolver_custom_create(void);
void resolver_custom_destroy(void * rctx);

"""
    hbody2 = '\n'.join([f"void resolver_default_op_{op}(struct onnx_node_t * n);" for op in ops])

    hbody3 = \
"""

extern struct onnx_resolver_t resolver_custom;

#ifdef __cplusplus
}
#endif

#endif
"""
    cbody1 = \
"""
#include "onnx_custom_resolver.h"

// Generated resolver
void * resolver_custom_create(void)
{
	return NULL;
}

void resolver_custom_destroy(void * rctx)
{
}

struct onnx_resolver_t resolver_custom = {
	.name 							= "custom",

	.create							= resolver_custom_create,
	.destroy						= resolver_custom_destroy,

        """
    
    cbody2 = '\n\t'.join([f".op_{op} = resolver_default_op_{op}," for op in ops])

    cbody3 = "\n};"

    return hbody1 + hbody2 + hbody3, cbody1 + cbody2 + cbody3

def generate_resolver_h(ops):
    hbody1 = """
#ifndef __ONNX_RESOLVER_H__
#define __ONNX_RESOLVER_H__

#ifdef __cplusplus
extern \"C\" {
#endif

#include "onnx_config.h"
#include "onnx_dtypes.h"

struct onnx_resolver_t {
	const char * name;

	void * (*create)(void);
	void (*destroy)(void * rctx);

    """

    hbody2 = '\n\t'.join([f"void (*op_{op})(struct onnx_node_t * n);" for op in ops])

    hbody3 = """
    };

void resolver_solve_operator(struct onnx_resolver_t * r, struct onnx_node_t * n);

void * resolver_default_create(void);
void resolver_default_destroy(void * rctx);

    """
    hbody4 = '\n'.join([f"void resolver_default_op_{op}(struct onnx_node_t * n);" for op in ops])

    hbody5 = """

extern struct onnx_resolver_t resolver_default;

#ifdef __cplusplus
}
#endif
#endif"""

    return hbody1 + hbody2 + hbody3 + hbody4 + hbody5

def add_resolver_h(ops):
    cwd = os.getcwd()
    if cwd[-5:] == 'uonnx':
        os.rename('./src/onnx_resolver.h','./src/onnx_resolver.h.bak')
        with open('./src/onnx_resolver.h', 'w') as f:
            f.write(generate_resolver_h(ops))

def restore_resolver_h():
    cwd = os.getcwd()
    if cwd[-5:] == 'uonnx':
        if (os.path.exists('./src/onnx_resolver.h') and os.path.exists('./src/onnx_resolver.h.bak')):
            os.remove('./src/onnx_resolver.h')
            os.rename('./src/onnx_resolver.h.bak', './src/onnx_resolver.h')
        

def get_args():
    parser = ArgumentParser(description="uONNX Preprocessor")
    parser.add_argument('--model', type=str, default='./scratch/model.onnx', help="ONNX model path")
    
    args = parser.parse_args()

    return args

def main(args=None):
    if args:
        model_file = args.model
        model = onnx.load(model_file)

        ops = get_ops(model)

        ops_dir = './src/ops/'
        ops_ext = '.c'

        ops_paths = [ops_dir + op + ops_ext for op in ops]
        opstr = ""

        # Check if ops_paths exists
        for op in ops_paths:
            if not os.path.exists(op):
                print(op + " does not exist.")
                return
            else:
                opstr = opstr + op + " "

        opstr = '\'' + opstr + '\''
        
        cwd = os.getcwd()
        if cwd[-5:] == 'uonnx':
            # add_resolver_h(ops)
            # os.system('make make_test OPS={0}'.format(opstr))
            # os.system('cat ./src/onnx_resolver.h')
            # restore_resolver_h()
            h,c = generate_resolver(ops)
            print(h)
            print(c)
            # os.system('make run_with_lib OPS={0}'.format(opstr))


if __name__ == "__main__":
    main(get_args())