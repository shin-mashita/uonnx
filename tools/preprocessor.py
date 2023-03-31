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

# Fix resolver

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

def get_args():
    parser = ArgumentParser(description="uONNX Preprocessor")
    parser.add_argument('--model', type=str, default='./scratch/model.onnx', help="ONNX model path")
    
    args = parser.parse_args()

    return args

def main(args=None):
    if args:
        model_file = args.model
        model = onnx.load(model_file)

        ops_dir = './src/ops/'
        ops_ext = '.c'

        ops = [ops_dir + op + ops_ext for op in get_ops(model)]
        opstr = ""

        # Check if ops_path exists
        for op in ops:
            if not os.path.exists(op):
                print(op + " does not exit.")
                return
            else:
                opstr = opstr + op + " "

        opstr = '\'' + opstr + '\''

        cwd = os.getcwd()
        if cwd[-5:] == 'uonnx':
            os.system('make run_with_lib OPS={0}'.format(opstr))


if __name__ == "__main__":
    main(get_args())