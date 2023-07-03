import onnx
import os
import sys
import json
import shutil

from argparse import ArgumentParser
from memplan import create_plannerproto_file, read_pbfile

def get_ops(model):
    return list(set([node.op_type for node in model.graph.node]))

def get_ops_dtype(model_file):
    model = onnx.load(model_file)
    weights = {weight.name: weight.data_type for weight in model.graph.initializer}
    node_dict = {n.name: {"op": n.op_type, "dtype": ""}
        for n in model.graph.node}

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
            L.append((node_dict[name]["op"] + ",",
                     dtype_map[node_dict[name]["dtype"]]))
        else:
            L.append((node_dict[name]["op"] + ",", "NO_OP"))
    return list(set(L))


def get_opsets(model):

    # Get the model's graph
    graph = model.graph

    # Get the operators and their versions in the model
    operators = []
    for node in graph.node:
        operator_dict = {}
        operator_dict["op_type"] = node.op_type
        # operator_set_index = node.domain
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


def check_opset(model):
    with open('./tools/supported_opset.json') as json_file:
        supported_opset = json.load(json_file)

    opsets = get_opsets(model)
    ret = True    

    for opset in opsets:
        op = opset['op_type']
        ver = opset['since_version']

        if op in supported_opset:
            if ver in supported_opset[op]:
                # print(f'{op}-{ver} is supported')
                pass
        else:
            print(f'[PY] {op}-{ver} is NOT supported')
            ret = False
    
    return ret


def change_macros(model: onnx.ModelProto):

    if(os.path.exists('./src/uonnx.h.bak')):
        print("[PY] Header already modified!")
        return
    
    ops = get_ops(model)

    shutil.copy('./src/uonnx.h', './src/uonnx.h.bak')
    lines = []

    with open('./src/uonnx.h','r') as f:
        lines = f.readlines()
    
    for i in range(len(lines)):
        if "#define UONNX_OPS_" in lines[i]:
            to_set = "// "+lines[i]
            for op in ops:
                if f"#define UONNX_OPS_{op.upper()}" in lines[i]:
                    to_set = lines[i]
            lines[i] = to_set

    with open('./src/uonnx.h','w') as f:
        for line in lines:
            f.write(f"{line}")
        print("[PY] Header generated!")
    shutil.copy('./src/uonnx.h','./build/misc/uonnx.h')

def restore_macros():
    if(os.path.exists('./src/uonnx.h') and os.path.exists('./src/uonnx.h.bak')):
        os.remove('./src/uonnx.h')
        os.rename('./src/uonnx.h.bak', './src/uonnx.h')
        print('[PY] Header restored!')

def generate_misc(model_path):
    planner_target = './build/misc/' + os.path.basename(model_path).rstrip(".onnx") + "_planner.pb"
    create_plannerproto_file(model_path, planner_target)
    print("[PY] Planner generated!")
    model_header = os.path.splitext(os.path.basename(model_path))[0] + '_onnx'
    model_header_path = './build/misc/'+model_header

    with open(model_header_path, 'w') as f:
        f.write(f'#ifndef __{model_header.upper()}__\n')
        f.write(f'#define __{model_header.upper()}__\n')

    os.system(f'xxd -i {model_path} >> {model_header_path}')
    os.system(f'xxd -i {planner_target} >> {model_header_path}')

    with open(model_header_path, 'a') as f:
        f.write(f'#endif __{model_header.upper()}__\n')
    

def get_args():
    parser = ArgumentParser(description="uONNX Preprocessor")
    parser.add_argument('--model', type=str, default='./examples/benchmarks/cpu/mnist/mnist.onnx', help="ONNX model path")
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()

    return args

def main(args = None):
    if not args:
        return
    
    if os.path.basename(os.path.normpath(os.getcwd())) != 'uonnx':
        print("[PY] Move to ./uonnx directory.")
        return
    
    if args.restore:
        restore_macros()
        return
    
    if not os.path.exists(args.model):
        print("[PY] File does not exist")
        return
    else: 
        print("[PY] ONNX Model found at", args.model)

    os.system('mkdir -p ./build/misc')

    model = onnx.load(args.model)

    if check_opset(model):
        change_macros(model)
        generate_misc(args.model)
    else:
        print("[PY] Model contains unsupported opsets.")
        return



if __name__ == "__main__":
    main(get_args())


    