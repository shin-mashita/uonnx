from planner.proto3_pb2 import Planner
import onnx
import numpy as np

# todo onnx proto parser
# todo algo for FFD given onnx model proto

test_dict = {
    "arena": {
        "max_ntensors": 1,
        "max_bytes": 2,
    },

    "plans": {
        "name1": {
            "start_idx": 1,
            "size": 1000,
            "type": 3,
            "ndata": 2,
            "dims": [1, 2, 3, 4],
        },
    },
}


def tensortype_sizeof(type):
    dtype_map = {
        0: 0,
        1: 4,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 4,
        7: 8,
        8: 8,
        9: 4,
        10: 2,
        11: 8,
        12: 4,
        13: 8,
        14: 8,
        15: 16,
        16: 2,
    }

    return dtype_map[type]


def parse_tensor_data_from_value_info(info) -> dict:
    plan = dict()
    plan["start_idx"] = -1
    plan["type"] = info.type.tensor_type.elem_type
    plan["dims"] = [max(d.dim_value, 1)
                    for d in info.type.tensor_type.shape.dim]
    plan["ndata"] = np.prod(plan["dims"])
    plan["size"] = tensortype_sizeof(
        info.type.tensor_type.elem_type)*plan["ndata"]

    return plan

def get_tensor_death(tensor, model):
    for node_idx in range(len(model.graph.node)-1, -1, -1):
        node = model.graph.node[node_idx]
        for input in node.input:
            if input == tensor:
                return node_idx
        for output in node.output:
            if output == tensor:
                return node_idx
    
    return -1

def in_blocks(tensor, blocks):
    for name, _ , _ in blocks:
        if tensor == name:
            return 1
    return 0

def get_first_available_pos(tensor_size, blocks):
    blocks.sort(key=lambda x:x[1])
    pos = 0
    for _, start, size in blocks:
        if pos + tensor_size > start and pos < start + size:
            pos = start + size
    return pos

def create_plannerdict(model_file) -> dict:
    planner = dict()

    model = onnx.load(model_file)

    v_count = 0
    for v in model.graph.value_info:
        v_count = v_count + 1

    if v_count == 0:
        model = onnx.shape_inference.infer_shapes(model)
        # with open("test.onnx", "wb") as f:
        #     onnx.save(model, f)

    plans = dict()

    initializers = [w.name for w in model.graph.initializer]

    for i in model.graph.input:
        if i.name not in initializers:
            plans[i.name] = parse_tensor_data_from_value_info(i)

    for i in model.graph.value_info:
        if i.name not in initializers:
            plans[i.name] = parse_tensor_data_from_value_info(i)

    for i in model.graph.output:
        if i.name not in initializers:
            plans[i.name] = parse_tensor_data_from_value_info(i)

    blocks = [] # [(name, start_idx, size)]

    for node_idx in range(len(model.graph.node)):
        # Get tensors to allocate
        node = model.graph.node[node_idx]
        to_allocs = list()
        for input in node.input:
            if input not in initializers and plans[input]["start_idx"] == -1:
                to_allocs.append(input)
        for output in node.output:
            if output not in initializers and plans[output]["start_idx"] == -1:
                to_allocs.append(output)

        to_allocs = [(name, plans[name]["size"]) for name in to_allocs]
        to_allocs.sort(key=lambda x:x[1], reverse=1)

        for name, size in to_allocs:
            if not in_blocks(name, blocks):
                plans[name]["start_idx"] = get_first_available_pos(size, blocks)
                blocks.append((name, plans[name]["start_idx"], size))

        for name, start, size in blocks:
            if node_idx >= get_tensor_death(name, model):
                blocks.remove((name, start, size))

    planner["arena"] = dict()
    planner["arena"]["max_ntensors"] = len(plans.keys())
    planner["arena"]["max_bytes"]  = max([val['start_idx'] + val['size'] for val in plans.values()])

    planner["plans"] = plans
    return planner

def planner_dict_to_proto(planner_dict):
    planner = Planner()

    planner.arena.max_ntensors = planner_dict["arena"]["max_ntensors"]
    planner.arena.max_bytes = planner_dict["arena"]["max_bytes"]

    for name, plan_dict in planner_dict["plans"].items():
        plan = planner.plans.add()
        plan.name = name
        plan.start_idx = plan_dict["start_idx"]
        plan.size = plan_dict["size"]
        plan.type = plan_dict["type"]
        plan.ndata = plan_dict["ndata"]
        plan.dims.extend(plan_dict["dims"])

    return planner

def create_plannerproto(model_file, target=""):
    planner = planner_dict_to_proto(create_plannerdict(model_file))

    if target == "":
        target = model_file.rstrip(".onnx") + "_planner.pb"

    with open(target, "wb") as f:
        f.write(planner.SerializeToString())

def read_pbfile(filename):
    planner = Planner()

    with open(filename, "rb") as f:
        planner.ParseFromString(f.read())

    print("Arena profile:")
    print("\tMax tensors:", planner.arena.max_ntensors)
    print("\tMax bytes", planner.arena.max_bytes)

    print(f"Plans:")
    for plan in planner.plans:
        mdims = []
        print("\tPlan:", plan.name[0:min(len(plan.name), 20)])
        print("\t\tstart_idx:", plan.start_idx)
        print("\t\tsize:", plan.size)
        print("\t\tndata:", plan.ndata)
        for dim in plan.dims:
            mdims.append(dim)
        print("\t\tdims:", mdims)


# TODO PLANNER Viz

if __name__ == "__main__":
    create_plannerproto("model.onnx")
    read_pbfile("model_planner.pb")
