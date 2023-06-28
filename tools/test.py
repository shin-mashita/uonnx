from planner.proto3_pb2 import Planner
import onnx
import numpy as np

test_planner = {
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

def __shash(s: str)->int:
    mod = 5381
    mask = 0xFFFFFFFF
    for c in s:
        mod = (mod << 5) + mod + ord(c)

    mod = mod & mask

    print(hex(mod))

    return mod

def __check_unique_shashes(planner: Planner, model: onnx.ModelProto) -> bool:
    mset  = set()
    onnx
    for plan in planner.plans:
        mset.add(plan.id)
    
    for initializer in model.graph.initializer:
        mset.add(__shash(initializer.name))
    
    if(len(mset) == planner.arena.max_ntensors):
        return 1
    return 0


def __tensortype_sizeof(type):
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

def __parse_tensor_data_from_value_info(info) -> dict:
    plan = dict()
    plan["start_idx"] = -1
    plan["type"] = info.type.tensor_type.elem_type
    plan["dims"] = [max(d.dim_value, 1)
                    for d in info.type.tensor_type.shape.dim]
    plan["ndata"] = np.prod(plan["dims"])
    plan["size"] = __tensortype_sizeof(
        info.type.tensor_type.elem_type)*plan["ndata"]

    return plan

def __get_tensor_death(tensor, model):
    for node_idx in range(len(model.graph.node)-1, -1, -1):
        node = model.graph.node[node_idx]
        for input in node.input:
            if input == tensor:
                return node_idx
        for output in node.output:
            if output == tensor:
                return node_idx
    
    return -1

def __in_blocks(tensor, blocks):
    for name, _ , _ in blocks:
        if tensor == name:
            return 1
    return 0

def __get_first_available_pos(tensor_size, blocks):
    blocks.sort(key=lambda x:x[1])
    pos = 0
    for _, start, size in blocks:
        if pos + tensor_size > start and pos < start + size:
            pos = start + size
    return pos


def create_plannerproto(model_file:str) -> Planner:
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
            plans[i.name] = __parse_tensor_data_from_value_info(i)

    for i in model.graph.value_info:
        if i.name not in initializers:
            plans[i.name] = __parse_tensor_data_from_value_info(i)

    for i in model.graph.output:
        if i.name not in initializers:
            plans[i.name] = __parse_tensor_data_from_value_info(i)

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
            if not __in_blocks(name, blocks):
                plans[name]["start_idx"] = __get_first_available_pos(size, blocks)
                blocks.append((name, plans[name]["start_idx"], size))

        for name, start, size in blocks:
            if node_idx >= __get_tensor_death(name, model):
                blocks.remove((name, start, size))

    planner["arena"] = dict()
    planner["arena"]["max_ntensors"] = len(plans.keys()) + len(model.graph.initializer)
    planner["arena"]["max_bytes"]  = max([val['start_idx'] + val['size'] for val in plans.values()])

    planner["plans"] = plans

    proto = Planner()

    proto.arena.max_ntensors = planner["arena"]["max_ntensors"]
    proto.arena.max_bytes = planner["arena"]["max_bytes"]

    for name, plan_dict in planner["plans"].items():
        planproto = proto.plans.add()
        planproto.id = __shash(name)
        planproto.start_idx = plan_dict["start_idx"]
        planproto.type = plan_dict["type"]
        planproto.dims.extend(plan_dict["dims"])

    if(__check_unique_shashes(proto, model)):
        return proto
    else:
        print("Collision occurred between tensor names hashes. Use another modulus.")


def create_plannerproto_file(model_file: str, target:str=""):
    planner = create_plannerproto(model_file)

    if target == "":
        target = model_file.rstrip(".onnx") + "_planner.pb"

    with open(target, "wb") as f:
        f.write(planner.SerializeToString())


def read_pbfile(filename: str):
    planner = Planner()

    with open(filename, "rb") as f:
        planner.ParseFromString(f.read())

    print("Arena profile:")
    print("\tMax tensors:", planner.arena.max_ntensors)
    print("\tMax bytes", planner.arena.max_bytes)

    print(f"Plans:")
    for plan in planner.plans:
        mdims = []
        print("\tPlan:", hex(plan.id))
        print("\t\tstart_idx:", plan.start_idx)
        print("\t\ttype:", plan.type)
        for dim in plan.dims:
            mdims.append(dim)
        print("\t\tdims:", mdims)
    

# TODO PLANNER Viz

if __name__ == "__main__":
    create_plannerproto_file("reference.onnx")
    read_pbfile("reference_planner.pb")


# MNIST: 505 B -> 248 B
# KWS: 3.1 KB -> 445 B
# VWW: 7.5 KB -> 1.1 KB
