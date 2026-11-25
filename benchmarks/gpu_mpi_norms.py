"""Reintegrate verified MPI rank fields on an FP64 OpenCL GPU with an independent triangle rule."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def triangle_rule():
    # Duffy pullback of independent 8-point Gauss-Legendre quadrature on [0,1]^2.
    points, weights = np.polynomial.legendre.leggauss(8)
    points, weights = (points + 1) / 2, weights / 2
    return np.asarray([(x, (1 - x)*y, wx*wy*(1 - x)) for x, wx in zip(points, weights)
                       for y, wy in zip(points, weights)], dtype=np.float64)


def rank_data(directory, level, rank, inventory):
    def read(name, dtype):
        key = f"n{level}/{name}"
        data = (directory / key).read_bytes()
        if digest(data) != inventory.get(key):
            raise ValueError(f"rank input digest mismatch for {key}")
        return np.frombuffer(data, dtype).reshape(-1, 3)
    nodes = read(f"nodes.{rank}.f64", "<f8")
    cells = read(f"cells.{rank}.i32", "<i4")
    if not len(cells) or not np.isfinite(nodes).all() or cells.min() < 0 or cells.max() >= len(nodes):
        raise ValueError("invalid rank field or connectivity")
    return nodes, cells


def agrees(calculated, expected):
    return bool(np.isfinite(calculated).all() and np.allclose(calculated, expected, atol=0., rtol=1e-7))


def capture(directory, receipt, receipt_sha256):
    import pyopencl as cl
    data = receipt.read_bytes()
    if digest(data) != receipt_sha256:
        raise ValueError("MPI receipt digest mismatch")
    row = json.loads(data)
    if row["verdict"] != "ACCEPTED" or row["arm"] != "mpi4":
        raise ValueError("requires a verified four-rank compact MPI receipt")
    levels = row["levels"]
    if (not levels or any(level["case"] != "sin_sin" or level["ranks"] != 4 or level["order"] != 1
                          or level["global_elements"] != 2*level["n"]**2 for level in levels)
            or any(b["n"] != 2*a["n"] for a, b in zip(levels, levels[1:]))):
        raise ValueError("incompatible MPI case or mesh sequence")
    devices = [dev for platform in cl.get_platforms() for dev in platform.get_devices()
               if dev.type & cl.device_type.GPU and dev.double_fp_config]
    if not devices:
        raise RuntimeError("requires an OpenCL GPU with double precision")
    dev = devices[0]
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    source = Path(__file__).with_name("mpi_field_norms.cl").read_bytes()
    kernel = cl.Program(ctx, source.decode()).build().field_norms
    rule = triangle_rule()
    def upload(array):
        return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=array)
    quadrature = upload(rule)
    def integrate(nodes, cells):
        out = np.empty((len(cells), 2), dtype=np.float64)
        destination = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out.nbytes)
        event = kernel(queue, (len(cells),), None, upload(nodes), upload(cells),
                       quadrature, np.int32(len(rule)), destination)
        cl.enqueue_copy(queue, out, destination).wait()
        if not np.isfinite(out).all() or (out < 0).any():
            raise ValueError("invalid GPU squared norm")
        return out.sum(axis=0), (event.profile.end - event.profile.start)*1e-9
    levels = []
    for level in row["levels"]:
        start = time.perf_counter()
        n = level["n"]
        squared = np.zeros(2)
        kernel_time = 0.
        damaged = np.zeros(2)
        count = 0
        for rank in range(4):
            nodes, cells = rank_data(directory, n, rank, row["files"])
            value, seconds = integrate(nodes, cells)
            kernel_time += seconds
            squared += value
            count += len(cells)
            if rank == 0:
                altered = nodes.copy()
                altered[:, 2] *= 0.99
                damaged += integrate(altered, cells)[0]
            else:
                damaged += value
        norms = np.sqrt(squared)
        expected = [level["l2_error"], level["h1_semi_error"]]
        if count != level["global_elements"] or not agrees(norms, expected) or agrees(np.sqrt(damaged), expected):
            raise ValueError(f"GPU audit or negative control failed on n={n}")
        levels.append({"n": n, "triangles": count, "gpu_norms": norms.tolist(), "cpu_receipt_norms": expected,
                       "wrong_field_control_norms": np.sqrt(damaged).tolist(), "wrong_field_rejected": True,
                       "kernel_seconds": kernel_time, "audit_seconds_including_control": time.perf_counter()-start})
    return {"device": dev.name, "driver": dev.driver_version, "opencl": dev.version, "fp64_config": dev.double_fp_config,
            "numpy": np.__version__, "pyopencl": cl.VERSION_TEXT, "kernel_sha256": digest(source),
            "mpi_receipt_sha256": receipt_sha256, "relative_tolerance": 1e-7, "levels": levels}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in ("directory", "receipt", "output"):
        parser.add_argument("--"+opt, type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    report = capture(args.directory, args.receipt, args.receipt_sha256)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
