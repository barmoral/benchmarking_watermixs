import os
import pickle
import pathlib
import argparse
from openff.units import unit
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.backends import ComputeResources, QueueWorkerResources
from openff.evaluator.backends.dask import DaskSLURMBackend
from openff.evaluator.storage.localfile import LocalFileStorage
from openff.evaluator.client import EvaluatorClient, RequestOptions, ConnectionOptions
from openff.evaluator.server.server import EvaluatorServer
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.toolkit import ForceField

# ------------------ ARGUMENT PARSING ------------------
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-path", default="training-properties-with-water.json")
parser.add_argument("-s", "--storage-directory", required=True)
parser.add_argument("-ff", "--forcefield", required=True)
parser.add_argument("-wff", "--waterforcefield", required=True)
parser.add_argument("-o", "--output-directory", default="output")
parser.add_argument("-r", "--replicate", type=int, default=1)
parser.add_argument("-p", "--port", type=int, default=8100)
parser.add_argument("-of", "--options-file", default="request-options.json")
parser.add_argument("--worker-id", type=int, required=True)
parser.add_argument("--num-workers", type=int, default=3)
args = parser.parse_args()

# ------------------ GPU SETUP ------------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.worker_id)
print(f"[INFO] Set CUDA_VISIBLE_DEVICES={args.worker_id}")

from openmm import Platform
print('PRINT PLATFORM HERE:')
for i in range(Platform.getNumPlatforms()):
    print(Platform.getPlatform(i).getName())
print('PRINT DEFAULTPLATFORM HERE:')
platform = Platform.getPlatformByName("CUDA")
print("Default platform:", platform.getName())

# ------------------ DATA LOADING ------------------
dataset = PhysicalPropertyDataSet.from_json(args.input_path)
all_props = dataset.properties

# Partition properties for this worker
chunk_size = (len(all_props) + args.num_workers - 1) // args.num_workers
start = args.worker_id * chunk_size
end = min((args.worker_id + 1) * chunk_size, len(all_props))

worker_dataset = PhysicalPropertyDataSet()
for prop in all_props[start:end]:
    worker_dataset.add_properties(prop)

print(f"[Worker {args.worker_id}] Processing {len(worker_dataset.properties)} properties")

# Set uncertainty manually if needed
for prop in worker_dataset.properties:
    prop.uncertainty = 0.001 * prop.value.units

# ------------------ FORCE FIELD ------------------
# Extract water model name (without extension)
water_model_name = pathlib.Path(args.waterforcefield).stem

# Extract version from force field filename (e.g., "openff-2.2.1.offxml" → "2.2.1")
ff_version = pathlib.Path(args.forcefield).stem.split("-")[-1]

# Combine to make directory name
combined_dir_name = f"{water_model_name}_{ff_version}"

# Create directory
force_field_dir = pathlib.Path(combined_dir_name)
force_field_dir.mkdir(parents=True, exist_ok=True)

# Save combined force field JSON inside that directory
force_field = ForceField(args.forcefield, args.waterforcefield)
force_field_json_path = force_field_dir / "force-field.json"
with open(force_field_json_path, "w") as file:
    file.write(SmirnoffForceFieldSource.from_object(force_field).json())

force_field_source = SmirnoffForceFieldSource.from_json(str(force_field_json_path))


# ------------------ OPTIONS AND STORAGE ------------------
options = RequestOptions.from_json(args.options_file)
storage_directory = LocalFileStorage(
    root_directory=str(pathlib.Path(args.storage_directory).resolve()),
    cache_objects_in_memory=True,
)

# ------------------ OUTPUT SETUP ------------------
output_directory = pathlib.Path(force_field_dir) / args.output_directory / f"rep-{args.replicate}"
if not output_directory.exists():
    output_directory.mkdir(parents=True, exist_ok=True)
os.chdir(output_directory)

pickle_file = f"results-worker{args.worker_id}.pkl"
results_file = f"results-worker{args.worker_id}.json"
if pathlib.Path(results_file).exists():
    print("Results already exist, exiting early.")
    exit()

# ------------------ BACKEND SETUP ------------------
print(f"Starting server on port {args.port}")
worker_resources = QueueWorkerResources(
    number_of_threads=1,
    number_of_gpus=1,
    preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA,
    per_thread_memory_limit=4 * unit.gigabyte,
    wallclock_time_limit="12:00:00",
)

with DaskSLURMBackend(
    minimum_number_of_workers=1,
    maximum_number_of_workers=1,
    resources_per_worker=worker_resources,
    queue_name="blanca-shirts",
    setup_script_commands=[
        "source ~/.bashrc",
        "conda activate evaluator-050-118",
        "conda env export > conda-env.yaml",
    ],
    extra_script_options=[f"--gres=gpu:1", f"--export=CUDA_VISIBLE_DEVICES={args.worker_id}"],
    adaptive_interval="1000ms",
) as calculation_backend:

    server = EvaluatorServer(
        calculation_backend=calculation_backend,
        working_directory="working-directory",
        delete_working_files=False,
        enable_data_caching=False,
        storage_backend=storage_directory,
        port=args.port,
    )
    with server:
        client = EvaluatorClient(ConnectionOptions(server_port=args.port))
        request, error = client.request_estimate(worker_dataset, force_field_source, options)
        assert error is None
        results, exception = request.results(synchronous=True, polling_interval=30)

assert exception is None

print("Simulation complete")
print(f"# estimated: {len(results.estimated_properties)}")
print(f"# unsuccessful: {len(results.unsuccessful_properties)}")
print(f"# exceptions: {len(results.exceptions)}")

with open(pickle_file, "wb") as f:
    pickle.dump(results, f)
print(f"Results dumped to {pickle_file}")

results.json(results_file, format=True)
print(f"Results saved to {results_file}")
