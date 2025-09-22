import click

from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.client import RequestOptions


from openff.evaluator.client import RequestOptions, ConnectionOptions
from openff.evaluator.layers.equilibration import EquilibrationProperty
from openff.evaluator.utils.observables import ObservableType

def modify_workflow_schema(original_schema):
    """
    Modify a schema to run faster for testing.
    """
    workflow_schema = original_schema.workflow_schema
    new_workflow_schema = []
    for schema in workflow_schema.protocol_schemas:
        if schema.id == "conditional_group":
            protocol = schema.to_protocol()
            simulation_protocol = protocol.protocols["production_simulation"]
            simulation_protocol.output_frequency = 1000
            schema = protocol.schema

        if schema.id == "conditional_group_mixture":
            protocol = schema.to_protocol()
            simulation_protocol = protocol.protocols["production_simulation_mixture"]
            simulation_protocol.output_frequency = 1000
            schema = protocol.schema

        if schema.id == "conditional_group_component_$(component_replicator)":
            protocol = schema.to_protocol()
            simulation_protocol = protocol.protocols["production_simulation_component_$(component_replicator)"]
            simulation_protocol.output_frequency = 1000
            schema = protocol.schema
        
        new_workflow_schema.append(schema)

    workflow_schema.protocol_schemas = new_workflow_schema
    workflow_schema.replace_protocol_types(
        {"BaseBuildSystem": "BuildSmirnoffSystem"},
    )
    original_schema.workflow_schema = workflow_schema
    return original_schema


@click.command()
@click.option(
    "--n-molecules",
    "-n",
    default=1000,
    help="Number of molecules in the simulation.",
)
@click.option(
    "--output-file",
    "-o",
    default="request-options.json",
    help="Output file for the request options.",
)
def main(
    n_molecules: int = 1000,
    output_file: str = "request-options.json"
):

    potential_energy = EquilibrationProperty()
    potential_energy.observable_type = ObservableType.PotentialEnergy
    # require at least 50 uncorrelated samples in equilibration
    # before moving to production simulation
    # note, by default equilibration is set to output 20 frames per 200 ps
    # so this is at least 600 ps equilibration
    # turn this down to equilibrate less
    potential_energy.n_uncorrelated_samples = 50

    density = EquilibrationProperty()
    density.observable_type = ObservableType.Density
    density.n_uncorrelated_samples = 50

    dhmix_schema = EnthalpyOfMixing.default_preequilibrated_simulation_schema(
        n_molecules=n_molecules,
        equilibration_error_tolerances=[potential_energy, density],
        # note, production simulation outputs 500 frames per 2 ns
        n_uncorrelated_samples=50, # at least 200 samples to calculate observable
        equilibration_max_iterations=5, # max 1 ns equilibration
        max_iterations=4, # max 4 ns production
    )
    density_schema = Density.default_preequilibrated_simulation_schema(
        n_molecules=n_molecules,
        equilibration_error_tolerances=[potential_energy, density],
        n_uncorrelated_samples=50, # at least 200 samples to calculate observable
        equilibration_max_iterations=5, # max 1 ns equilibration
        max_iterations=4, # max 4 ns production
    )
    hmix_schema=modify_workflow_schema(dhmix_schema)
    dens_schema=modify_workflow_schema(density_schema)
    options = RequestOptions()
    options.calculation_layers = ["PreequilibratedSimulationLayer"]
    options.add_schema("PreequilibratedSimulationLayer", "Density", dens_schema)
    options.add_schema("PreequilibratedSimulationLayer", "EnthalpyOfMixing", hmix_schema)
    options.json(file_path=output_file, format=True)
    print(f"Request options saved to {output_file}")


if __name__ == "__main__":
    main()
