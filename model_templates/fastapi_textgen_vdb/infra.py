import pulumi_datarobot as dr
import datarobot as dr_api
from datarobot.models.genai.vector_database import CustomModelVectorDatabaseValidation
from pulumi import Output


# Look up the execution environtmnet
all_envs = dr_api.ExecutionEnvironment.list()
use_case = dr.UseCase("FastAPI Sample Text Generation VDB")
execution_environment = next(
    (env for env in all_envs if env.name == "[DataRobot] Python 3.11 Drop-In"), None
)

prediction_environment = dr.PredictionEnvironment(
    "FastAPI Text Generation VDB", platform="datarobotServerless"
)

custom_model = dr.CustomModel(
    resource_name="fastapi-textgen-vdb-cm",
    base_environment_id=execution_environment.id,
    description="FastAPI Text Generation Model with Vector DB",
    language="python",
    folder_path=".",
    # For Unstructured models, the model type must be set to "unstructured"
    # This model also works with TextGen models uncomment the two lines below
    target_type="Unstructured",
    # target_name="relevant",
    # target_type="TextGeneration",
    use_case_ids=[use_case.id],
    resource_bundle_id="cpu.3xlarge",  # need boosted memory for ONNX model
)

registered_model = dr.RegisteredModel(
    "FastAPI Text Generation VDB",
    custom_model_version_id=custom_model.version_id,
    name="fastapi-textgen-vdb-rm",
    use_case_ids=[use_case.id],
)

deployment = dr.Deployment(
    "FastAPI Text Generation VDB",
    label="FastAPI Text Generation VDB",
    prediction_environment_id=prediction_environment.id,
    registered_model_version_id=registered_model.version_id,
    use_case_ids=[use_case.id],
    predictions_data_collection_settings={"enabled": True},
    # description="FastAPI Text Generation Model with Vector DB",
)


def register_vector_db(deployment_id: str, use_case_id: str):
    external_vdb_validation = CustomModelVectorDatabaseValidation.create(
        prompt_column_name="question",
        target_column_name="relevant",
        deployment_id=deployment_id,
        use_case=use_case_id,
        wait_for_completion=True,
    )
    print(external_vdb_validation.__dict__)


Output.all(deployment.id, use_case.id).apply(lambda args: register_vector_db(args[0], args[1]))


