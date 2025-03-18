from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, LongType

def clean_resource_references(resource: DataFrame, columns: list ) -> DataFrame:
    """
    Cleans the references in the given DataFrame by removing the prefix before the resource id
    i.e. "Organization/1" -> "1"
    :param resource: The DataFrame to clean
    :param columns: The columns to clean
    :return:
    """
    for column in columns:
        # check if column exists in the dataframe
        if column in resource.columns:
            resource = resource.withColumn(
                column, F.regexp_replace(F.col(column), ".*/", "")
            )
    return resource

def get_procedure(base: DataFrame, views: dict) -> DataFrame:
    procedure = views["Procedure"]

    return base.join(procedure, on="encounter_id").select(
        base.patient_id,
        base.encounter_id,
        base.visit_provider_id,
        
        procedure.event_time.cast(DateType()),
        procedure.code,
        procedure.system,
        F.lit("procedure").alias("type"),
        procedure.description_name,
        base.visitType,
        base.visit_type_code,
        base.account_id,
        F.lit(None).alias("value_date_time").cast(DateType()),
        F.lit(None).alias("value_string"),
    )
        
def get_diagnosis(base: DataFrame, views: dict) -> DataFrame:
    condition = views["Condition"].where(
        views["Condition"].category == "encounter-diagnosis"
    )

    return base.join(condition, on="encounter_id").select(
        base.patient_id,
        base.encounter_id,
        base.visit_provider_id,
        condition.event_time.cast(DateType()),
        condition.code,
        condition.system,
        F.lit("condition").alias("type"),
        condition.description_name,
        base.visitType,
        base.visit_type_code,
        base.account_id,
        F.lit(None).alias("value_date_time").cast(DateType()),
        F.lit(None).alias("value_string"),
    )
    
def get_medication(base: DataFrame, views: dict) -> DataFrame:
    medication = views["MedicationDispense"]

    return base.join(medication, on="encounter_id").select(
        base.patient_id,
        base.encounter_id,
        base.visit_provider_id,
        medication.event_time.cast(DateType()),
        medication.code,
        medication.system,
        F.lit("medication").alias("type"),
        medication.description_name,
        base.visitType,
        base.visit_type_code,
        base.account_id,
        F.lit(None).alias("value_date_time").cast(DateType()),
        F.lit(None).alias("value_string"),
    )

def get_observation(base: DataFrame, views: dict) -> DataFrame:

    first_encounter = (
        base.withColumn(
            "rank",
            F.dense_rank().over(
                Window.partitionBy("patient_id").orderBy("visit_start_date")
            ),
        ).where(F.col("rank") == 1)
    ).drop("rank")
    observation = views["Observation"]
    if "encounter_id" not in observation.columns:
        observation = observation.withColumn(
            "encounter_id", F.lit(None).cast(LongType())
        )
    no_encounter_id_observation = (
        observation.where(observation.encounter_id.isNull())
        .join(first_encounter, on="patient_id")
        .drop(observation.encounter_id, first_encounter.patient_id)
    )
    has_encounter_id_observation = (
        observation.where(observation.encounter_id.isNotNull())
        .join(base, on="encounter_id")
        .drop(base.encounter_id, base.patient_id)
    )
    observation = no_encounter_id_observation.union(has_encounter_id_observation)

    return observation.select(
        observation.patient_id,
        observation.encounter_id,
        observation.visit_provider_id,
        observation.visit_start_date.alias("event_time").cast(DateType()),
        observation.code,
        observation.system,
        F.lit("observation").alias("type"),
        observation.description_name,
        observation.visitType,
        observation.visit_type_code,
        observation.account_id,
        observation.value_date_time.cast(DateType()),
        F.lit(None).alias("value_string"),
    ).distinct()
    

    
    
