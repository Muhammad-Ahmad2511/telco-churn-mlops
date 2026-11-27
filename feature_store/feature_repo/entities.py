"""Feast entity definitions."""

from feast import Entity, ValueType

customer = Entity(
    name="customer",
    join_keys=["customerID"],
    value_type=ValueType.STRING,
    description="A telecom customer. customerID is a stable key and is not a model input.",
    tags={"owner": "data-ml-engineering"},
)
