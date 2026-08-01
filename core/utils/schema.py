"""
This contains schemas the program uses to accept and respond to requests

If the format is not in the schema described here the program will throw an error
"""

import base64
import binascii
import uuid
from uuid import UUID   
from pydantic import BaseModel, field_validator, PrivateAttr, model_validator, computed_field
from pydantic_core import PydanticCustomError
from datetime import datetime

class Request(BaseModel):
    site_id: str
    order_id: str
    image_base64: str
    @field_validator('image_base64', mode='after')
    @classmethod
    def validate_base64_image(cls, v):
        if not v or not v.strip():
            raise PydanticCustomError('empty_value', "base64 image must not be empty")
        try:
            base64.b64decode(v, validate=True)
            return v
        except binascii.Error:
            raise PydanticCustomError('invalid_base64', "Invalid base64 string")

    @field_validator('site_id', 'order_id', mode='after')
    @classmethod
    def validate_site_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise PydanticCustomError(
                "empty_value",
                "site_id must not be empty"
            )
        return v

    @field_validator('order_id', mode='after')
    @classmethod
    def validate_order_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise PydanticCustomError(
                "empty_value",
                "order_id must not be empty"
            )
        return v

class Response(BaseModel):
    site_id: str
    order_id: str
    pof: str
    certainty: float
    _created_at: str = PrivateAttr()
    _modified_at: str = PrivateAttr()
    _task_id: UUID = PrivateAttr()

    @model_validator(mode='after')
    def _update_timestamps(self) -> 'Response':
        """
        Initializes or updates timestamps.
        - On first validation, sets created_at, modified_at, and task_id.
        - On subsequent validations (on field modification), updates modified_at
          if specific fields have been changed.
        """
        if not hasattr(self, '_created_at'):
            # First initialization
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._created_at = now_str
            self._modified_at = now_str
            self._task_id = uuid.uuid4()
        elif self.model_fields_set.intersection({'order_id', 'site_id', 'pof'}):
            # Instance already exists and a tracked field was modified
            self._modified_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self

    #@computed_field
    @property
    def created_at(self) -> str:
        return self._created_at

    #@computed_field
    @property
    def modified_at(self) -> str:
        return self._modified_at

    @property
    def task_id(self) -> UUID:
        return self._task_id