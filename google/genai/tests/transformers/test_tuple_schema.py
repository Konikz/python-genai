# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Tests tuple schema handling in the _transformers module."""

import pydantic
import pytest

from ... import _transformers
from ... import client as google_genai_client_module
from ... import types


class ModelWithTuples(pydantic.BaseModel):
    fixed_tuple: tuple[int, str]
    var_tuple: tuple[int, ...]
    nested_tuple: tuple[tuple[int, str], tuple[float, bool]]


@pytest.fixture
def client():
    # Create a client with a dummy API key for testing
    return google_genai_client_module.Client(api_key="dummy-api-key")


def test_fixed_length_tuple_schema(client):
    """Tests schema generation for fixed-length tuples."""
    schema = _transformers.t_schema(client, ModelWithTuples)
    
    # Check fixed_tuple schema
    fixed_tuple_schema = schema.properties['fixed_tuple']
    assert fixed_tuple_schema.type == types.Type.ARRAY
    assert fixed_tuple_schema.items.type == types.Type.OBJECT
    assert fixed_tuple_schema.items.properties['0'].type == types.Type.INTEGER
    assert fixed_tuple_schema.items.properties['1'].type == types.Type.STRING
    assert fixed_tuple_schema.items.required == ['0', '1']


def test_variable_length_tuple_schema(client):
    """Tests schema generation for variable-length tuples."""
    schema = _transformers.t_schema(client, ModelWithTuples)
    
    # Check var_tuple schema
    var_tuple_schema = schema.properties['var_tuple']
    assert var_tuple_schema.type == types.Type.ARRAY
    assert var_tuple_schema.items.type == types.Type.INTEGER


def test_nested_tuple_schema(client):
    """Tests schema generation for nested tuples."""
    schema = _transformers.t_schema(client, ModelWithTuples)
    
    # Check nested_tuple schema
    nested_tuple_schema = schema.properties['nested_tuple']
    assert nested_tuple_schema.type == types.Type.ARRAY
    assert nested_tuple_schema.items.type == types.Type.OBJECT
    
    # First tuple
    first_tuple = nested_tuple_schema.items.properties['0']
    assert first_tuple.type == types.Type.OBJECT
    assert first_tuple.properties['0'].type == types.Type.INTEGER
    assert first_tuple.properties['1'].type == types.Type.STRING
    assert first_tuple.required == ['0', '1']
    
    # Second tuple
    second_tuple = nested_tuple_schema.items.properties['1']
    assert second_tuple.type == types.Type.OBJECT
    assert second_tuple.properties['0'].type == types.Type.NUMBER
    assert second_tuple.properties['1'].type == types.Type.BOOLEAN
    assert second_tuple.required == ['0', '1'] 