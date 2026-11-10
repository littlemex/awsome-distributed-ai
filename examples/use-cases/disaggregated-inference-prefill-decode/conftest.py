# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""This lab's tests use paths relative to this directory, and pytest does not change into it."""
import os
import pytest

LAB_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def _lab_cwd():
    previous = os.getcwd()
    os.chdir(LAB_DIR)
    try:
        yield
    finally:
        os.chdir(previous)
