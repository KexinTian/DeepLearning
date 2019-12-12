# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for run_experiment.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys




from tensorflow import gfile



class Logger(object):
  """Logging object to write to file and stdout."""

  def __init__(self, filename):
    self.terminal = sys.stdout
    self.log = gfile.GFile(filename, "w")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)

  def flush(self):
    self.terminal.flush()

  def flush_file(self):
    self.log.flush()

