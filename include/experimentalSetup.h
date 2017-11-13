/*******************************************************************************
experimentalSetup.h

Wrapper for useful yaml parsing utilities. Also practice with C templates. 
Similar to (a -> a)'s a

Authors: Eric Klinkhammer

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*******************************************************************************/

#ifndef _EXPERIMENTAL_SETUP_H
#define _EXPERIMENTAL_SETUP_H

#include "yaml-cpp/yaml.h"
#include <vector>
#include <Eigen/Eigen>
#include <string>
#include <iostream>
#include "Domains/Target.h"
#include "Domains/Objective.h"
#include "Domains/G.h"
#include "Domains/TeamForming.h"
#include "yaml_constants.h"

using std::vector;
using std::string;

Target targetFromYAML(YAML::Node);
std::vector<Target> targetsFromYAML(YAML::Node);

Vector2d vector2dFromYAML(YAML::Node);
std::vector<Vector2d> vector2dsFromYAML(YAML::Node);

size_t size_tFromYAML(YAML::Node, string);
int intFromYAML(YAML::Node, string);
bool boolFromYAML(YAML::Node, string);
string stringFromYAML(YAML::Node, string);
double doubleFromYAML(YAML::Node, string);
YAML::Node nodeFromYAML(YAML::Node, string);

Objective* objFromYAML(YAML::Node, string);

template<class T> T fromYAML(const YAML::Node node, string s) {
  T result;
  result = node[s].as<T>();
  return result;
}

template<class T> T fromYAML(YAML::Node node, vector<string> keys) {
  YAML::Node currentNode = node;
  for (auto& key : keys) {
    currentNode.reset(currentNode[key]);
  }

  T result;
  result = currentNode.as<T>();
  return result;
}

#endif
