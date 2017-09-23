#include "experimentalSetup.h"

Target targetFromYAML(YAML::Node node) {
  Vector2d xy = vector2dFromYAML(node);
  int coupling = 1;
  double value = 1;

  if (node["coupling"]) {
    coupling = node["coupling"].as<int>();
  }

  if (node["value"]) {
    value = node["value"].as<double>();
  }

  Target t(xy, value, coupling);
  return t;
}

// - x:
//   y:
// - x:
std::vector<Target> targetsFromYAML(YAML::Node node) {
  std::vector<Target> targets;

  for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
    YAML::Node targetNode = it->as<YAML::Node>();
    targets.push_back(targetFromYAML(targetNode));
  }

  return targets;
}

Vector2d vector2dFromYAML(YAML::Node node) {
  Vector2d xy;
  if (node["x"] && node["y"]) {
    xy(0) = node["x"].as<double>();
    xy(1) = node["y"].as<double>();
  }

  return xy;
}

std::vector<Vector2d> vector2dsFromYAML(YAML::Node node) {
  std::vector<Vector2d> targets;

  for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
    YAML::Node targetNode = it->as<YAML::Node>();
    targets.push_back(vector2dFromYAML(targetNode));
  }

  return targets;
}

size_t size_tFromYAML(YAML::Node node, string key) {
  return fromYAML<size_t>(node, key);
}

int intFromYAML(YAML::Node node, string key) {
  return fromYAML<int>(node, key);
}

bool boolFromYAML(YAML::Node node, string key) {
  return fromYAML<bool>(node, key);
}

string stringFromYAML(YAML::Node node, string key) {
  return fromYAML<string>(node, key);
}

double doubleFromYAML(YAML::Node node, string key) {
  return fromYAML<double>(node, key);
}

YAML::Node nodeFromYAML(YAML::Node node, string key) {
  return fromYAML<YAML::Node>(node, key);
}
