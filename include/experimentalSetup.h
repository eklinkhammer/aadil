#include "yaml-cpp/yaml.h"
#include <vector>
#include <Eigen/Eigen>

#include "Domains/Target.h"

using std::vector;

Target targetFromYAML(YAML::Node);
std::vector<Target> targetsFromYAML(YAML::Node);

Vector2d vector2dFromYAML(YAML::Node);
std::vector<Vector2d> vector2dsFromYAML(YAML::Node);
