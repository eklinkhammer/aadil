/*******************************************************************************
Main file

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

#include "experimentUtil.h"



using std::vector ;
using std::string ;
using namespace Eigen ;

int readTrialNum() {
  int trialNum;
  std:: cout << "Please enter trial number [NOTE: no checks enabled to prevent"
	     << " overwriting existing files, user must make sure trial number "
	     << "is unique]: ";
  std::cin >> trialNum;
  return trialNum;
}

void makeDir(std::string dir) {
  std::string command = "mkdir -p " + dir;
  system(command.c_str());
}

void testDomainOnce(MultiRover* domain, bool output) {
  domain->setVerbose(output);
  domain->InitialiseEpoch();
  domain->ResetEpochEvals();
  //  domain->SimulateEpoch(false);
}

int main() {
  YAML::Node config = YAML::LoadFile("../input/config.yaml");


  
  int trialNum = 1;//readTrialNum();
  string fileDir = "Results/" + std::to_string(trialNum);
  makeDir(fileDir);

  vector< vector<size_t>> inds;
  vector< vector< NeuralNet* > > teams;
  
  MultiRover* domain;
  Objective* o;
  vector< Objective* > objs;
  YAML::Node root = nodeFromYAML(config, "control");
  o = objFromYAML(root, objectiveS);
  objs.push_back(o);
  
  YAML::Node experiments = nodeFromYAML(config, "experiments");
  vector<std::string> experimentStrings;


  trainingCurves(nodeFromYAML(config, "control"), 1000, 1, 10);
  
  for (YAML::const_iterator it = experiments.begin(); it != experiments.end(); ++it) {
    std::string experimentName = it->as<std::string>();
    experimentStrings.push_back(experimentName);
  }
  
  VectorXd input;
  for (auto& expKey : experimentStrings) {
    YAML::Node expNode = nodeFromYAML(config, expKey);
    //trainingCurves(expNode, 1000, 5, 10);
    o = objFromYAML(expNode, objectiveS);
    teams.push_back(bestTeamForObjective(expNode, expKey, domain, o, fileDir));
    vector<size_t> ind = fromYAML<vector<size_t>>(expNode, "ind");
    inds.push_back(ind);
    objs.push_back(o);
  } // for each experiment


  root = nodeFromYAML(config, "control");
  G g(3,4,1);
  o = &g;//objFromYAML(root, objectiveS);
  //trainingCurves(root, 1000, 5, 10);

  Alignments as(objs, 10, 0.4);
  as.addAlignments(1);

  // Alignment Domain
  // Get variables from node to construct domain
  size_t nRovs  = size_tFromYAML(root, nRovsS);
  size_t nPOIs  = size_tFromYAML(root, nPOIsS);
  size_t nSteps = size_tFromYAML(root, nStepsS);

  string type = stringFromYAML(root, typeS);

  double xmin = fromYAML<double>(root, xminS);
  double ymin = fromYAML<double>(root, yminS);
  double xmax = fromYAML<double>(root, xmaxS);
  double ymax = fromYAML<double>(root, ymaxS);
  std::vector<double> world = {xmin, xmax, ymin, ymax};

  vector< Agent* > roverTeam;
  for (size_t i = 0; i < nRovs; i++) {
    vector<NeuralNet*> netsAgent;
    for (size_t j = 0; j < teams.size(); j++) {
      int netI = teams[j].size() > i ? i : 0;
      netsAgent.push_back(teams[j][netI]);
    }

    roverTeam.push_back(new AlignmentAgent(netsAgent, &as, inds));
  }
  
  std::cout << "Creating domain with alignment agents." << std::endl;
  MultiRover domainD(world, nSteps, nPOIs, nRovs, roverTeam);

  int biasStart = intFromYAML(root, biasStartS);
  if (biasStart == 0) {
    domainD.setBias(false);
  }

  //as.addAlignments(300);
  
  vector< size_t > agentsForSim(nRovs, 0);
  for (int m = 1; m < 20; m++) {
    //std::cout << "Adding alignment values to map... " << m*50 << std::endl;
    as.addAlignments(50);
    domainD.setVerbose(false);
    std::vector<double> alignScores;

    //roverTeam[0]->setOutputBool(true);
    for (int reps = 0; reps < 100; reps++) {
      //std::cout << "New Epoch..." << std::endl;
      domainD.InitialiseEpoch();
      alignScores.push_back(domainD.runSim(domainD.createSim(domainD.getNPop()),
  					   agentsForSim, o));
      domainD.ResetEpochEvals();
    }

    std::cout << m*50 << " " << mean(alignScores) << " "
  	      << stddev(alignScores) << " " << statstderr(alignScores) << std::endl;
  }
  return 0 ;
}
